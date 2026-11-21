"""Live director channel: the owner talks to their own coach while the match keeps running.

Quick commands (buttons / short phrases) patch the strategy immediately with
no model call. Anything else is interpreted by the coach's LLM in the
background; the bot keeps fighting with the current strategy until the answer
arrives. Every change is applied between ticks and logged, so replays stay exact.

The coach always carries out the director's intent; disagreement goes into a
separate opinion field.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from dataclasses import dataclass, field, replace

from arena.coaching import extract_json_object
from arena.events.log import EventLog, truncate
from arena.tanks import rules as R
from arena.tanks.coach import build_coach_prompt
from arena.tanks.strategy import STRATEGY_SCHEMA, Strategy, StrategyError, parse_strategy

MAX_LLM_COMMANDS = 20
TEMP_RETREAT_SECONDS = 10.0
SHORT_PHRASE = 10  # longer messages always go to the LLM


def _clamp(v: float) -> float:
    return round(min(1.0, max(0.0, v)), 2)


# id -> (button label, keywords, patch(current) -> changed fields)
QUICK: dict[str, tuple[str, tuple[str, ...], Callable[[Strategy], dict]]] = {
    "attack": ("공격적으로", ("공격", "압박", "밀어붙", "들어가", "잡아"),
               lambda s: {"aggression": _clamp(max(s.aggression + 0.3, 0.75)),
                          "engage_range": _clamp(s.engage_range - 0.2), "seek": _clamp(max(s.seek, 0.8))}),
    "careful": ("신중하게", ("신중", "조심", "무리하지"),
                lambda s: {"aggression": _clamp(min(s.aggression - 0.3, 0.3)),
                           "retreat_hp": _clamp(s.retreat_hp + 0.15), "seek": 0.0}),
    "distance": ("거리 벌려", ("거리", "멀리서"),
                 lambda s: {"engage_range": _clamp(s.engage_range + 0.3), "strafe": _clamp(s.strafe + 0.2)}),
    "retreat": ("후퇴 (10초)", ("후퇴", "도망", "빠져"),
                lambda s: {"retreat_hp": 1.0}),
    "farm": ("농사해", ("농사", "파밍", "도형"),
             lambda s: {"aggression": _clamp(s.aggression - 0.4), "farm_target": "valuable", "seek": 0.0}),
    "center": ("중앙으로", ("중앙", "가운데"),
               lambda s: {"center_bias": _clamp(s.center_bias + 0.4), "zone_pull": _clamp(max(s.zone_pull, 0.8))}),
    "edge": ("외곽으로", ("외곽", "구석", "가장자리"),
             lambda s: {"center_bias": _clamp(s.center_bias - 0.4), "zone_pull": _clamp(max(s.zone_pull, 0.8))}),
}
TEMPORARY = {"retreat": TEMP_RETREAT_SECONDS}


def classify(text: str) -> str | None:
    """Return a quick-command id for a button label or a short unambiguous phrase, else None (use the LLM)."""
    t = text.strip()
    for qid, (label, _, _) in QUICK.items():
        if t == label or t == qid:
            return qid
    if len(t) > SHORT_PHRASE:
        return None
    hits = {qid for qid, (_, words, _) in QUICK.items() if any(w in t for w in words)}
    return hits.pop() if len(hits) == 1 else None


INTERPRET_SCHEMA = {
    "type": "object",
    "properties": {
        "strategy": STRATEGY_SCHEMA,
        "interpretation": {"type": "string"},
        "opinion": {"type": "string"},
        "feasible": {"type": "boolean"},
    },
    "required": ["strategy", "interpretation", "opinion", "feasible"],
    "additionalProperties": False,
}


def build_interpret_prompt(briefing, instruction: str, history: list[str]) -> str:
    base = build_coach_prompt(briefing).rsplit("\n\n", 1)[0]  # drop the generic reply instruction
    parts = [
        base,
        "Your owner (the director) is watching the match live and just told you:",
        f"\"{instruction}\"",
    ]
    if history:
        parts.append("Earlier director instructions this match (oldest first):\n" + "\n".join(history))
    parts.append(
        "Carry out the director's intent as closely as the strategy fields allow, even if you disagree - the "
        "director has the final say. Change only what the instruction needs. If it asks for something this game "
        "cannot do, pick the closest possible strategy, set feasible to false and explain. Put any disagreement or "
        "warning in `opinion` (Korean, one or two sentences). `interpretation` says in Korean what you changed.\n"
        f"You have about {briefing.seconds:g} seconds; the match keeps running meanwhile. Do not run commands or read files.\n"
        'Reply with ONLY a JSON object: {"strategy": {...all strategy fields...}, "interpretation": "...", '
        '"opinion": "...", "feasible": true}'
    )
    return "\n\n".join(parts)


def parse_interpretation(text: str) -> tuple[Strategy, str, str, bool]:
    data = extract_json_object(text)
    raw_strategy = data.get("strategy", data if "build" in data else None)
    strategy = parse_strategy(raw_strategy)
    return (strategy, str(data.get("interpretation", ""))[:300], str(data.get("opinion", ""))[:300],
            bool(data.get("feasible", True)))


@dataclass
class Command:
    id: int
    side: int
    text: str
    kind: str  # "quick" | "llm"
    tick: int
    quick_id: str | None = None
    submitted: float = field(default_factory=time.monotonic)


@dataclass
class Ready:
    command: Command
    changes: dict  # field -> new value, applied on top of the strategy current at apply time
    revert: bool = False


class Director:
    def __init__(self, side: int, agent=None, max_llm: int = MAX_LLM_COMMANDS, timeout: float = 30.0):
        self.side = side
        self.agent = agent  # CLI agent with complete(prompt, schema); None = quick commands only
        self.max_llm = max_llm
        self.timeout = timeout
        self.log: EventLog | None = None
        self._tick: Callable[[], int] = lambda: 0
        self._briefing: Callable[[], object] | None = None
        self.reset()

    # ---- lifecycle (called by the match runner) ----
    def reset(self) -> None:
        self.next_id = 1
        self.llm_used = 0
        self.ready: list[Ready] = []
        self.reverts: list[tuple[int, Command, dict]] = []  # (tick, command, {field: (old, temp)})
        self.history: list[str] = []
        self.task: asyncio.Task | None = None
        self.pending: Command | None = None

    def bind(self, log: EventLog, tick: Callable[[], int], briefing: Callable[[], object]) -> None:
        self.reset()
        self.log, self._tick, self._briefing = log, tick, briefing

    def finish(self) -> None:
        if self.task and not self.task.done():
            self.task.cancel()
        self.log = None

    @property
    def active(self) -> bool:
        return self.log is not None

    # ---- input ----
    def submit(self, text: str) -> dict:
        text = (text or "").strip()[:300]
        if not self.active:
            return {"ok": False, "error": "no match is running"}
        if not text:
            return {"ok": False, "error": "empty command"}
        qid = classify(text)
        kind = "quick" if qid else "llm"
        if kind == "llm" and self.agent is None:
            return {"ok": False, "error": "this side has no LLM coach; use a quick command"}
        if kind == "llm" and self.llm_used >= self.max_llm:
            return {"ok": False, "error": f"LLM command limit reached ({self.max_llm} per match)"}
        cmd = Command(self.next_id, self.side, text, kind, self._tick(), qid)
        self.next_id += 1
        self.history.append(f"[{_clock(cmd.tick)}] {text}")
        self.log.emit("director_command", id=cmd.id, side=self.side, text=text, kind=kind, quick_id=qid, tick=cmd.tick)
        if kind == "quick":
            self.ready.append(Ready(cmd, {}))  # patch computed at apply time from the then-current strategy
        else:
            self.llm_used += 1
            if self.task and not self.task.done():
                self.task.cancel()  # kills the CLI process; the newer instruction wins
                self._result(self.pending, ok=False, error=f"superseded by command #{cmd.id}")
            self.pending = cmd
            self.task = asyncio.create_task(self._interpret(cmd))
        return {"ok": True, "id": cmd.id, "kind": kind, "quick_id": qid}

    def notes(self) -> str | None:
        """Recent director instructions, for the coach's own timeouts (owner_notes)."""
        if not self.history:
            return None
        return ("The director (your owner) gave these instructions during the match; keep honouring them "
                "unless a newer one replaces them:\n" + "\n".join(self.history[-8:]))

    # ---- interpretation ----
    async def _interpret(self, cmd: Command) -> None:
        briefing = self._briefing()
        base = Strategy(**{**briefing.current, "build": tuple(briefing.current["build"])})
        prompt = build_interpret_prompt(briefing, cmd.text, self.history[:-1])
        t0 = time.monotonic()
        try:
            completion = await asyncio.wait_for(self.agent.complete(prompt, INTERPRET_SCHEMA), self.timeout)
        except asyncio.CancelledError:
            raise
        except TimeoutError:
            return self._result(cmd, ok=False, error=f"no answer within {self.timeout:g}s", t0=t0)
        except Exception as exc:
            return self._result(cmd, ok=False, error=f"coach error: {type(exc).__name__}: {exc}", t0=t0)
        if completion.error:
            return self._result(cmd, ok=False, error=completion.error, raw=completion.raw, t0=t0)
        try:
            strategy, interpretation, opinion, feasible = parse_interpretation(completion.text)
        except (ValueError, StrategyError) as exc:
            return self._result(cmd, ok=False, error=str(exc), raw=completion.text, t0=t0)
        # Only what the coach changed relative to what it saw is applied, on top of the strategy current
        # at apply time (a timeout may have changed other fields in the meantime).
        new, old = strategy.to_dict(), base.to_dict()
        changes = {k: v for k, v in new.items() if old.get(k) != v}
        changes["build"] = tuple(changes["build"]) if "build" in changes else None
        if changes["build"] is None:
            del changes["build"]
        self.pending = None
        self.ready.append(Ready(cmd, changes))
        self._result(cmd, ok=True, raw=completion.text, t0=t0, interpretation=interpretation, opinion=opinion,
                     feasible=feasible, changes={k: (list(v) if isinstance(v, tuple) else v) for k, v in changes.items()})

    def _result(self, cmd: Command | None, ok: bool, t0: float | None = None, raw: str = "", **extra) -> None:
        if cmd is None or self.log is None:
            return
        latency = round((time.monotonic() - (t0 if t0 is not None else cmd.submitted)) * 1000)
        self.log.emit("director_result", id=cmd.id, side=cmd.side, ok=ok, latency_ms=latency,
                      raw=truncate(raw), **extra)

    # ---- application (called by the runner between ticks) ----
    def drain(self, tick: int, current: Strategy) -> list[tuple[Command, Strategy, bool]]:
        """Changes to apply before `tick`, in order, each as a full new strategy."""
        out = []
        for due, cmd, fields in list(self.reverts):
            if due <= tick:
                self.reverts.remove((due, cmd, fields))
                # Undo a temporary change unless something else has changed that field since.
                undo = {k: old for k, (old, temp) in fields.items() if getattr(current, k) == temp}
                if undo:
                    current = replace(current, **undo)
                    out.append((cmd, current, True))
        for item in self.ready:
            if item.command.kind == "quick":
                qid = item.command.quick_id
                changes = QUICK[qid][2](current)
                if qid in TEMPORARY:
                    due = tick + round(TEMPORARY[qid] * R.TICK_RATE)
                    self.reverts.append((due, item.command, {k: (getattr(current, k), v) for k, v in changes.items()}))
                self._result(item.command, ok=True, interpretation=QUICK[qid][0], changes=changes)
            else:
                changes = item.changes
            current = replace(current, **changes)
            out.append((item.command, current, False))
        self.ready = []
        return out


def _clock(tick: int) -> str:
    s = int(tick * R.DT)
    return f"{s // 60}:{s % 60:02d}"


def quick_buttons() -> list[dict]:
    return [{"id": qid, "label": label} for qid, (label, _, _) in QUICK.items()]


__all__ = ["Director", "QUICK", "classify", "quick_buttons", "parse_interpretation", "INTERPRET_SCHEMA"]
