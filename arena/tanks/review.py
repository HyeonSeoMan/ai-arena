"""Post-match review: the director's coach writes a report, then the director can keep talking to it.

The coach only sees its own side plus public events (kills, class changes,
scoreboard); the opponent's strategy stays hidden even after the match.
Report and conversation are saved next to the match log as `<id>.review.json`.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

from arena.events.log import Event
from arena.tanks import rules as R
from arena.tanks.analysis import director_effects

MAX_TURNS = 20


def _t(tick: int) -> str:
    s = int(tick * R.DT)
    return f"{s // 60}:{s % 60:02d}"


def match_context(events: list[Event], side: int) -> dict:
    """Everything the director's coach may know about the finished match, as plain JSON."""
    start = next(e for e in events if e["type"] == "match_started")
    end = next(e for e in reversed(events) if e["type"] == "match_ended")
    names = [start["left"], start["right"]]
    me, opp = names[side], names[1 - side]

    def who(i):
        return "you" if i == side else "opponent" if i in (0, 1) else "shape" if i is None else "NPC"

    timeline = []
    for e in events:
        if e["type"] == "kill" and (e["victim_ai"] or e["killer"] == side):
            timeline.append(f"{_t(e['tick'])} {who(e['killer'])} killed {who(e['victim'])}")
        elif e["type"] == "class_changed":
            timeline.append(f"{_t(e['tick'])} {who(e['tank'])} became {e['cls']}")
        elif e["type"] == "timeout_started" and e["phase"] != "pregame":
            timeline.append(f"{_t(round(e['match_time'] * R.TICK_RATE))} coach timeout ({e['phase']})")
        elif e["type"] == "strategy_set" and e["side"] == side:
            timeline.append(f"    your coach set: {e['strategy']['note']}")
    effects = [eff for eff in director_effects(events) if eff["command_id"] is not None]
    failed = [{"text": c["text"], "error": r.get("error")} for c in events if c["type"] == "director_command"
              for r in events if r["type"] == "director_result" and r["id"] == c["id"] and not r["ok"]]
    return {
        "you": me,
        "opponent": opp,
        "result": "win" if end["winner"] == side else "loss",
        "reason": end["reason"],
        "duration": _t(end["tick"]),
        "kills": {"you": end["kills"][side], "opponent": end["kills"][1 - side]},
        "scores": {"you": end["scores"][side], "opponent": end["scores"][1 - side]},
        "levels": {"you": end["levels"][side], "opponent": end["levels"][1 - side]},
        "timeline": timeline,
        "director_commands": effects,
        "failed_commands": failed,
        "notes": ("Each director command's `effect` compares the next seconds with a counterfactual where the "
                  "command was not given (same match state, deterministic simulation): `with`, `without`, `delta`."),
    }


REPORT_INSTRUCTIONS = """You are the coach of a tank in Coach Tanks and the match just ended. Your owner, the director,
gave you live instructions during the match. Write the post-match review for the director in Korean.

Cover, briefly and concretely, using only the data below:
1. 결과 요약 (승패, 결정적 순간)
2. 감독 지시별 효과: 각 지시가 반사실 대비 실제로 무엇을 바꿨는지 (거리, 교전 시간, 피해, 경험치). 효과가 작거나
   역효과였으면 솔직하게 말하기. 지시가 없었다면 이 항목은 "감독 지시 없음".
3. 코치(당신)의 판단 중 잘한 점과 잘못한 점
4. 다음 경기를 위한 제안 2~3개 (빌드, 행동 방침, 지시 타이밍)

Do not invent numbers that are not in the data. Plain text, no code blocks, under about 350 Korean words."""

CHAT_INSTRUCTIONS = """You are the coach continuing the post-match review with your director (in Korean). Answer the
director's latest message using the match data and your report. Be concrete and honest; say so when the data cannot
answer something. Keep replies short (a few sentences) unless asked for detail. Do not run commands or read files."""


class Review:
    def __init__(self, agent, side: int, path: Path, notify: Callable[[str, dict], None] | None = None):
        self.agent = agent  # CLI agent with complete(prompt, schema)
        self.side = side
        self.path = path
        self.notify = notify or (lambda type_, data: None)
        self.context: dict | None = None
        self.report: str | None = None
        self.messages: list[dict] = []
        self.busy = False

    def state(self) -> dict:
        return {"report": self.report, "messages": self.messages, "busy": self.busy, "ready": self.report is not None}

    async def start(self, events: list[Event]) -> str:
        self.context = match_context(events, self.side)
        self.busy = True
        self.notify("review_started", {})
        prompt = "\n\n".join([REPORT_INSTRUCTIONS, "Match data:", json.dumps(self.context, ensure_ascii=False)])
        completion = await self.agent.complete(prompt, None)
        self.report = completion.text.strip() if not completion.error else f"(보고서 생성 실패: {completion.error})"
        self.busy = False
        self._save()
        self.notify("review_report", {"text": self.report})
        return self.report

    async def chat(self, message: str) -> dict:
        message = (message or "").strip()[:1000]
        if self.report is None:
            return {"ok": False, "error": "the report is not ready yet"}
        if self.busy:
            return {"ok": False, "error": "the coach is still answering"}
        if not message:
            return {"ok": False, "error": "empty message"}
        if sum(m["role"] == "director" for m in self.messages) >= MAX_TURNS:
            return {"ok": False, "error": f"review conversation limit reached ({MAX_TURNS})"}
        self.messages.append({"role": "director", "text": message})
        self.notify("review_message", self.messages[-1])
        self.busy = True
        history = "\n".join(f"{'Director' if m['role'] == 'director' else 'Coach'}: {m['text']}" for m in self.messages)
        prompt = "\n\n".join([CHAT_INSTRUCTIONS, "Match data:", json.dumps(self.context, ensure_ascii=False),
                              "Your report:", self.report, "Conversation so far:", history, "Coach:"])
        try:
            completion = await self.agent.complete(prompt, None)
        finally:
            self.busy = False
        reply = completion.text.strip() if not completion.error else f"(응답 실패: {completion.error})"
        self.messages.append({"role": "coach", "text": reply})
        self._save()
        self.notify("review_message", self.messages[-1])
        return {"ok": True, "reply": reply}

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        data = {"side": self.side, "context": self.context, "report": self.report, "messages": self.messages}
        self.path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
