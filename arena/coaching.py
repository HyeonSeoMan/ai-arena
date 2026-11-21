"""Game-independent coaching: ask every coach at once, apply valid answers, log everything.

A game supplies briefings (what each coach may know) and a parser that turns
the model's text into that game's strategy object (anything with to_dict()).
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

from arena.events.log import EventLog, truncate


@dataclass(frozen=True)
class CoachReply:
    raw: str
    strategy: Any | None = None
    error: str | None = None
    meta: dict = field(default_factory=dict)


class Coach(Protocol):
    name: str

    async def plan(self, briefing: Any) -> CoachReply: ...


def extract_json_object(text: str) -> dict:
    """Parse the outermost {...} in a model answer, tolerating prose or code fences around it."""
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("no JSON object found")
    try:
        data = json.loads(text[start : end + 1])
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON: {exc.msg}") from exc
    if not isinstance(data, dict):
        raise ValueError("answer must be a JSON object")
    return data


def reply_from_text(text: str, parse_strategy: Callable[[dict], Any]) -> CoachReply:
    try:
        return CoachReply(raw=text, strategy=parse_strategy(extract_json_object(text)))
    except ValueError as exc:  # includes each game's StrategyError
        return CoachReply(raw=text, error=str(exc))


class LLMCoach:
    """Wraps a CLI agent exposing `complete(prompt, schema)` with a game's prompt and parser."""

    def __init__(self, agent, build_prompt: Callable[[Any], str], parse_strategy: Callable[[dict], Any],
                 schema: dict, name: str | None = None):
        self.agent = agent
        self.build_prompt = build_prompt
        self.parse_strategy = parse_strategy
        self.schema = schema
        self.name = name or agent.name

    async def plan(self, briefing: Any) -> CoachReply:
        completion = await self.agent.complete(self.build_prompt(briefing), self.schema)
        if completion.error:
            return CoachReply(raw=completion.raw, error=completion.error)
        reply = reply_from_text(completion.text, self.parse_strategy)
        return CoachReply(reply.raw, reply.strategy, reply.error, completion.meta)


async def consult(
    coaches: Sequence[Coach],
    briefings: Sequence[Any],
    strategies: list,
    log: EventLog,
    seconds: float,
    phase: str,
    clock: Callable[[], float] = time.monotonic,
    **context: Any,
) -> None:
    """Run one timeout: every coach is asked in parallel; `strategies[i]` is replaced on a valid answer.

    A timeout, crash or invalid answer keeps the previous strategy (no forfeits).
    """
    log.emit("timeout_started", phase=phase, seconds=seconds, **context)

    async def ask(side: int) -> None:
        t0 = clock()
        try:
            reply = await asyncio.wait_for(coaches[side].plan(briefings[side]), seconds)
        except TimeoutError:
            log.emit("strategy_kept", side=side, reason=f"no answer within {seconds:g}s",
                     elapsed_ms=round((clock() - t0) * 1000), raw="")
            return
        except Exception as exc:
            reply = CoachReply(raw="", error=f"coach error: {type(exc).__name__}: {exc}")
        elapsed_ms = round((clock() - t0) * 1000)
        if reply.strategy is None:
            log.emit("strategy_kept", side=side, reason=reply.error or "no strategy",
                     elapsed_ms=elapsed_ms, raw=truncate(reply.raw))
            return
        strategies[side] = reply.strategy
        log.emit("strategy_set", side=side, strategy=reply.strategy.to_dict(),
                 elapsed_ms=elapsed_ms, raw=truncate(reply.raw))

    await asyncio.gather(*(ask(side) for side in range(len(coaches))))
    log.emit("timeout_ended", phase=phase)
