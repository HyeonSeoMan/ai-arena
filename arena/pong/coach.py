"""Coaches decide strategies. LLM coaches wrap a CLI agent; dummies run offline."""

from __future__ import annotations

import asyncio
import json
import random
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Protocol

from arena.pong.strategy import AIMS, STRATEGY_SCHEMA, Strategy, StrategyError, parse_strategy


@dataclass(frozen=True)
class Briefing:
    side: int  # 0 = left, 1 = right
    phase: str  # "pregame" or "timeout"
    point: int  # points played so far
    score: tuple[int, int]  # (you, opponent)
    points_to_win: int
    seconds: float  # time limit for this answer
    current: dict  # your strategy in force
    rallies: list[dict]  # recent points from your perspective, newest last
    owner_notes: str | None = None  # reserved for owner -> own coach channel


@dataclass(frozen=True)
class CoachReply:
    raw: str
    strategy: Strategy | None = None
    error: str | None = None
    meta: dict = field(default_factory=dict)


class Coach(Protocol):
    name: str

    async def plan(self, briefing: Briefing) -> CoachReply: ...


def parse_reply(text: str) -> CoachReply:
    """Read a strategy from the model's answer, tolerating surrounding prose or code fences."""
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        return CoachReply(raw=text, error="no JSON object found")
    try:
        data = json.loads(text[start : end + 1])
        return CoachReply(raw=text, strategy=parse_strategy(data))
    except json.JSONDecodeError as exc:
        return CoachReply(raw=text, error=f"invalid JSON: {exc.msg}")
    except StrategyError as exc:
        return CoachReply(raw=text, error=str(exc))


GUIDE = """You are the coach of a paddle bot in a real-time 2D Pong match against another AI's bot.
You cannot move the paddle yourself: your bot plays the points in real time using the strategy you set.
Between points there is a short timeout where you may change the strategy. You only see observable
results; the opponent's strategy is hidden from you.

Field 160 x 90, paddles on the left and right edges, ball bounces off top and bottom walls.
The ball starts at 90/s, gets 5% faster on every hit (max 170/s). Faster balls are harder to meet.
First to {points_to_win} points wins.

Strategy fields (numbers are 0..1; every benefit has a cost):
- aim: where your bot sends the ball. "away" = far side from the opponent paddle, "corners" = random
  corner, "center" = safe middle (easy for the opponent), "random" = mix of the above.
- angle: maximum return angle (15..60 degrees). Wider makes the opponent run more, but mis-hits
  (weak, slow, straight balls) become more likely.
- power: shot speed up to +35%. After a power shot your paddle freezes briefly (up to 0.3s), aim is
  less accurate and mis-hits become more likely.
- effort: paddle top speed. Moving fast drains stamina; low stamina makes the paddle slower.
  Stamina recovers when standing still and partly between points.
- anticipation: 0 = follow the ball's current height, 1 = move early to the predicted landing spot.
  Early predictions are noisy while the ball is far away, which can cause wasted movement.
- home: after returning the ball, 0 = stay where you are, 1 = go back to the centre.
- note: one short sentence in Korean explaining your plan (shown to spectators).

Rally fields: "won" (did you win the point), "end" (how it ended), "hits", "miss_distance" (how far
the loser's paddle edge was from the ball), "your_stamina", and observed shot stats for you and the
opponent: average |angle|, average speed, mis-hits, and where shots went (top/middle/bottom counts)."""


def build_coach_prompt(b: Briefing) -> str:
    situation = {
        "phase": b.phase,
        "points_played": b.point,
        "score": {"you": b.score[0], "opponent": b.score[1]},
        "your_side": "left" if b.side == 0 else "right",
        "your_current_strategy": b.current,
        "recent_rallies": b.rallies,
    }
    parts = [GUIDE.format(points_to_win=b.points_to_win), "Situation:", json.dumps(situation, ensure_ascii=False)]
    if b.owner_notes:
        parts.append(f"Notes from your owner:\n{b.owner_notes}")
    parts.append(
        f"You have {b.seconds:g} seconds. Do not run commands or read files.\n"
        "Reply with ONLY a JSON object with keys aim, angle, power, effort, anticipation, home, note."
    )
    return "\n\n".join(parts)


class LLMCoach:
    """Wraps a CLI agent (ClaudeCodeAgent / CodexAgent) that exposes `complete(prompt, schema)`."""

    def __init__(self, agent, name: str | None = None):
        self.agent = agent
        self.name = name or agent.name

    async def plan(self, briefing: Briefing) -> CoachReply:
        completion = await self.agent.complete(build_coach_prompt(briefing), STRATEGY_SCHEMA)
        if completion.error:
            return CoachReply(raw=completion.raw, error=completion.error)
        reply = parse_reply(completion.text)
        return CoachReply(reply.raw, reply.strategy, reply.error, completion.meta)


class FixedCoach:
    def __init__(self, strategy: Strategy | None = None, name: str = "fixed", delay: float = 0.0):
        self.name = name
        self.strategy = strategy or Strategy(note="기본 작전 유지")
        self.delay = delay

    async def plan(self, briefing: Briefing) -> CoachReply:
        if self.delay:
            await asyncio.sleep(self.delay)
        return CoachReply(raw=json.dumps(self.strategy.to_dict(), ensure_ascii=False), strategy=self.strategy)


class RandomCoach:
    """Picks a fresh random strategy every timeout. Useful for watching variety offline."""

    def __init__(self, name: str = "random", seed: int | None = None, delay: float = 0.0):
        self.name = name
        self.rng = random.Random(seed)
        self.delay = delay

    async def plan(self, briefing: Briefing) -> CoachReply:
        if self.delay:
            await asyncio.sleep(self.delay)
        r = self.rng
        s = Strategy(
            aim=r.choice(AIMS),
            angle=round(r.random(), 2),
            power=round(r.random(), 2),
            effort=round(r.uniform(0.4, 1.0), 2),
            anticipation=round(r.random(), 2),
            home=round(r.random(), 2),
            note="무작위 작전",
        )
        return CoachReply(raw=json.dumps(s.to_dict(), ensure_ascii=False), strategy=s)


class ScriptedCoach:
    """Test helper: each step is raw text, a float (sleep), or an Exception to raise."""

    def __init__(self, steps: Iterable[str | float | Exception], name: str = "scripted"):
        self.name = name
        self._steps = iter(steps)
        self.briefings: list[Briefing] = []

    async def plan(self, briefing: Briefing) -> CoachReply:
        self.briefings.append(briefing)
        step = next(self._steps, '{"aim": "away", "angle": 0.5, "power": 0.3, "effort": 0.7, "anticipation": 0.5, "home": 0.6}')
        while isinstance(step, float):
            await asyncio.sleep(step)
            step = next(self._steps)
        if isinstance(step, Exception):
            raise step
        return parse_reply(step)
