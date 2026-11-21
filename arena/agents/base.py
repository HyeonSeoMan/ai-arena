"""Common agent interface.

An agent receives an Observation (public game state only) and returns an
AgentReply. Extracting the move from the agent's raw output is the
adapter's job; judging legality is the engine's job.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Protocol


@dataclass(frozen=True)
class Observation:
    match_id: str
    turn: int
    color: int
    board: list[list[int]]
    board_text: str
    moves: list[tuple[int, int, int]]  # (player, row, col)
    attempt: int = 1
    last_error: str | None = None  # why the previous attempt this turn was rejected
    owner_notes: str | None = None  # reserved for owner -> own agent channel (MVP 2+)


@dataclass(frozen=True)
class AgentReply:
    raw: str
    row: object = None
    col: object = None
    error: str | None = None  # set when no move could be extracted from `raw`
    meta: dict = field(default_factory=dict)


class Agent(Protocol):
    name: str

    async def act(self, obs: Observation) -> AgentReply: ...


_JSON_OBJECT = re.compile(r"\{[^{}]*\}")


def parse_move(raw: str) -> AgentReply:
    """Extract the first JSON object containing `row` and `col` from free-form text."""
    for match in _JSON_OBJECT.finditer(raw):
        try:
            data = json.loads(match.group(0))
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict) and "row" in data and "col" in data:
            return AgentReply(raw=raw, row=data["row"], col=data["col"])
    return AgentReply(raw=raw, error="no JSON object with 'row' and 'col' found")
