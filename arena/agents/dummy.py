"""Offline agents for development and tests. They never call any model."""

from __future__ import annotations

import asyncio
import json
import random
from collections.abc import Iterable

from arena.agents.base import AgentReply, Observation, parse_move
from arena.engine import EMPTY, SIZE, opponent

_DIRECTIONS = ((0, 1), (1, 0), (1, 1), (1, -1))


def _empty_cells(board: list[list[int]]) -> list[tuple[int, int]]:
    return [(r, c) for r in range(SIZE) for c in range(SIZE) if board[r][c] == EMPTY]


def _reply(row: int, col: int) -> AgentReply:
    return parse_move(json.dumps({"row": row, "col": col}))


class RandomAgent:
    """Plays a uniformly random empty cell."""

    def __init__(self, name: str = "random", seed: int | None = None):
        self.name = name
        self._rng = random.Random(seed)

    async def act(self, obs: Observation) -> AgentReply:
        return _reply(*self._rng.choice(_empty_cells(obs.board)))


class HeuristicAgent:
    """Scores each empty cell by the lines it would extend for itself and block for the opponent."""

    def __init__(self, name: str = "heuristic", seed: int | None = None, delay: float = 0.0):
        self.name = name
        self._rng = random.Random(seed)
        self._delay = delay  # simulated thinking time, useful when spectating

    async def act(self, obs: Observation) -> AgentReply:
        if self._delay:
            await asyncio.sleep(self._delay)
        me, them = obs.color, opponent(obs.color)
        best_score, best = -1, []
        for r, c in _empty_cells(obs.board):
            # Completing own line slightly outranks blocking an equal opponent line.
            score = 1.1 * _line_score(obs.board, r, c, me) + _line_score(obs.board, r, c, them)
            score -= 0.01 * (abs(r - SIZE // 2) + abs(c - SIZE // 2))
            if score > best_score + 1e-9:
                best_score, best = score, [(r, c)]
            elif abs(score - best_score) <= 1e-9:
                best.append((r, c))
        return _reply(*self._rng.choice(best))


def _line_score(board: list[list[int]], row: int, col: int, player: int) -> float:
    score = 0.0
    for dr, dc in _DIRECTIONS:
        run = 1
        for sign in (1, -1):
            r, c = row + sign * dr, col + sign * dc
            while 0 <= r < SIZE and 0 <= c < SIZE and board[r][c] == player:
                run += 1
                r += sign * dr
                c += sign * dc
        score += 10 ** min(run, 5)
    return score


class ScriptedAgent:
    """Replays canned outputs, for testing failure handling.

    Each step is a raw string, a float (sleep that many seconds, then use the
    next step), or an Exception instance to raise.
    """

    def __init__(self, steps: Iterable[str | float | Exception], name: str = "scripted"):
        self.name = name
        self._steps = iter(steps)
        self.observations: list[Observation] = []

    async def act(self, obs: Observation) -> AgentReply:
        self.observations.append(obs)
        step = next(self._steps)
        while isinstance(step, float):
            await asyncio.sleep(step)
            step = next(self._steps)
        if isinstance(step, Exception):
            raise step
        return parse_move(step)
