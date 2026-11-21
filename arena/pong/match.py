"""Pong match runner: real-time points, simultaneous coach timeouts between points."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass

from arena.coaching import Coach, consult
from arena.events.log import EventLog
from arena.pong import physics
from arena.pong.coach import Briefing
from arena.pong.physics import DT, HEIGHT, Paddle, PointResult, PointSim, between_points
from arena.pong.strategy import Strategy


@dataclass(frozen=True)
class PongConfig:
    points_to_win: int = 7
    timeout_seconds: float = 20.0
    realtime: bool = True  # False: simulate as fast as possible (tests, terminal runs)
    frame_every: int = 2  # broadcast every 2nd tick = 30Hz
    serve_pause: float = 1.0
    history: int = 6  # rallies included in each briefing


def rally_summary(point: int, r: PointResult) -> dict:
    """Neutral record of a finished point; perspective() turns it into a coach's view."""
    stats = {}
    for side in (0, 1):
        shots = [s for s in r.shots if s.side == side]
        zones = {"top": 0, "middle": 0, "bottom": 0}
        for s in shots:
            zones["top" if s.target_y < HEIGHT / 3 else "bottom" if s.target_y > 2 * HEIGHT / 3 else "middle"] += 1
        stats[side] = {
            "shots": len(shots),
            "avg_abs_angle": round(sum(abs(s.angle_deg) for s in shots) / len(shots), 1) if shots else None,
            "avg_speed": round(sum(s.speed for s in shots) / len(shots), 1) if shots else None,
            "mishits": sum(s.mishit for s in shots),
            "zones": zones,
        }
    return {
        "point": point,
        "winner": r.winner,
        "reason": r.reason,
        "hits": r.hits,
        "miss_distance": r.miss_distance,
        "stamina": list(r.stamina),
        "stats": stats,
    }


def perspective(summary: dict, side: int) -> dict:
    other = 1 - side
    if summary["winner"] is None:
        end = "rally limit reached, point replayed"
    elif summary["winner"] == side:
        end = "opponent missed"
    else:
        end = "you missed"
    return {
        "point": summary["point"],
        "won": None if summary["winner"] is None else summary["winner"] == side,
        "end": end,
        "hits": summary["hits"],
        "miss_distance": summary["miss_distance"],
        "your_stamina": summary["stamina"][side],
        "you": summary["stats"][side],
        "opponent": summary["stats"][other],
    }


async def run_pong(
    left: Coach,
    right: Coach,
    log: EventLog,
    config: PongConfig | None = None,
    seed: str | None = None,
    clock: Callable[[], float] = time.monotonic,
) -> tuple[int, list[int]]:
    config = config or PongConfig()
    seed = seed or log.match_id
    coaches = (left, right)
    strategies = [Strategy(), Strategy()]
    score = [0, 0]
    history: list[dict] = []
    paddles = (Paddle(), Paddle())
    point = 0

    log.emit(
        "match_started",
        game="pong",
        left=left.name,
        right=right.name,
        seed=seed,
        config=asdict(config),
        field={"width": physics.WIDTH, "height": physics.HEIGHT, "paddle_half": physics.PADDLE_HALF,
               "paddle_x": [physics.PADDLE_X[0], physics.PADDLE_X[1]], "ball_r": physics.BALL_R},
    )

    async def strategy_phase(phase: str) -> None:
        briefings = [
            Briefing(
                side=side,
                phase=phase,
                point=point,
                score=(score[side], score[1 - side]),
                points_to_win=config.points_to_win,
                seconds=config.timeout_seconds,
                current=strategies[side].to_dict(),
                rallies=[perspective(h, side) for h in history[-config.history :]],
            )
            for side in (0, 1)
        ]
        await consult(coaches, briefings, strategies, log, config.timeout_seconds, phase, clock, score=score[:])

    await strategy_phase("pregame")

    while max(score) < config.points_to_win:
        server = point % 2
        log.emit("point_started", point=point, server=server, score=score[:],
                 strategies=[s.to_dict() for s in strategies], paddles=[asdict(p) for p in paddles])
        sim = PointSim(f"{seed}:{point}", server, (strategies[0], strategies[1]), paddles)
        result = await _play(sim, point, log, config, clock)

        if result.winner is not None:
            score[result.winner] += 1
        summary = rally_summary(point, result)
        history.append(summary)
        log.emit("point_ended", **summary, ticks=result.ticks, score=score[:])
        point += 1
        between_points(paddles)
        if max(score) < config.points_to_win:
            await strategy_phase("timeout")

    winner = 0 if score[0] > score[1] else 1
    log.emit("match_ended", winner=winner, score=score, points=point)
    return winner, score


async def _play(sim: PointSim, point: int, log: EventLog, config: PongConfig, clock) -> PointResult:
    if config.realtime:
        log.broadcast("frame", point=point, **sim.frame())
        await asyncio.sleep(config.serve_pause)
    seen = 0
    next_tick = clock()
    while sim.step() is None:
        for shot in sim.shots[seen:]:
            log.emit("hit", point=point, **asdict(shot))
        seen = len(sim.shots)
        if config.realtime:
            if sim.tick % config.frame_every == 0:
                log.broadcast("frame", point=point, **sim.frame())
            next_tick += DT
            delay = next_tick - clock()
            if delay > 0:
                await asyncio.sleep(delay)
        elif sim.tick % 1200 == 0:
            await asyncio.sleep(0)
    if config.realtime:
        log.broadcast("frame", point=point, **sim.frame())
    return sim.result
