"""Verify a Pong log: every point is re-simulated from its logged start state and strategies."""

from __future__ import annotations

from arena.events.log import Event
from arena.events.replay import ReplayCheck
from arena.pong.physics import Paddle, simulate_point
from arena.pong.strategy import parse_strategy


def verify_pong(events: list[Event]) -> ReplayCheck:
    start = next((e for e in events if e["type"] == "match_started"), None)
    if start is None:
        return ReplayCheck(False, ["no match_started event"], None)
    ended = {e["point"]: e for e in events if e["type"] == "point_ended"}
    problems = []
    score = [0, 0]
    for e in (e for e in events if e["type"] == "point_started"):
        point = e["point"]
        if point not in ended:
            problems.append(f"point {point}: no point_ended event")
            continue
        strategies = tuple(parse_strategy(s) for s in e["strategies"])
        paddles = tuple(Paddle(**p) for p in e["paddles"])
        sim = simulate_point(f"{start['seed']}:{point}", e["server"], strategies, paddles)
        r, rec = sim.result, ended[point]
        if r is None:
            problems.append(f"point {point}: replay did not finish")
            continue
        for key, got in (("winner", r.winner), ("hits", r.hits), ("ticks", r.ticks)):
            if got != rec[key]:
                problems.append(f"point {point}: {key} replay={got} record={rec[key]}")
        if r.winner is not None:
            score[r.winner] += 1
    final = next((e for e in reversed(events) if e["type"] == "match_ended"), None)
    if final is None:
        problems.append("no match_ended event")
    elif final["score"] != score:
        problems.append(f"final score replay={score} record={final['score']}")
    return ReplayCheck(not problems, problems, None)
