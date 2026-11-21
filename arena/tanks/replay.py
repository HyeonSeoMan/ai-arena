"""Verify a Coach Tanks log by re-simulating the whole match from its seed and applied strategies."""

from __future__ import annotations

from arena.events.log import Event
from arena.events.replay import ReplayCheck
from arena.tanks.strategy import parse_strategy
from arena.tanks.world import World


def verify_tanks(events: list[Event]) -> ReplayCheck:
    start = next((e for e in events if e["type"] == "match_started"), None)
    end = next((e for e in reversed(events) if e["type"] == "match_ended"), None)
    if start is None or end is None:
        return ReplayCheck(False, ["missing match_started or match_ended"], None)
    applied = [e for e in events if e["type"] == "strategy_applied"]
    world = World(start["seed"], names=(start["left"], start["right"]))
    problems = []
    for i, e in enumerate(applied):
        while world.tick < e["tick"]:
            world.step()
        for side, data in enumerate(e["strategies"]):
            world.set_strategy(side, parse_strategy(data))
        if world.state_hash() != e["hash"]:
            problems.append(f"state differs at timeout {i + 1} (tick {e['tick']})")
            break
    if not problems:
        while world.tick < end["tick"]:
            world.step()
        if world.state_hash() != end["hash"]:
            problems.append(f"final state differs (tick {end['tick']})")
        kills = [world.tanks[0].kills, world.tanks[1].kills]
        if kills != end["kills"]:
            problems.append(f"kills replay={kills} record={end['kills']}")
    return ReplayCheck(not problems, problems, None)
