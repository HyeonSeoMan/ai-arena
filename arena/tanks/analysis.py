"""What did a strategy change actually do? Counterfactual comparison on the deterministic world.

At each director-applied change we fork the world: one branch continues as
logged, the other keeps that side's previous strategy. Both run for a short
window and the commanded tank's behaviour is compared.
"""

from __future__ import annotations

import copy
import math

from arena.events.log import Event
from arena.tanks import rules as R
from arena.tanks.strategy import Strategy, parse_strategy
from arena.tanks.world import World

WINDOW_SECONDS = 10.0
METRICS = ("avg_distance", "engaged_seconds", "damage_dealt", "damage_taken", "xp_gained", "kills", "deaths")


def measure(world: World, side: int, ticks: int) -> dict:
    """Advance `world` by `ticks` and return the metrics of tank `side` over that span."""
    me, opp = world.tanks[side], world.tanks[1 - side]
    seg0 = copy.deepcopy(me.segment)
    score0, kills0, deaths0 = me.score, me.kills, me.deaths
    dist_sum, dist_n, engaged = 0.0, 0, 0
    end = world.tick + ticks
    while world.tick < end:
        world.step()
        if me.alive and opp.alive:
            d = math.hypot(me.x - opp.x, me.y - opp.y)
            dist_sum += d
            dist_n += 1
            engaged += d <= me.view
    seg = me.segment
    taken = sum(seg["damage_taken"].values()) - sum(seg0["damage_taken"].values())
    return {
        "avg_distance": round(dist_sum / dist_n, 1) if dist_n else None,
        "engaged_seconds": round(engaged * R.DT, 1),
        "damage_dealt": round(seg["damage_to_opponent"] - seg0["damage_to_opponent"], 1),
        "damage_taken": round(taken, 1),
        "xp_gained": round(me.score - score0),
        "kills": me.kills - kills0,
        "deaths": me.deaths - deaths0,
    }


def compare(world: World, side: int, new: Strategy, ticks: int) -> dict:
    """Fork `world` now: apply `new` to `side` in one branch, keep the current strategy in the other."""
    actual, counterfactual = copy.deepcopy(world), copy.deepcopy(world)
    actual.set_strategy(side, new)
    with_cmd = measure(actual, side, ticks)
    without = measure(counterfactual, side, ticks)
    delta = {k: (None if with_cmd[k] is None or without[k] is None else round(with_cmd[k] - without[k], 1))
             for k in METRICS}
    return {"with": with_cmd, "without": without, "delta": delta, "seconds": round(ticks * R.DT, 1)}


def changed_fields(old: dict, new: dict) -> dict:
    return {k: [old.get(k), new[k]] for k in new if k != "note" and old.get(k) != new[k]}


def director_effects(events: list[Event], window: float = WINDOW_SECONDS) -> list[dict]:
    """Replay a tanks log and evaluate every director-applied strategy change counterfactually."""
    start = next(e for e in events if e["type"] == "match_started")
    end = next((e for e in reversed(events) if e["type"] == "match_ended"), None)
    applied = [e for e in events if e["type"] == "strategy_applied"]
    commands = {e["id"]: e for e in events if e["type"] == "director_command"}
    results = {e["id"]: e for e in events if e["type"] == "director_result"}
    final_tick = end["tick"] if end else max((e["tick"] for e in applied), default=0)

    world = World(start["seed"], names=(start["left"], start["right"]))
    previous = [Strategy(), Strategy()]
    effects = []
    for i, e in enumerate(applied):
        while world.tick < e["tick"]:
            world.step()
        strategies = [parse_strategy(d) for d in e["strategies"]]
        if e.get("source") == "director":
            side = e["side"]
            # Compare until the next strategy change of any kind (or the window / match end).
            next_tick = next((a["tick"] for a in applied[i + 1:] if a["tick"] > e["tick"]), final_tick)
            horizon = min(e["tick"] + round(window * R.TICK_RATE), final_tick, next_tick)
            fork = copy.deepcopy(world)
            for j, s in enumerate(previous):
                fork.set_strategy(j, s)
            ticks = max(0, horizon - e["tick"])
            cmd = commands.get(e.get("command_id"), {})
            res = results.get(e.get("command_id"), {})
            effects.append({
                "command_id": e.get("command_id"),
                "text": cmd.get("text"),
                "kind": cmd.get("kind"),
                "issued_s": round(cmd.get("tick", e["tick"]) * R.DT, 1),
                "applied_s": round(e["tick"] * R.DT, 1),
                "latency_ms": res.get("latency_ms"),
                "opinion": res.get("opinion"),
                "changes": changed_fields(previous[side].to_dict(), strategies[side].to_dict()),
                "effect": compare(fork, side, strategies[side], ticks) if ticks else None,
            })
        for j, s in enumerate(strategies):
            world.set_strategy(j, s)
        previous = strategies
    return effects
