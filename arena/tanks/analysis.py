"""What did a strategy change actually do? Counterfactual comparison on the deterministic world.

At each director-applied change we fork the world: one branch continues as
logged, the other keeps that side's previous strategy. Both run until the
next strategy change (at most WINDOW_SECONDS) and the commanded tank's
behaviour and outcomes (kills, deaths) are compared.
"""

from __future__ import annotations

import copy
import math

from arena.events.log import Event
from arena.tanks import rules as R
from arena.tanks.strategy import Strategy, parse_strategy
from arena.tanks.world import World

WINDOW_SECONDS = 30.0  # compared until the next strategy change, at most this long
METRICS = ("avg_distance", "engaged_seconds", "damage_dealt", "damage_taken", "xp_gained", "kills", "deaths")


def measure(world: World, side: int, ticks: int) -> dict:
    """Advance `world` by `ticks`; return tank `side`'s totals, per-tick distances and notable events."""
    me, opp = world.tanks[side], world.tanks[1 - side]
    seg0 = copy.deepcopy(me.segment)
    score0, kills0, deaths0 = me.score, me.kills, me.deaths
    start = world.tick
    distances: list[float | None] = []  # None while either tank is dead
    events = []
    while world.tick < start + ticks:
        for e in world.step():
            if e["type"] == "kill" and side in (e["killer"], e["victim"]):
                if e["victim"] == side:
                    by = "shape" if e["killer"] is None else "opponent" if e["killer_ai"] else "npc"
                    what = f"you died ({by})"
                else:
                    what = "you killed the opponent" if e["victim_ai"] else "you killed an NPC"
                events.append({"after_s": round((world.tick - start) * R.DT, 1), "event": what})
        distances.append(math.hypot(me.x - opp.x, me.y - opp.y) if me.alive and opp.alive else None)
    seg = me.segment
    taken = sum(seg["damage_taken"].values()) - sum(seg0["damage_taken"].values())
    return {
        "damage_dealt": round(seg["damage_to_opponent"] - seg0["damage_to_opponent"], 1),
        "damage_taken": round(taken, 1),
        "xp_gained": round(me.score - score0),
        "kills": me.kills - kills0,
        "deaths": me.deaths - deaths0,
        "events": events,
        "_distances": distances,
        "_view": me.view,
    }


def compare(world: World, side: int, new: Strategy, ticks: int) -> dict:
    """Fork `world` now: apply `new` to `side` in one branch, keep the current strategy in the other.

    Distance and engagement are averaged only over ticks where both tanks are alive in *both* branches,
    so a death in one branch does not skew the comparison.
    """
    actual, counterfactual = copy.deepcopy(world), copy.deepcopy(world)
    actual.set_strategy(side, new)
    branches = [measure(actual, side, ticks), measure(counterfactual, side, ticks)]
    common = [i for i, (a, b) in enumerate(zip(branches[0]["_distances"], branches[1]["_distances"]))
              if a is not None and b is not None]
    for m in branches:
        d = [m["_distances"][i] for i in common]
        m["avg_distance"] = round(sum(d) / len(d), 1) if d else None
        m["engaged_seconds"] = round(sum(x <= m["_view"] for x in d) * R.DT, 1)
        del m["_distances"], m["_view"]
    with_cmd, without = branches
    delta = {k: (None if with_cmd[k] is None or without[k] is None else round(with_cmd[k] - without[k], 1))
             for k in METRICS}
    return {"with": with_cmd, "without": without, "delta": delta, "seconds": round(ticks * R.DT, 1),
            "compared_seconds": round(len(common) * R.DT, 1)}


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
