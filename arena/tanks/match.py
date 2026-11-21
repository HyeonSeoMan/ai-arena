"""Coach Tanks match runner: real-time world, event-driven coach timeouts."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass

from arena.coaching import Coach, consult
from arena.events.log import EventLog
from arena.tanks import rules as R
from arena.tanks.coach import Briefing
from arena.tanks.strategy import Strategy
from arena.tanks.world import Tank, World, new_segment


@dataclass(frozen=True)
class TanksConfig:
    timeout_seconds: float = 20.0
    realtime: bool = True  # False: simulate as fast as possible
    frame_every: int = 1  # broadcast every tick (30Hz)
    quiet_seconds: float = 90.0  # open a timeout if nothing notable happened for this long
    max_timeouts: int = 15  # pregame included
    match_seconds: float = R.MATCH_SECONDS
    kills_to_win: int = R.KILLS_TO_WIN
    start_pause: float = 1.0


def _last_seen(world: World, me: Tank, opp: Tank) -> dict | None:
    seen = me.seen.get(opp.id)
    if seen is None:
        return None
    tick, x, y, vx, vy, level, cls = seen
    return {
        "seconds_ago": round((world.tick - tick) * R.DT, 1),
        "position": [round(x), round(y)],
        "your_position": [round(me.x), round(me.y)],
        "level": level,
        "class": cls,
    }


def _trigger_text(phase: str, side: int, detail: dict) -> str:
    if phase == "pregame":
        return "match start: choose your opening build and plan"
    if phase == "quiet":
        return f"no major event for {detail['quiet']:g} seconds"
    if phase == "level15":
        who = "you" if detail["tank"] == side else "the opponent"
        return f"{who} reached level {R.CLASS_LEVEL}"
    parts = []
    for k in detail["kills"]:
        victim = "you" if k["victim"] == side else "the opponent"
        if k["killer"] is None:
            by = "a shape"
        elif k["killer"] == side:
            by = "you"
        elif k["killer_ai"]:
            by = "the opponent"
        else:
            by = "an NPC"
        parts.append(f"{victim} were killed by {by}" if victim == "you" else f"{victim} was killed by {by}")
    return "; ".join(parts)


def _briefing(world: World, side: int, phase: str, detail: dict, strategies: list, config: TanksConfig) -> Briefing:
    me, opp = world.tanks[side], world.tanks[1 - side]
    you = world.public_status(me)
    you["next_level_xp"] = R.xp_for_level(min(me.level + 1, R.MAX_LEVEL))
    you["position"] = [round(me.x), round(me.y)]
    return Briefing(
        side=side,
        phase=phase,
        trigger=_trigger_text(phase, side, detail),
        seconds=config.timeout_seconds,
        match_time=world.seconds,
        time_left=max(0.0, config.match_seconds - world.seconds),
        kills_to_win=config.kills_to_win,
        you=you,
        opponent={"kills": opp.kills, "deaths": opp.deaths, "score": round(opp.score)},
        current=strategies[side].to_dict(),
        since_last={k: (round(v, 1) if isinstance(v, float) else v) for k, v in me.segment.items()},
        last_seen_opponent=_last_seen(world, me, opp),
        choose_class_now=phase == "level15" and detail["tank"] == side and me.cls == "basic",
    )


def outcome(world: World, config: TanksConfig) -> tuple[int, str] | None:
    a, b = world.tanks[0], world.tanks[1]
    if a.kills >= config.kills_to_win or b.kills >= config.kills_to_win:
        return (0 if a.kills >= config.kills_to_win else 1), "kills"
    if world.seconds >= config.match_seconds:
        if a.score == b.score:
            return (0 if a.kills >= b.kills else 1), "score_tie"
        return (0 if a.score > b.score else 1), "score"
    return None


async def run_tanks(
    first: Coach,
    second: Coach,
    log: EventLog,
    config: TanksConfig | None = None,
    seed: str | None = None,
    clock: Callable[[], float] = time.monotonic,
) -> tuple[int, str]:
    config = config or TanksConfig()
    seed = seed or log.match_id
    coaches = (first, second)
    world = World(seed, names=(first.name, second.name))
    strategies: list[Strategy] = [Strategy(), Strategy()]
    used = 0
    last_consult = 0
    level15_seen: set[int] = set()

    log.emit("match_started", game="tanks", left=first.name, right=second.name, seed=seed, config=asdict(config),
             map_size=R.MAP_SIZE, npcs=[t.name for t in world.tanks[2:]])

    async def timeout(phase: str, detail: dict) -> None:
        nonlocal used, last_consult
        used += 1
        briefings = [_briefing(world, side, phase, detail, strategies, config) for side in (0, 1)]
        await consult(coaches, briefings, strategies, log, config.timeout_seconds, phase, clock,
                      trigger=briefings[0].trigger, match_time=round(world.seconds, 1), number=used)
        for side in (0, 1):
            world.set_strategy(side, strategies[side])
            world.tanks[side].segment = new_segment()
        last_consult = world.tick
        log.emit("strategy_applied", tick=world.tick, strategies=[s.to_dict() for s in strategies],
                 hash=world.state_hash())

    await timeout("pregame", {})
    if config.realtime:
        log.broadcast("frame", **world.frame())
        await asyncio.sleep(config.start_pause)

    next_tick = clock()
    result = None
    while result is None:
        events = world.step()
        detail = None
        for e in events:
            if e["type"] == "kill":
                log.emit("kill", tick=world.tick, **{k: v for k, v in e.items() if k != "type"},
                         kills=[world.tanks[0].kills, world.tanks[1].kills])
                if e["victim_ai"]:
                    detail = detail if detail and detail.get("kills") else {"kills": []}
                    detail["kills"].append(e)
            elif e["type"] == "class":
                log.emit("class_changed", tick=world.tick, tank=e["tank"], cls=e["class"])
            elif e["type"] == "level" and e["tank"] < 2:
                log.emit("level_up", tick=world.tick, tank=e["tank"], level=e["level"])
                if e["level"] >= R.CLASS_LEVEL and e["tank"] not in level15_seen:
                    level15_seen.add(e["tank"])
                    detail = detail or {"tank": e["tank"], "phase": "level15"}
        if config.realtime and world.tick % config.frame_every == 0:
            log.broadcast("frame", **world.frame(), status=[_scoreboard(world, i) for i in (0, 1)])

        result = outcome(world, config)
        if result is not None:
            break
        phase = None
        if detail is not None:
            phase = "death" if "kills" in detail else "level15"
        elif world.tick - last_consult >= config.quiet_seconds * R.TICK_RATE:
            phase, detail = "quiet", {"quiet": config.quiet_seconds}
        if phase and used < config.max_timeouts:
            await timeout(phase, detail)
            next_tick = clock()

        if config.realtime:
            next_tick += R.DT
            delay = next_tick - clock()
            if delay > 0:
                await asyncio.sleep(delay)
        elif world.tick % 300 == 0:
            await asyncio.sleep(0)

    winner, reason = result
    log.emit("match_ended", winner=winner, reason=reason, tick=world.tick, hash=world.state_hash(),
             kills=[world.tanks[0].kills, world.tanks[1].kills],
             scores=[round(world.tanks[0].score), round(world.tanks[1].score)],
             levels=[world.tanks[0].level, world.tanks[1].level], timeouts=used)
    if config.realtime:
        log.broadcast("frame", **world.frame(), status=[_scoreboard(world, i) for i in (0, 1)])
    return winner, reason


def _scoreboard(world: World, i: int) -> dict:
    t = world.tanks[i]
    lo, hi = R.xp_for_level(t.level), R.xp_for_level(min(t.level + 1, R.MAX_LEVEL))
    return {
        "level": t.level, "cls": t.cls, "score": round(t.score), "kills": t.kills, "deaths": t.deaths,
        "stats": t.stats, "hp": round(t.hp), "max_hp": round(t.max_hp), "alive": t.alive,
        "xp_progress": 1.0 if hi == lo else round((t.xp - lo) / (hi - lo), 3),
    }
