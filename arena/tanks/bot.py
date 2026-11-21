"""Tank bot: turns a coach's strategy into per-tick movement, aim and fire.

The bot only uses what its tank can see (view radius) plus its own memory of
where other tanks were last seen.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from arena.tanks import rules as R

if TYPE_CHECKING:
    from arena.tanks.world import Tank, World

STRAFE_PERIOD = 36  # ticks between strafe direction changes (1.2s)
CENTRE = R.MAP_SIZE / 2


def decide(world: World, me: Tank) -> tuple[float, float, float, bool]:
    mx, my, aim, fire = _decide(world, me)
    ax, ay = _avoid_shapes(world, me)
    return mx + ax, my + ay, aim, fire


def _decide(world: World, me: Tank) -> tuple[float, float, float, bool]:
    s = me.strategy
    visible = [t for t in world.tanks
               if t is not me and t.alive and math.hypot(t.x - me.x, t.y - me.y) <= me.view]
    for t in visible:
        me.seen[t.id] = (world.tick, t.x, t.y, t.vx, t.vy, t.level, t.cls)
        if me.is_ai and t.is_ai:
            me.segment["opponent_visible_seconds"] += R.DT
    if (world.tick + me.id * 11) % STRAFE_PERIOD == 0:
        me.strafe_dir = -me.strafe_dir

    hp_frac = me.hp / me.max_hp
    threats = [t for t in visible if _dist(me, t) < 350]
    if threats and hp_frac < s.retreat_hp:
        nearest = min(threats, key=lambda t: _dist(me, t))
        away_x = sum(me.x - t.x for t in threats)
        away_y = sum(me.y - t.y for t in threats)
        mx, my = _unit(away_x, away_y)
        mx, my = _avoid_walls(me, mx, my)
        return mx, my, _lead_aim(me, nearest, s.aim_lead), True

    target = _pick_target(me, visible, hp_frac)
    if target is not None:
        return _engage(me, target)

    # Keep away from NPCs if told to, and from the opponent when not engaging it.
    avoid = [t for t in visible if _dist(me, t) < 300 and (t.is_ai or s.npc_policy == "avoid") and me.is_ai]
    fx, fy, aim, fire = _farm(world, me)
    px, py = _zone_pull(me)
    if avoid:
        npc = min(avoid, key=lambda t: _dist(me, t))
        ax, ay = _unit(me.x - npc.x, me.y - npc.y)
        return fx * 0.4 + ax + px, fy * 0.4 + ay + py, aim, fire
    if s.seek > 0 and me.is_ai and not any(t.is_ai for t in visible):
        sx, sy = _seek(world, me)
        return fx * (1 - s.seek) + sx * s.seek * 1.5 + px, fy * (1 - s.seek) + sy * s.seek * 1.5 + py, aim, fire
    return fx + px, fy + py, aim, fire


def _zone_pull(me: Tank) -> tuple[float, float]:
    """Drift towards the ring that center_bias prefers (1 = map centre, 0 = near the edge).

    center_bias alone only changes which visible shape is picked; zone_pull makes the tank actually
    relocate. 0 (the default) adds nothing, so older strategies behave exactly as before.
    """
    s = me.strategy
    if s.zone_pull <= 0:
        return 0.0, 0.0
    dx, dy = me.x - CENTRE, me.y - CENTRE
    r = math.hypot(dx, dy)
    error = r - (1 - s.center_bias) * CENTRE * 0.9
    if abs(error) < 60:
        return 0.0, 0.0
    ux, uy = _unit(dx, dy) if r > 1e-6 else (1.0, 0.0)
    weight = 1.5 * s.zone_pull * min(1.0, abs(error) / 200) * (1 if error > 0 else -1)
    return -ux * weight, -uy * weight


def _seek(world: World, me: Tank) -> tuple[float, float]:
    """Head for where the opponent was last seen; once that spot is checked, explore."""
    opp_id = 1 - me.id
    seen = me.seen.get(opp_id)
    if seen is not None and seen[0] != me.searched_tick:
        goal = (seen[1], seen[2])
        if math.hypot(goal[0] - me.x, goal[1] - me.y) < 60:
            me.searched_tick = seen[0]  # nothing here any more
        else:
            return _unit(goal[0] - me.x, goal[1] - me.y)
    if me.search_goal is None or math.hypot(me.search_goal[0] - me.x, me.search_goal[1] - me.y) < 60:
        me.search_goal = (world.rng.uniform(150, R.MAP_SIZE - 150), world.rng.uniform(150, R.MAP_SIZE - 150))
    return _unit(me.search_goal[0] - me.x, me.search_goal[1] - me.y)


def _pick_target(me: Tank, visible: list[Tank], hp_frac: float) -> Tank | None:
    s = me.strategy
    best, best_score = None, 0.5  # aggression above 0.5 = engage an equal opponent at full health
    for t in visible:
        d = _dist(me, t)
        if not me.is_ai:
            if d >= 220:
                continue
            bonus = 0.25  # NPCs only defend themselves
        elif t.is_ai:
            bonus = 0.0
        elif s.npc_policy == "hunt":
            bonus = 0.2
        elif s.npc_policy == "ignore" and d < 220:
            bonus = 0.1  # self-defence
        else:
            continue
        drive = s.aggression + 0.04 * (me.level - t.level) + 0.4 * (hp_frac - 1.0) + bonus - d / 4000
        if drive > best_score:
            best, best_score = t, drive
    return best


def _engage(me: Tank, t: Tank) -> tuple[float, float, float, bool]:
    s = me.strategy
    d = _dist(me, t)
    want = 80 + 420 * s.engage_range
    ux, uy = _unit(t.x - me.x, t.y - me.y)
    radial = max(-1.0, min(1.0, (d - want) / 100))
    px, py = -uy * me.strafe_dir, ux * me.strafe_dir
    mx = ux * radial + px * s.strafe
    my = uy * radial + py * s.strafe
    mx, my = _avoid_walls(me, mx, my)
    return mx, my, _lead_aim(me, t, s.aim_lead), d < me.bullet_range * 1.05


def _farm(world: World, me: Tank) -> tuple[float, float, float, bool]:
    s = me.strategy
    opponent_seen = next((v for k, v in me.seen.items() if k in (0, 1) and k != me.id), None)
    best, best_value = None, -math.inf
    for sh in world.shapes:
        d = math.hypot(sh.x - me.x, sh.y - me.y)
        if d > me.view:
            continue
        if s.farm_target == "nearest":
            value = -d
        else:
            value = sh.spec.xp / (d + 80)
            if s.farm_target == "safe" and opponent_seen is not None:
                if math.hypot(sh.x - opponent_seen[1], sh.y - opponent_seen[2]) < 400:
                    value *= 0.3
        centre = 1 - math.hypot(sh.x - CENTRE, sh.y - CENTRE) / (CENTRE * math.sqrt(2))
        value *= 1 + (s.center_bias - 0.5) * 2 * (centre - 0.5) if value > 0 else 1
        if value > best_value:
            best, best_value = sh, value

    if best is None:
        if me.waypoint is None or math.hypot(me.waypoint[0] - me.x, me.waypoint[1] - me.y) < 40:
            rx = world.rng.uniform(80, R.MAP_SIZE - 80)
            ry = world.rng.uniform(80, R.MAP_SIZE - 80)
            me.waypoint = (rx + (CENTRE - rx) * s.center_bias * 0.7, ry + (CENTRE - ry) * s.center_bias * 0.7)
        mx, my = _unit(me.waypoint[0] - me.x, me.waypoint[1] - me.y)
        return mx, my, math.atan2(my, mx) if (mx or my) else me.aim, False

    d = math.hypot(best.x - me.x, best.y - me.y)
    ux, uy = _unit(best.x - me.x, best.y - me.y)
    stand = 110 + best.spec.radius
    radial = max(-1.0, min(1.0, (d - stand) / 60))
    return ux * radial, uy * radial, math.atan2(best.y - me.y, best.x - me.x), d < me.bullet_range


def _avoid_shapes(world: World, me: Tank) -> tuple[float, float]:
    """Gentle push away from shapes the tank is about to bump into."""
    ax = ay = 0.0
    for sh in world.shapes:
        gap = math.hypot(me.x - sh.x, me.y - sh.y) - me.radius - sh.spec.radius
        if gap < 25:
            ux, uy = _unit(me.x - sh.x, me.y - sh.y)
            weight = (25 - gap) / 25
            ax += ux * weight
            ay += uy * weight
    return ax, ay


def _lead_aim(me: Tank, t: Tank, lead: float) -> float:
    d = _dist(me, t)
    eta = d / me.bullet_speed * lead
    tx, ty = t.x + t.vx * eta, t.y + t.vy * eta
    return math.atan2(ty - me.y, tx - me.x)


def _avoid_walls(me: Tank, mx: float, my: float) -> tuple[float, float]:
    margin = 80
    if me.x < margin:
        mx += (margin - me.x) / margin
    elif me.x > R.MAP_SIZE - margin:
        mx -= (me.x - (R.MAP_SIZE - margin)) / margin
    if me.y < margin:
        my += (margin - me.y) / margin
    elif me.y > R.MAP_SIZE - margin:
        my -= (me.y - (R.MAP_SIZE - margin)) / margin
    return mx, my


def _dist(a, b) -> float:
    return math.hypot(a.x - b.x, a.y - b.y)


def _unit(x: float, y: float) -> tuple[float, float]:
    n = math.hypot(x, y)
    return (x / n, y / n) if n > 1e-9 else (0.0, 0.0)
