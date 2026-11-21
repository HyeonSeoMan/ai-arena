import math
from dataclasses import replace

import pytest

from arena.tanks import rules as R
from arena.tanks.strategy import DEFAULT_BUILD, Strategy, StrategyError, next_stat, parse_strategy
from arena.tanks.world import Bullet, Shape, World


def empty_world(npcs=0):
    w = World("test", npcs=npcs)
    w.shapes.clear()
    return w


def run_seconds(w, seconds):
    events = []
    for _ in range(round(seconds * R.TICK_RATE)):
        events += w.step()
    return events


def set_level(w, i, level, strategy=None):
    t = w.tanks[i]
    if strategy:
        w.set_strategy(i, strategy)
    t.xp = float(R.xp_for_level(level))
    w._set_level(t, level, reset=True)
    t.hp = t.max_hp
    return t


# ---- rules / strategy ----

def test_xp_curve_is_monotonic_and_round_trips():
    for level in range(1, R.MAX_LEVEL + 1):
        assert R.level_for_xp(R.xp_for_level(level)) == level
        if level > 1:
            assert R.xp_for_level(level) > R.xp_for_level(level - 1)
    assert R.level_for_xp(10**9) == R.MAX_LEVEL


def test_next_stat_follows_build_order_without_respec():
    stats = {s: 0 for s in R.STATS}
    build = ("reload", "reload", "max_health")
    picks = []
    for _ in range(3):
        stat = next_stat(build, stats, R.STAT_MAX)
        stats[stat] += 1
        picks.append(stat)
    assert picks == ["reload", "reload", "max_health"]
    # A new build only steers future points; points already spent stay.
    assert next_stat(("max_health",), stats, R.STAT_MAX) == DEFAULT_BUILD[0]
    assert next_stat(("bullet_speed",), stats, R.STAT_MAX) == "bullet_speed"


def test_next_stat_skips_maxed_stats():
    stats = {s: 0 for s in R.STATS} | {"reload": R.STAT_MAX}
    assert next_stat(("reload", "reload", "bullet_damage"), stats, R.STAT_MAX) == "bullet_damage"


def test_parse_strategy_validates_and_clamps():
    data = Strategy().to_dict() | {"aggression": 3, "retreat_hp": -1, "note": "x" * 999}
    s = parse_strategy(data)
    assert (s.aggression, s.retreat_hp, len(s.note)) == (1.0, 0.0, 200)
    for bad in ({"build": ["laser"]}, {"build": []}, {"class_choice": "drone"}, {"npc_policy": "befriend"},
                {"strafe": "lots"}, {"aim_lead": True}):
        with pytest.raises(StrategyError):
            parse_strategy(Strategy().to_dict() | bad)


# ---- world basics ----

def test_world_setup():
    w = World("setup")
    assert [t.is_ai for t in w.tanks] == [True, True, False, False, False]
    assert len(w.shapes) == R.SHAPE_COUNT
    lo, hi = R.NPC_LEVELS
    assert all(lo <= t.level <= hi for t in w.tanks[2:])
    assert all(t.hp == t.max_hp for t in w.tanks)


def test_simulation_is_deterministic():
    a, b = World("same"), World("same")
    for w in (a, b):
        w.set_strategy(0, replace(Strategy(), aggression=0.9))
    run_seconds(a, 30)
    run_seconds(b, 30)
    assert a.state_hash() == b.state_hash()
    assert a.state_hash() != World("other").state_hash()


def test_tanks_stay_inside_the_map():
    w = World("bounds")
    for _ in range(30 * 40):
        w.step()
        for t in w.tanks:
            if t.alive:
                assert t.radius - 1e-6 <= t.x <= R.MAP_SIZE - t.radius + 1e-6
                assert t.radius - 1e-6 <= t.y <= R.MAP_SIZE - t.radius + 1e-6


def test_destroying_a_shape_gives_xp_and_levels_up():
    w = empty_world()
    me = w.tanks[0]
    w.shapes.append(Shape(999, "pentagon", me.x + 140, me.y, R.SHAPES["pentagon"].hp, 0.0))
    events = run_seconds(w, 10)
    assert not w.shapes or all(s.id != 999 for s in w.shapes)
    assert me.xp >= R.SHAPES["pentagon"].xp
    assert me.score == me.xp
    assert any(e["type"] == "level" and e["tank"] == 0 for e in events)
    assert sum(me.stats.values()) == me.level - 1
    assert me.segment["xp_by_source"]["pentagon"] == R.SHAPES["pentagon"].xp


def test_penetrating_bullet_damages_each_target_once():
    w = empty_world()
    target = w.tanks[1]
    target.x, target.y = 600.0, 600.0
    hp = target.hp
    w.bullets.append(Bullet(owner=0, x=580.0, y=600.0, vx=30.0, vy=0.0, damage=10.0, pen=1000.0, life=2.0))
    for _ in range(30):  # the slow bullet overlaps the tank for many ticks
        w._move_bullets()
        w._bullet_hits()
    assert target.hp == pytest.approx(hp - 10.0)


def test_class_changes_at_level_15_using_class_choice():
    w = empty_world()
    set_level(w, 0, R.CLASS_LEVEL, replace(Strategy(), class_choice="sniper"))
    events = w.step()
    assert w.tanks[0].cls == "sniper"
    assert {"type": "class", "tank": 0, "class": "sniper"} in events
    assert w.tanks[0].view == pytest.approx(R.VIEW_RADIUS * R.CLASSES["sniper"].view)


@pytest.mark.parametrize("cls", R.UPGRADE_CLASSES)
def test_every_class_fires(cls):
    w = empty_world()
    t = w.tanks[0]
    w._change_class(t, cls)
    for _ in range(30):
        w._fire(t, True)
    assert len(w.bullets) >= len(R.CLASSES[cls].barrels)
    if cls == "flank":
        assert any(b.vx < 0 for b in w.bullets) and any(b.vx > 0 for b in w.bullets)


def test_twin_barrels_alternate():
    w = empty_world()
    t = w.tanks[0]
    w._change_class(t, "twin")
    shots = []
    for tick in range(60):
        before = len(w.bullets)
        w._fire(t, True)
        shots += [tick] * (len(w.bullets) - before)
    gaps = {b - a for a, b in zip(shots, shots[1:])}
    assert 0 not in gaps  # never both barrels on the same tick


def test_kill_rewards_killer_and_penalises_ai_victim():
    w = empty_world()
    killer, victim = w.tanks[0], set_level(w, 1, 20)
    victim_xp = victim.xp
    victim.last_hit_by = 0
    victim.hp = 0
    events = w.step()
    kill = next(e for e in events if e["type"] == "kill")
    assert kill == {"type": "kill", "victim": 1, "killer": 0, "cause": killer.name, "victim_ai": True, "killer_ai": True}
    assert killer.kills == 1 and victim.deaths == 1
    assert killer.xp == pytest.approx(victim_xp * R.KILL_XP_SHARE)
    assert victim.xp == pytest.approx(victim_xp * R.DEATH_XP_KEPT)
    assert victim.level == R.level_for_xp(victim.xp)
    assert sum(victim.stats.values()) == victim.level - 1
    assert not victim.alive


def test_npc_kill_is_not_an_ai_kill():
    w = empty_world(npcs=1)
    npc = w.tanks[2]
    npc.last_hit_by = 0
    npc.hp = 0
    w.step()
    assert w.tanks[0].kills == 0
    assert w.tanks[0].segment["kills"] == [{"t": 0, "victim": "npc"}]


def test_respawn_after_delay_with_spawn_protection():
    w = empty_world()
    victim = w.tanks[1]
    victim.last_hit_by = 0
    victim.hp = 0
    w.step()
    assert not victim.alive
    events = run_seconds(w, R.RESPAWN_SECONDS + 0.1)
    assert victim.alive and victim.hp == victim.max_hp
    assert any(e["type"] == "respawn" and e["tank"] == 1 for e in events)
    hp = victim.hp
    w._damage(victim, 50, w.tanks[0])
    assert victim.hp == hp  # protected
    run_seconds(w, R.SPAWN_PROTECTION_SECONDS)
    w._damage(victim, 5, w.tanks[0])
    assert victim.hp < victim.max_hp


def test_shapes_respawn_to_target_count():
    w = World("refill")
    del w.shapes[:20]
    run_seconds(w, 5)
    assert len(w.shapes) == R.SHAPE_COUNT


def test_pentagons_are_more_common_near_the_centre():
    w = World("dist")
    centre = edge = 0
    for _ in range(3000):
        w._spawn_shape()
        s = w.shapes.pop()
        if s.kind == "pentagon":
            if math.hypot(s.x - 600, s.y - 600) < 250:
                centre += 1
            else:
                edge += 1
    area_centre = math.pi * 250**2
    area_edge = R.MAP_SIZE**2 - area_centre
    assert centre / area_centre > 3 * edge / area_edge


def test_frame_is_json_friendly():
    import json

    frame = World("frame").frame()
    json.dumps(frame)
    assert len(frame["tanks"]) == 5 and len(frame["shapes"]) == R.SHAPE_COUNT
