"""Coach Tanks rules and numbers. Every balance knob lives here."""

from __future__ import annotations

from dataclasses import dataclass

TICK_RATE = 30
DT = 1 / TICK_RATE

MAP_SIZE = 1200.0
VIEW_RADIUS = 450.0
MATCH_SECONDS = 360
KILLS_TO_WIN = 3
RESPAWN_SECONDS = 3.0
SPAWN_PROTECTION_SECONDS = 3.0
DEATH_XP_KEPT = 0.6  # fraction of XP an AI tank keeps when it dies
KILL_XP_SHARE = 0.35  # fraction of the victim's XP the killer gains
NPC_RESPAWN_SECONDS = 10.0
NPC_COUNT = 3
NPC_LEVELS = (3, 10)

MAX_LEVEL = 30
STAT_MAX = 7
STATS = (
    "health_regen",
    "max_health",
    "body_damage",
    "bullet_speed",
    "bullet_penetration",
    "bullet_damage",
    "reload",
    "movement_speed",
)

TANK_RADIUS = 15.0
BULLET_RADIUS = 5.0
BULLET_LIFETIME = 1.6  # seconds


def xp_for_level(level: int) -> int:
    """Total XP needed to reach `level` (level 1 needs 0)."""
    return round(16 * (level - 1) ** 1.9)


def level_for_xp(xp: float) -> int:
    level = 1
    while level < MAX_LEVEL and xp >= xp_for_level(level + 1):
        level += 1
    return level


@dataclass(frozen=True)
class ShapeKind:
    name: str
    sides: int
    radius: float
    hp: float
    xp: int
    body_damage: float  # damage per second to a tank touching it
    toughness: float  # penetration a bullet loses when hitting it


SHAPES = {
    "square": ShapeKind("square", 4, 11.0, 10.0, 10, 5.0, 4.0),
    "triangle": ShapeKind("triangle", 3, 12.0, 30.0, 25, 8.0, 6.0),
    "pentagon": ShapeKind("pentagon", 5, 22.0, 100.0, 100, 12.0, 10.0),
}
SHAPE_COUNT = 60
TANK_TOUGHNESS = 8.0


@dataclass(frozen=True)
class Barrel:
    angle: float  # offset from the aim direction, radians
    side_offset: float  # perpendicular offset of the muzzle (twin)
    delay: float  # fraction of the reload before this barrel fires (alternating barrels)


@dataclass(frozen=True)
class TankClass:
    name: str
    barrels: tuple[Barrel, ...]
    damage: float = 1.0
    reload: float = 1.0  # multiplier on reload time (lower = faster)
    bullet_speed: float = 1.0
    lifetime: float = 1.0
    spread: float = 0.0  # radians of random spread per shot
    view: float = 1.0


CLASSES = {
    "basic": TankClass("basic", (Barrel(0.0, 0.0, 0.0),)),
    "twin": TankClass("twin", (Barrel(0.0, -6.0, 0.0), Barrel(0.0, 6.0, 0.5)), damage=0.7),
    "sniper": TankClass("sniper", (Barrel(0.0, 0.0, 0.0),), reload=1.45, bullet_speed=1.5, lifetime=1.5, view=1.25),
    "machine_gun": TankClass("machine_gun", (Barrel(0.0, 0.0, 0.0),), damage=0.65, reload=0.5, spread=0.17),
    "flank": TankClass("flank", (Barrel(0.0, 0.0, 0.0), Barrel(3.14159265, 0.0, 0.0)), damage=0.9),
}
CLASS_LEVEL = 15
UPGRADE_CLASSES = ("twin", "sniper", "machine_gun", "flank")


# ---- stat effects (s = stat level 0..7, level = tank level) ----

def max_health(s: int, level: int) -> float:
    return 80 + 3 * (level - 1) + 40 * s


def regen_per_second(s: int, max_hp: float) -> float:
    return max_hp * (0.004 + 0.009 * s)


def body_damage(s: int) -> float:  # per second of contact
    return 30 + 28 * s


def bullet_speed(s: int) -> float:
    return 200 + 28 * s


def bullet_penetration(s: int) -> float:
    return 6 + 5 * s


def bullet_damage(s: int) -> float:
    return 7 + 2.5 * s


def reload_seconds(s: int) -> float:
    return 0.6 * 0.9**s


def move_speed(s: int, level: int) -> float:
    return 110 + 9 * s - 0.8 * (level - 1)
