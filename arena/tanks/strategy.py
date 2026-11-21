"""What a tank coach controls: stat build order, class choice and bot behaviour."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

from arena.tanks.rules import STATS, UPGRADE_CLASSES

FARM_TARGETS = ("nearest", "valuable", "safe")
NPC_POLICIES = ("avoid", "hunt", "ignore")
UNIT_FIELDS = ("aggression", "engage_range", "retreat_hp", "strafe", "aim_lead", "center_bias")
# Added after the first matches were logged: optional when parsing so old logs still replay.
OPTIONAL_UNIT_FIELDS = {"seek": 0.0, "zone_pull": 0.0}
MAX_BUILD = 60
MAX_NOTE = 200

DEFAULT_BUILD = (
    "reload", "bullet_damage", "bullet_penetration", "movement_speed",
    "reload", "bullet_damage", "bullet_penetration", "bullet_speed",
    "max_health", "health_regen",
) * 4


@dataclass(frozen=True)
class Strategy:
    build: tuple[str, ...] = DEFAULT_BUILD
    class_choice: str = "twin"
    farm_target: str = "valuable"
    npc_policy: str = "ignore"
    aggression: float = 0.5
    engage_range: float = 0.5
    retreat_hp: float = 0.3
    strafe: float = 0.5
    aim_lead: float = 0.5
    center_bias: float = 0.5
    seek: float = 0.0
    zone_pull: float = 0.0
    note: str = ""

    def to_dict(self) -> dict:
        d = asdict(self)
        d["build"] = list(self.build)
        return d


STRATEGY_SCHEMA = {
    "type": "object",
    "properties": {
        "build": {"type": "array", "items": {"type": "string", "enum": list(STATS)}},
        "class_choice": {"type": "string", "enum": list(UPGRADE_CLASSES)},
        "farm_target": {"type": "string", "enum": list(FARM_TARGETS)},
        "npc_policy": {"type": "string", "enum": list(NPC_POLICIES)},
        **{name: {"type": "number"} for name in (*UNIT_FIELDS, *OPTIONAL_UNIT_FIELDS)},
        "note": {"type": "string"},
    },
    "required": ["build", "class_choice", "farm_target", "npc_policy", *UNIT_FIELDS, *OPTIONAL_UNIT_FIELDS, "note"],
    "additionalProperties": False,
}


class StrategyError(ValueError):
    pass


def _choice(data: dict, key: str, options: tuple[str, ...]) -> str:
    value = data.get(key)
    if value not in options:
        raise StrategyError(f"{key} must be one of {', '.join(options)}, got {value!r}")
    return value


def parse_strategy(data: object) -> Strategy:
    """Build a Strategy from untrusted JSON: clamp numbers, reject unknown names."""
    if not isinstance(data, dict):
        raise StrategyError("strategy must be a JSON object")
    build = data.get("build")
    if not isinstance(build, list) or not build:
        raise StrategyError("build must be a non-empty list of stat names")
    bad = [b for b in build if b not in STATS]
    if bad:
        raise StrategyError(f"unknown stats in build: {bad[:3]}; valid: {', '.join(STATS)}")
    values = {}
    for name in (*UNIT_FIELDS, *OPTIONAL_UNIT_FIELDS):
        value = data.get(name, OPTIONAL_UNIT_FIELDS.get(name))
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise StrategyError(f"{name} must be a number between 0 and 1, got {value!r}")
        values[name] = min(1.0, max(0.0, float(value)))
    note = data.get("note", "")
    return Strategy(
        build=tuple(build[:MAX_BUILD]),
        class_choice=_choice(data, "class_choice", UPGRADE_CLASSES),
        farm_target=_choice(data, "farm_target", FARM_TARGETS),
        npc_policy=_choice(data, "npc_policy", NPC_POLICIES),
        note=note[:MAX_NOTE] if isinstance(note, str) else "",
        **values,
    )


def next_stat(build: tuple[str, ...], stats: dict[str, int], stat_max: int) -> str | None:
    """Where the next stat point goes: the first build entry not yet satisfied.

    Points already spent are never refunded, so changing the build only steers
    future points (no respec). Entries for maxed stats are skipped; if the build
    is exhausted, remaining points go to the first non-maxed stat in DEFAULT_BUILD.
    """
    seen: dict[str, int] = {}
    for name in build:
        seen[name] = seen.get(name, 0) + 1
        if stats[name] < seen[name] and stats[name] < stat_max:
            return name
    for name in DEFAULT_BUILD:
        if stats[name] < stat_max:
            return name
    return None
