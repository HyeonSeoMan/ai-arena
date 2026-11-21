"""Coach strategy: the only thing an AI coach controls. Validated and clamped here."""

from __future__ import annotations

from dataclasses import asdict, dataclass

AIMS = ("away", "corners", "center", "random")
_UNIT_FIELDS = ("angle", "power", "effort", "anticipation", "home")
MAX_NOTE = 200


@dataclass(frozen=True)
class Strategy:
    aim: str = "away"
    angle: float = 0.5
    power: float = 0.3
    effort: float = 0.7
    anticipation: float = 0.5
    home: float = 0.6
    note: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


STRATEGY_SCHEMA = {
    "type": "object",
    "properties": {
        "aim": {"type": "string", "enum": list(AIMS)},
        **{name: {"type": "number"} for name in _UNIT_FIELDS},
        "note": {"type": "string"},
    },
    "required": ["aim", *_UNIT_FIELDS, "note"],
    "additionalProperties": False,
}


class StrategyError(ValueError):
    pass


def parse_strategy(data: object) -> Strategy:
    """Build a Strategy from untrusted JSON. Out-of-range numbers are clamped; bad types are rejected."""
    if not isinstance(data, dict):
        raise StrategyError("strategy must be a JSON object")
    aim = data.get("aim")
    if aim not in AIMS:
        raise StrategyError(f"aim must be one of {', '.join(AIMS)}, got {aim!r}")
    values = {}
    for name in _UNIT_FIELDS:
        value = data.get(name)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise StrategyError(f"{name} must be a number between 0 and 1, got {value!r}")
        values[name] = min(1.0, max(0.0, float(value)))
    note = data.get("note", "")
    note = note[:MAX_NOTE] if isinstance(note, str) else ""
    return Strategy(aim=aim, note=note, **values)
