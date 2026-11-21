"""Tank coaches: briefings (what each coach may know), prompt, and offline dummies."""

from __future__ import annotations

import asyncio
import json
import random
from dataclasses import dataclass, replace

from arena import coaching
from arena.coaching import CoachReply
from arena.tanks import rules as R
from arena.tanks.strategy import FARM_TARGETS, NPC_POLICIES, STRATEGY_SCHEMA, Strategy, parse_strategy


@dataclass(frozen=True)
class Briefing:
    side: int
    phase: str  # pregame | death | level15 | quiet
    trigger: str  # human-readable reason for this timeout, from this coach's perspective
    seconds: float  # time limit for the answer
    match_time: float
    time_left: float
    kills_to_win: int
    you: dict
    opponent: dict  # public scoreboard only: kills, deaths, score
    current: dict  # your strategy in force
    since_last: dict  # your segment stats since the previous timeout
    last_seen_opponent: dict | None
    choose_class_now: bool
    owner_notes: str | None = None


GUIDE = """You are the coach of a tank in a real-time top-down arena shooter (in the style of diep.io).
You cannot steer the tank yourself: a bot plays in real time using the strategy you set. The game pauses
for a short timeout at key moments (deaths, reaching level {class_level}, or after a long quiet spell) so you
can adjust. You only know what your tank has seen; the opponent's build and plan are hidden.

Arena: {size}x{size}. Two AI tanks (you and the opponent) plus {npcs} neutral NPC tanks that farm and defend
themselves. Shapes give XP: square 10, triangle 25, pentagon 100 (pentagons cluster near the centre).
Win: kill the opponent AI tank {kills} times, or have the higher score (total XP earned) after {minutes} minutes.
Dying costs {lost:.0%} of your XP (levels and stat points drop) and the killer gains {share:.0%} of it.
NPC kills give XP but do not count as kills; dying to an NPC is a death but not an opponent kill.

Leveling: each level gives one stat point (max level {max_level}, so {points} points). Each of the 8 stats can
take at most {stat_max} points, so you must choose. Stats: health_regen, max_health, body_damage (ramming),
bullet_speed (also range), bullet_penetration (bullets pass through shapes), bullet_damage, reload (fire rate),
movement_speed. Points are spent automatically, in the order of your `build` list (a stat listed N times
gets up to N points). Changing the build only affects future points; spent points are never refunded.

At level {class_level} your tank upgrades once to `class_choice`: twin (two alternating barrels, lower damage
per bullet), sniper (fast long-range bullets, slower reload, wider view), machine_gun (very fast fire, spread,
low damage per bullet), flank (front and rear barrel).

Behaviour fields (numbers 0..1):
- farm_target: "nearest" | "valuable" (big XP per distance) | "safe" (away from where the opponent was seen)
- npc_policy: "avoid" | "hunt" | "ignore" (only fight NPCs that get close)
- aggression: how readily to fight the opponent when seen. Above ~0.5 engages an equal opponent at full health;
  being ahead in level or health makes the bot bolder, being hurt makes it more careful.
- engage_range: distance kept while fighting (0 = ram range, 1 = long range)
- retreat_hp: flee when health falls below this fraction while enemies are close
- strafe: sideways dodging while fighting
- aim_lead: 0 = aim at the target, 1 = aim where it is heading
- center_bias: preference for the richer but more contested centre
- seek: when the opponent is out of sight, how much to go looking for it (last seen spot, then exploring)
  instead of farming. 0 = never hunt; aggression alone only matters once the opponent is visible.
- zone_pull: how strongly to actually move to the area center_bias prefers (centre or edge). 0 = center_bias
  only affects which visible shape is picked, without relocating.
- note: one short sentence in Korean explaining your plan (shown to spectators)"""


def guide() -> str:
    return GUIDE.format(
        class_level=R.CLASS_LEVEL, size=int(R.MAP_SIZE), npcs=R.NPC_COUNT, kills=R.KILLS_TO_WIN,
        minutes=R.MATCH_SECONDS // 60, lost=1 - R.DEATH_XP_KEPT, share=R.KILL_XP_SHARE,
        max_level=R.MAX_LEVEL, points=R.MAX_LEVEL - 1, stat_max=R.STAT_MAX,
    )


def build_coach_prompt(b: Briefing) -> str:
    situation = {
        "timeout_reason": b.trigger,
        "match_time_seconds": round(b.match_time),
        "time_left_seconds": round(b.time_left),
        "you": b.you,
        "opponent_scoreboard": b.opponent,
        "your_current_strategy": b.current,
        "since_last_timeout": b.since_last,
        "opponent_last_seen": b.last_seen_opponent,
    }
    parts = [guide(), "Situation:", json.dumps(situation, ensure_ascii=False)]
    if b.choose_class_now:
        parts.append(f"Your tank just reached level {R.CLASS_LEVEL}: the class_choice you give now is applied immediately.")
    if b.owner_notes:
        parts.append(f"Notes from your owner:\n{b.owner_notes}")
    parts.append(
        f"You have {b.seconds:g} seconds. Do not run commands or read files.\n"
        "Reply with ONLY a JSON object with keys build, class_choice, farm_target, npc_policy, aggression, "
        "engage_range, retreat_hp, strafe, aim_lead, center_bias, seek, zone_pull, note."
    )
    return "\n\n".join(parts)


def LLMCoach(agent, name: str | None = None) -> coaching.LLMCoach:
    return coaching.LLMCoach(agent, build_coach_prompt, parse_strategy, STRATEGY_SCHEMA, name)


def _reply(strategy: Strategy) -> CoachReply:
    return CoachReply(raw=json.dumps(strategy.to_dict(), ensure_ascii=False), strategy=strategy)


class FixedCoach:
    def __init__(self, strategy: Strategy | None = None, name: str = "fixed", delay: float = 0.0):
        self.name = name
        self.strategy = strategy or Strategy(note="기본 작전 유지")
        self.delay = delay

    async def plan(self, briefing: Briefing) -> CoachReply:
        if self.delay:
            await asyncio.sleep(self.delay)
        return _reply(self.strategy)


class RandomCoach:
    """Picks a random build and behaviour each timeout; handy for watching variety offline."""

    def __init__(self, name: str = "random", seed: int | None = None, delay: float = 0.0):
        self.name = name
        self.rng = random.Random(seed)
        self.delay = delay

    async def plan(self, briefing: Briefing) -> CoachReply:
        if self.delay:
            await asyncio.sleep(self.delay)
        r = self.rng
        favourites = r.sample(R.STATS, 4)
        build = tuple(r.choice(favourites) if r.random() < 0.8 else r.choice(R.STATS) for _ in range(40))
        s = Strategy(
            build=build,
            class_choice=r.choice(R.UPGRADE_CLASSES),
            farm_target=r.choice(FARM_TARGETS),
            npc_policy=r.choice(NPC_POLICIES),
            aggression=round(r.random(), 2),
            engage_range=round(r.random(), 2),
            retreat_hp=round(r.uniform(0, 0.5), 2),
            strafe=round(r.random(), 2),
            aim_lead=round(r.random(), 2),
            center_bias=round(r.random(), 2),
            note=f"무작위 작전: {', '.join(favourites[:2])} 위주",
        )
        return _reply(s)


class ScriptedCoach:
    """Test helper: each step is raw text, a float (sleep), or an Exception; then the default strategy."""

    def __init__(self, steps, name: str = "scripted"):
        self.name = name
        self._steps = iter(steps)
        self.briefings: list[Briefing] = []

    async def plan(self, briefing: Briefing) -> CoachReply:
        self.briefings.append(briefing)
        step = next(self._steps, None)
        while isinstance(step, float):
            await asyncio.sleep(step)
            step = next(self._steps, None)
        if isinstance(step, Exception):
            raise step
        if step is None:
            return _reply(replace(Strategy(), note="scripted default"))
        return coaching.reply_from_text(step, parse_strategy)
