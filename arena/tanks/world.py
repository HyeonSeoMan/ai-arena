"""Deterministic Coach Tanks simulation (fixed 30Hz step).

Same seed + same strategies applied at the same ticks => identical match.
Tanks 0 and 1 are the AI-coached tanks; the rest are NPCs.
"""

from __future__ import annotations

import hashlib
import math
import random
from dataclasses import dataclass, field

from arena.tanks import rules as R
from arena.tanks.strategy import Strategy, next_stat

# NPCs farm and only fight what comes close (npc_policy "ignore" = self-defence within range).
NPC_STRATEGY = Strategy(
    build=("max_health", "health_regen", "bullet_damage", "reload", "movement_speed") * 6,
    farm_target="nearest",
    npc_policy="ignore",
    aggression=0.3,
    engage_range=0.35,
    retreat_hp=0.2,
    strafe=0.3,
    aim_lead=0.2,
    center_bias=0.5,
)


def new_segment() -> dict:
    """Per-tank stats accumulated between two coach timeouts."""
    return {
        "xp_by_source": {"square": 0, "triangle": 0, "pentagon": 0, "npc": 0, "opponent": 0},
        "damage_taken": {"opponent": 0.0, "npc": 0.0, "shape": 0.0},
        "damage_to_opponent": 0.0,
        "opponent_visible_seconds": 0.0,
        "deaths": [],
        "kills": [],
    }


@dataclass
class Tank:
    id: int
    name: str
    is_ai: bool
    x: float
    y: float
    strategy: Strategy
    vx: float = 0.0
    vy: float = 0.0
    aim: float = 0.0
    level: int = 1
    xp: float = 0.0
    score: float = 0.0
    cls: str = "basic"
    hp: float = 0.0
    alive: bool = True
    respawn_tick: int = 0
    protected_until: int = 0  # tick until which the tank takes no damage (after respawn)
    kills: int = 0  # kills of the other AI tank
    deaths: int = 0
    stats: dict = field(default_factory=lambda: {s: 0 for s in R.STATS})
    barrel_timers: list = field(default_factory=lambda: [0.0])
    last_hit_by: int | None = None  # tank id, or None for shapes
    # bot memory
    strafe_dir: int = 1
    waypoint: tuple[float, float] | None = None
    seen: dict = field(default_factory=dict)  # tank id -> (tick, x, y, vx, vy, level, cls)
    segment: dict = field(default_factory=new_segment)

    @property
    def radius(self) -> float:
        return R.TANK_RADIUS + 0.25 * (self.level - 1)

    @property
    def tank_class(self) -> R.TankClass:
        return R.CLASSES[self.cls]

    @property
    def max_hp(self) -> float:
        return R.max_health(self.stats["max_health"], self.level)

    @property
    def view(self) -> float:
        return R.VIEW_RADIUS * self.tank_class.view

    @property
    def bullet_speed(self) -> float:
        return R.bullet_speed(self.stats["bullet_speed"]) * self.tank_class.bullet_speed

    @property
    def bullet_range(self) -> float:
        return self.bullet_speed * R.BULLET_LIFETIME * self.tank_class.lifetime

    @property
    def reload_time(self) -> float:
        return R.reload_seconds(self.stats["reload"]) * self.tank_class.reload

    @property
    def speed(self) -> float:
        return R.move_speed(self.stats["movement_speed"], self.level)


@dataclass
class Shape:
    id: int
    kind: str
    x: float
    y: float
    hp: float
    angle: float

    @property
    def spec(self) -> R.ShapeKind:
        return R.SHAPES[self.kind]


@dataclass
class Bullet:
    owner: int
    x: float
    y: float
    vx: float
    vy: float
    damage: float
    pen: float
    life: float
    hit: set = field(default_factory=set)  # ids already damaged ("s<id>" shapes, "t<id>" tanks)


class World:
    def __init__(self, seed: str, names: tuple[str, str] = ("A", "B"), npcs: int = R.NPC_COUNT):
        self.rng = random.Random(seed)
        self.tick = 0
        self.tanks: list[Tank] = [
            Tank(0, names[0], True, 150.0, R.MAP_SIZE / 2, Strategy()),
            Tank(1, names[1], True, R.MAP_SIZE - 150.0, R.MAP_SIZE / 2, Strategy()),
        ]
        self.tanks[1].aim = math.pi
        for i in range(npcs):
            lo, hi = R.NPC_LEVELS
            npc = Tank(2 + i, f"NPC {i + 1}", False, 0.0, 0.0, NPC_STRATEGY)
            self._set_level(npc, self.rng.randint(lo, hi), reset=True)
            # Start NPCs on the top/bottom edges, away from the AI spawns on the left/right.
            npc.x = self.rng.uniform(300, R.MAP_SIZE - 300)
            npc.y = 60.0 if i % 2 == 0 else R.MAP_SIZE - 60.0
            self.tanks.append(npc)
        for t in self.tanks:
            self._reset_barrels(t)
            t.hp = t.max_hp
        self.shapes: list[Shape] = []
        self.bullets: list[Bullet] = []
        self._next_shape_id = 0
        while len(self.shapes) < R.SHAPE_COUNT:
            self._spawn_shape()
        self.events: list[dict] = []

    # ------------------------------------------------------------------ API
    def set_strategy(self, tank_id: int, strategy: Strategy) -> None:
        self.tanks[tank_id].strategy = strategy

    def step(self) -> list[dict]:
        """Advance one tick and return the notable events it produced."""
        from arena.tanks.bot import decide  # local import: bot reads World

        self.events = []
        self.tick += 1
        for t in self.tanks:
            if t.alive and t.is_ai and t.cls == "basic" and t.level >= R.CLASS_LEVEL:
                self._change_class(t, t.strategy.class_choice)

        for t in self.tanks:
            if not t.alive:
                continue
            move_x, move_y, aim, fire = decide(self, t)
            self._move(t, move_x, move_y)
            t.aim = aim
            self._fire(t, fire)

        self._move_bullets()
        self._bullet_hits()
        self._body_contacts()
        # Deaths are settled before regeneration so a lethal hit can't be healed away.
        for t in self.tanks:
            if t.alive and t.hp <= 0:
                self._kill(t)
            elif not t.alive and self.tick >= t.respawn_tick:
                self._respawn(t)
            elif t.alive:
                t.hp = min(t.max_hp, t.hp + R.regen_per_second(t.stats["health_regen"], t.max_hp) * R.DT)
        if len(self.shapes) < R.SHAPE_COUNT and self.tick % 3 == 0:
            self._spawn_shape()
        return self.events

    @property
    def seconds(self) -> float:
        return self.tick * R.DT

    def state_hash(self) -> str:
        parts = [self.tick]
        for t in self.tanks:
            parts += [round(t.x, 6), round(t.y, 6), round(t.hp, 6), round(t.xp, 6), t.level, t.cls, t.alive, t.kills]
        parts += [(s.id, round(s.hp, 6)) for s in self.shapes]
        parts.append(len(self.bullets))
        return hashlib.sha1(repr(parts).encode()).hexdigest()[:16]

    def frame(self) -> dict:
        return {
            "t": self.tick,
            "tanks": [
                {
                    "id": t.id, "name": t.name, "ai": t.is_ai, "x": round(t.x, 1), "y": round(t.y, 1),
                    "aim": round(t.aim, 3), "r": round(t.radius, 1), "hp": round(t.hp / t.max_hp, 3),
                    "cls": t.cls, "level": t.level, "alive": t.alive, "view": round(t.view),
                    "shield": self.tick < t.protected_until,
                }
                for t in self.tanks
            ],
            "shapes": [[s.id, s.kind[0], round(s.x, 1), round(s.y, 1), round(s.angle, 2), round(s.hp / s.spec.hp, 2)]
                       for s in self.shapes],
            "bullets": [[round(b.x, 1), round(b.y, 1), b.owner] for b in self.bullets],
        }

    def public_status(self, t: Tank) -> dict:
        return {
            "level": t.level, "xp": round(t.xp), "score": round(t.score), "class": t.cls,
            "hp": round(t.hp), "max_hp": round(t.max_hp), "alive": t.alive,
            "kills": t.kills, "deaths": t.deaths, "stats": dict(t.stats),
        }

    # ------------------------------------------------------------- movement
    def _move(self, t: Tank, mx: float, my: float) -> None:
        norm = math.hypot(mx, my)
        if norm > 1:
            mx, my = mx / norm, my / norm
        blend = min(1.0, 8 * R.DT)
        t.vx += (mx * t.speed - t.vx) * blend
        t.vy += (my * t.speed - t.vy) * blend
        t.x = min(R.MAP_SIZE - t.radius, max(t.radius, t.x + t.vx * R.DT))
        t.y = min(R.MAP_SIZE - t.radius, max(t.radius, t.y + t.vy * R.DT))

    # --------------------------------------------------------------- firing
    def _reset_barrels(self, t: Tank) -> None:
        t.barrel_timers = [b.delay * t.reload_time for b in t.tank_class.barrels]

    def _fire(self, t: Tank, fire: bool) -> None:
        cls, cooldown = t.tank_class, t.reload_time
        for i, barrel in enumerate(cls.barrels):
            t.barrel_timers[i] -= R.DT
            if not fire:
                # Idle barrels wait at their stagger offset so alternating barrels stay alternating.
                t.barrel_timers[i] = max(t.barrel_timers[i], barrel.delay * cooldown)
                continue
            if t.barrel_timers[i] > 0:
                continue
            t.barrel_timers[i] += cooldown
            angle = t.aim + barrel.angle
            if cls.spread:
                angle += self.rng.uniform(-cls.spread, cls.spread)
            ox = -math.sin(t.aim) * barrel.side_offset
            oy = math.cos(t.aim) * barrel.side_offset
            muzzle = t.radius + R.BULLET_RADIUS
            speed = t.bullet_speed
            self.bullets.append(Bullet(
                owner=t.id,
                x=t.x + math.cos(angle) * muzzle + ox,
                y=t.y + math.sin(angle) * muzzle + oy,
                vx=math.cos(angle) * speed + t.vx * 0.3,
                vy=math.sin(angle) * speed + t.vy * 0.3,
                damage=R.bullet_damage(t.stats["bullet_damage"]) * cls.damage,
                pen=R.bullet_penetration(t.stats["bullet_penetration"]),
                life=R.BULLET_LIFETIME * cls.lifetime,
            ))

    def _move_bullets(self) -> None:
        alive = []
        for b in self.bullets:
            b.x += b.vx * R.DT
            b.y += b.vy * R.DT
            b.life -= R.DT
            if b.life > 0 and 0 <= b.x <= R.MAP_SIZE and 0 <= b.y <= R.MAP_SIZE:
                alive.append(b)
        self.bullets = alive

    # ----------------------------------------------------------- collisions
    def _bullet_hits(self) -> None:
        survivors = []
        for b in self.bullets:
            for s in list(self.shapes):
                if b.pen <= 0:
                    break
                reach = s.spec.radius + R.BULLET_RADIUS
                key = f"s{s.id}"
                if key in b.hit:
                    continue
                if abs(s.x - b.x) < reach and abs(s.y - b.y) < reach and math.hypot(s.x - b.x, s.y - b.y) < reach:
                    b.hit.add(key)
                    s.hp -= b.damage
                    b.pen -= s.spec.toughness
                    if s.hp <= 0:
                        self._destroy_shape(s, self.tanks[b.owner])
            for t in self.tanks:
                if b.pen <= 0:
                    break
                if t.id == b.owner or not t.alive:
                    continue
                reach = t.radius + R.BULLET_RADIUS
                key = f"t{t.id}"
                if key in b.hit:
                    continue
                if abs(t.x - b.x) < reach and abs(t.y - b.y) < reach and math.hypot(t.x - b.x, t.y - b.y) < reach:
                    b.hit.add(key)
                    self._damage(t, b.damage, self.tanks[b.owner])
                    b.pen -= R.TANK_TOUGHNESS
            if b.pen > 0:
                survivors.append(b)
        self.bullets = survivors

    def _body_contacts(self) -> None:
        live = [t for t in self.tanks if t.alive]
        for t in live:
            for s in list(self.shapes):
                reach = t.radius + s.spec.radius
                dx, dy = t.x - s.x, t.y - s.y
                if abs(dx) >= reach or abs(dy) >= reach:
                    continue
                dist = math.hypot(dx, dy)
                if dist >= reach:
                    continue
                push = (reach - dist) / (dist or 1)
                t.x = min(R.MAP_SIZE - t.radius, max(t.radius, t.x + dx * push))
                t.y = min(R.MAP_SIZE - t.radius, max(t.radius, t.y + dy * push))
                self._damage(t, s.spec.body_damage * R.DT, None)
                s.hp -= R.body_damage(t.stats["body_damage"]) * R.DT
                if s.hp <= 0:
                    self._destroy_shape(s, t)
        for i, a in enumerate(live):
            for b in live[i + 1:]:
                reach = a.radius + b.radius
                dx, dy = b.x - a.x, b.y - a.y
                dist = math.hypot(dx, dy)
                if dist >= reach:
                    continue
                push = (reach - dist) / 2 / (dist or 1)
                a.x, a.y = a.x - dx * push, a.y - dy * push
                b.x, b.y = b.x + dx * push, b.y + dy * push
                self._damage(a, R.body_damage(b.stats["body_damage"]) * R.DT, b)
                self._damage(b, R.body_damage(a.stats["body_damage"]) * R.DT, a)

    def _damage(self, victim: Tank, amount: float, attacker: Tank | None) -> None:
        if not victim.alive or amount <= 0 or self.tick < victim.protected_until:
            return
        victim.hp -= amount
        victim.last_hit_by = attacker.id if attacker else None
        if victim.is_ai:
            source = "shape" if attacker is None else "opponent" if attacker.is_ai else "npc"
            victim.segment["damage_taken"][source] += amount
        if attacker is not None and attacker.is_ai and victim.is_ai:
            attacker.segment["damage_to_opponent"] += amount

    # ------------------------------------------------------- xp and levels
    def _gain_xp(self, t: Tank, amount: float, source: str) -> None:
        t.xp += amount
        t.score += amount
        if t.is_ai:
            t.segment["xp_by_source"][source] += round(amount)
        new_level = R.level_for_xp(t.xp)
        if new_level > t.level:
            old_max = t.max_hp
            self._set_level(t, new_level)
            t.hp += t.max_hp - old_max
            self.events.append({"type": "level", "tank": t.id, "level": t.level})

    def _set_level(self, t: Tank, level: int, reset: bool = False) -> None:
        if reset:
            t.stats = {s: 0 for s in R.STATS}
            if t.xp < R.xp_for_level(level):
                t.xp = float(R.xp_for_level(level))
        t.level = level
        while sum(t.stats.values()) < t.level - 1:
            stat = next_stat(t.strategy.build, t.stats, R.STAT_MAX)
            if stat is None:
                break
            t.stats[stat] += 1

    def _change_class(self, t: Tank, cls: str) -> None:
        t.cls = cls
        self._reset_barrels(t)
        self.events.append({"type": "class", "tank": t.id, "class": cls})

    # ------------------------------------------------------ death and life
    def _kill(self, victim: Tank) -> None:
        killer = self.tanks[victim.last_hit_by] if victim.last_hit_by is not None else None
        victim.alive = False
        victim.deaths += 1
        victim.hp = 0.0
        victim.respawn_tick = self.tick + round((R.RESPAWN_SECONDS if victim.is_ai else R.NPC_RESPAWN_SECONDS) * R.TICK_RATE)
        reward = (victim.xp if victim.is_ai else R.xp_for_level(victim.level)) * R.KILL_XP_SHARE
        if killer is not None and killer.alive:
            self._gain_xp(killer, reward, "opponent" if victim.is_ai else "npc")
            if killer.is_ai and victim.is_ai:
                killer.kills += 1
        cause = "shape" if killer is None else killer.name
        if victim.is_ai:
            victim.segment["deaths"].append({"t": round(self.seconds), "by": "shape" if killer is None else
                                             "opponent" if killer.is_ai else "npc"})
            victim.xp *= R.DEATH_XP_KEPT
            self._set_level(victim, R.level_for_xp(victim.xp), reset=True)
        if killer is not None and killer.is_ai:
            killer.segment["kills"].append({"t": round(self.seconds), "victim": "opponent" if victim.is_ai else "npc"})
        self.events.append({"type": "kill", "victim": victim.id, "killer": None if killer is None else killer.id,
                            "cause": cause, "victim_ai": victim.is_ai, "killer_ai": bool(killer and killer.is_ai)})

    def _respawn(self, t: Tank) -> None:
        if not t.is_ai:
            lo, hi = R.NPC_LEVELS
            t.xp = 0.0
            self._set_level(t, self.rng.randint(lo, hi), reset=True)
        self._place_at_edge(t, away_from=[o for o in self.tanks if o is not t and o.alive and (o.is_ai or not t.is_ai)])
        t.alive = True
        t.protected_until = self.tick + round(R.SPAWN_PROTECTION_SECONDS * R.TICK_RATE)
        t.vx = t.vy = 0.0
        t.hp = t.max_hp
        t.last_hit_by = None
        t.waypoint = None
        self._reset_barrels(t)
        self.events.append({"type": "respawn", "tank": t.id})

    def _place_at_edge(self, t: Tank, away_from: list[Tank] | None = None) -> None:
        """Pick the best of a few random edge spots, preferring ones far from `away_from`."""
        near, far = 60.0, R.MAP_SIZE - 60.0
        best, best_d = None, -1.0
        for _ in range(4 if away_from else 1):
            edge, pos = self.rng.randrange(4), self.rng.uniform(100, R.MAP_SIZE - 100)
            spot = [(near, pos), (far, pos), (pos, near), (pos, far)][edge]
            d = min((math.hypot(o.x - spot[0], o.y - spot[1]) for o in away_from), default=0.0) if away_from else 0.0
            if d > best_d:
                best, best_d = spot, d
        t.x, t.y = best

    # ---------------------------------------------------------------- shapes
    def _spawn_shape(self) -> None:
        for _ in range(6):
            x = self.rng.uniform(30, R.MAP_SIZE - 30)
            y = self.rng.uniform(30, R.MAP_SIZE - 30)
            if all(math.hypot(t.x - x, t.y - y) > 80 for t in self.tanks if t.alive):
                break
        centre = 1 - math.hypot(x - R.MAP_SIZE / 2, y - R.MAP_SIZE / 2) / (R.MAP_SIZE / 2 * math.sqrt(2))
        roll = self.rng.random()
        pentagon = 0.02 + 0.30 * centre**3
        kind = "pentagon" if roll < pentagon else "triangle" if roll < pentagon + 0.25 else "square"
        self.shapes.append(Shape(self._next_shape_id, kind, x, y, R.SHAPES[kind].hp, self.rng.uniform(0, 6.28)))
        self._next_shape_id += 1

    def _destroy_shape(self, s: Shape, by: Tank) -> None:
        if s in self.shapes:
            self.shapes.remove(s)
            if by.alive:
                self._gain_xp(by, s.spec.xp, s.kind)
