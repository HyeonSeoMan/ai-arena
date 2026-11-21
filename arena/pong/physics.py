"""Deterministic fixed-timestep Pong simulation with strategy-driven paddle bots.

Given the same starting state, strategies and RNG seed, a point always plays
out identically. That is what makes per-point replay verification possible.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

from arena.pong.strategy import Strategy

DT = 1 / 60
WIDTH = 160.0
HEIGHT = 90.0
BALL_R = 1.5
PADDLE_HALF = 7.0
PADDLE_X = {0: 3.0, 1: WIDTH - 3.0}  # side 0 = left, side 1 = right
PADDLE_SPEED = 60.0
BALL_SPEED = 90.0
SPEEDUP = 1.05
BALL_MAX = 170.0
MAX_HITS = 300

STAMINA_MAX = 100.0
STAMINA_DRAIN = 22.0  # per second at full base speed
STAMINA_REGEN = 12.0  # per second while (nearly) still
STAMINA_BETWEEN_POINTS = 30.0

MIN_ANGLE = math.radians(15)
MAX_ANGLE = math.radians(60)


@dataclass
class Paddle:
    y: float = HEIGHT / 2
    stamina: float = STAMINA_MAX
    frozen: float = 0.0  # seconds of post-power-shot recovery left
    guess_error: float = 0.0  # standard-normal draws fixed per incoming shot
    contact_error: float = 0.0


@dataclass
class Shot:
    tick: int
    side: int
    angle_deg: float
    speed: float
    target_y: float
    mishit: bool


@dataclass
class PointResult:
    winner: int | None  # None = rally limit reached, point replayed
    reason: str
    hits: int
    ticks: int
    miss_distance: float | None  # how far the loser's paddle edge was from the ball
    shots: list[Shot] = field(default_factory=list)
    stamina: tuple[float, float] = (0.0, 0.0)


class PointSim:
    """Simulates one point. Call step() until it returns a PointResult."""

    def __init__(self, seed: str, server: int, strategies: tuple[Strategy, Strategy], paddles: tuple[Paddle, Paddle]):
        self.rng = random.Random(seed)
        self.strategies = strategies
        self.paddles = paddles
        self.tick = 0
        self.hits = 0
        self.shots: list[Shot] = []
        self.result: PointResult | None = None
        self._passed = False  # ball already got past the receiving paddle
        self._miss_gap: float | None = None
        # Serve from the centre towards the receiver.
        receiver = 1 - server
        theta = self.rng.uniform(-math.radians(30), math.radians(30))
        direction = 1 if receiver == 1 else -1
        self.bx, self.by = WIDTH / 2, HEIGHT / 2
        self.vx = direction * BALL_SPEED * math.cos(theta)
        self.vy = BALL_SPEED * math.sin(theta)
        self._new_incoming(receiver)

    # ---- per-tick update ----
    def step(self) -> PointResult | None:
        if self.result:
            return self.result
        self.tick += 1
        for side in (0, 1):
            self._move_paddle(side)

        self.bx += self.vx * DT
        self.by += self.vy * DT
        if self.by < BALL_R:
            self.by, self.vy = 2 * BALL_R - self.by, -self.vy
        elif self.by > HEIGHT - BALL_R:
            self.by, self.vy = 2 * (HEIGHT - BALL_R) - self.by, -self.vy

        side = 0 if self.vx < 0 else 1
        face = PADDLE_X[side] + (BALL_R if side == 0 else -BALL_R)
        crossed = self.bx <= face if side == 0 else self.bx >= face
        if crossed and not self._passed:
            gap = abs(self.by - self.paddles[side].y) - (PADDLE_HALF + BALL_R)
            if gap <= 0:
                self.bx = face
                self._hit(side)
            else:
                self._passed = True
                self._miss_gap = gap
        if self.bx < -BALL_R or self.bx > WIDTH + BALL_R:
            loser = 0 if self.bx < 0 else 1
            self._finish(1 - loser, "miss", self._miss_gap)
        elif self.hits >= MAX_HITS:
            self._finish(None, "rally_limit", None)
        return self.result

    def frame(self) -> dict:
        return {
            "t": self.tick,
            "ball": [round(self.bx, 2), round(self.by, 2)],
            "paddles": [round(p.y, 2) for p in self.paddles],
            "stamina": [round(p.stamina, 1) for p in self.paddles],
            "speed": round(math.hypot(self.vx, self.vy), 1),
        }

    # ---- paddle bot ----
    def _move_paddle(self, side: int) -> None:
        p, s = self.paddles[side], self.strategies[side]
        if p.frozen > 0:
            p.frozen = max(0.0, p.frozen - DT)
            self._regen(p)
            return
        incoming = (self.vx < 0) == (side == 0)
        if incoming:
            dist = abs(self.bx - PADDLE_X[side])
            predicted = self._predict_y(side)
            # Early guesses are noisy; the error shrinks as the ball approaches.
            guess = predicted + p.guess_error * 18.0 * (dist / WIDTH)
            # Fast balls are harder to meet cleanly, whatever the strategy.
            fast = (math.hypot(self.vx, self.vy) - BALL_SPEED) / (BALL_MAX - BALL_SPEED)
            target = self.by + s.anticipation * (guess - self.by) + p.contact_error * (1.0 + 5.0 * fast)
        else:
            target = p.y + s.home * (HEIGHT / 2 - p.y)

        max_speed = PADDLE_SPEED * (0.55 + 0.45 * s.effort) * (0.6 + 0.4 * p.stamina / STAMINA_MAX)
        delta = target - p.y
        move = max(-max_speed * DT, min(max_speed * DT, delta)) if abs(delta) > 0.3 else 0.0
        p.y = min(HEIGHT - PADDLE_HALF, max(PADDLE_HALF, p.y + move))
        speed_frac = abs(move) / DT / PADDLE_SPEED
        if speed_frac < 0.15:
            self._regen(p)
        else:
            p.stamina = max(0.0, p.stamina - STAMINA_DRAIN * speed_frac**2 * DT)

    def _regen(self, p: Paddle) -> None:
        p.stamina = min(STAMINA_MAX, p.stamina + STAMINA_REGEN * DT)

    def _predict_y(self, side: int) -> float:
        """Where the ball will cross this paddle's x, including wall bounces."""
        t = (PADDLE_X[side] - self.bx) / self.vx if self.vx else 0.0
        y = self.by + self.vy * max(t, 0.0)
        span = HEIGHT - 2 * BALL_R
        y = (y - BALL_R) % (2 * span)
        return BALL_R + (y if y <= span else 2 * span - y)

    # ---- hitting ----
    def _hit(self, side: int) -> None:
        self.hits += 1
        s, p, opp = self.strategies[side], self.paddles[side], self.paddles[1 - side]
        rng = self.rng
        max_angle = MIN_ANGLE + (MAX_ANGLE - MIN_ANGLE) * s.angle
        speed = min(BALL_MAX, math.hypot(self.vx, self.vy) * SPEEDUP)

        target_y = self._aim_target(s, opp, rng)
        dx = abs(PADDLE_X[1 - side] - PADDLE_X[side])
        theta = math.atan2(target_y - self.by, dx)
        theta = max(-max_angle, min(max_angle, theta))
        theta += rng.gauss(0.0, math.radians(2 + 8 * s.power))

        mishit = rng.random() < 0.08 * s.power**2 + 0.12 * s.angle**2
        if mishit:
            theta = rng.uniform(-math.radians(10), math.radians(10))
            speed = BALL_SPEED * 0.8
        else:
            speed = min(BALL_MAX, speed * (1 + 0.35 * s.power))
            p.frozen = 0.3 * s.power

        direction = 1 if side == 0 else -1
        self.vx = direction * speed * math.cos(theta)
        self.vy = speed * math.sin(theta)
        self.shots.append(Shot(self.tick, side, round(math.degrees(theta), 1), round(speed, 1), round(target_y, 1), mishit))
        self._new_incoming(1 - side)

    def _aim_target(self, s: Strategy, opp: Paddle, rng: random.Random) -> float:
        aim = s.aim if s.aim != "random" else rng.choice(("away", "corners", "center"))
        if aim == "center":
            return HEIGHT / 2
        if aim == "corners":
            return 5.0 if rng.random() < 0.5 else HEIGHT - 5.0
        return HEIGHT - 5.0 if opp.y < HEIGHT / 2 else 5.0  # away from the opponent

    def _new_incoming(self, receiver: int) -> None:
        self.paddles[receiver].guess_error = self.rng.gauss(0.0, 1.0)
        self.paddles[receiver].contact_error = self.rng.gauss(0.0, 1.0)
        self._passed = False
        self._miss_gap = None

    def _finish(self, winner: int | None, reason: str, gap: float | None) -> None:
        self.result = PointResult(
            winner=winner,
            reason=reason,
            hits=self.hits,
            ticks=self.tick,
            miss_distance=None if gap is None else round(gap, 1),
            shots=self.shots,
            stamina=(round(self.paddles[0].stamina, 1), round(self.paddles[1].stamina, 1)),
        )


def between_points(paddles: tuple[Paddle, Paddle]) -> None:
    for p in paddles:
        p.stamina = min(STAMINA_MAX, p.stamina + STAMINA_BETWEEN_POINTS)
        p.y = HEIGHT / 2
        p.frozen = 0.0


def simulate_point(seed: str, server: int, strategies, paddles, max_ticks: int = 60 * 600) -> PointSim:
    sim = PointSim(seed, server, strategies, paddles)
    while sim.step() is None and sim.tick < max_ticks:
        pass
    return sim
