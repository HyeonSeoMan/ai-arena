import asyncio
import json
import math
import time

import pytest

from arena.events import EventLog
from arena.pong import physics
from arena.pong.coach import Briefing, FixedCoach, LLMCoach, RandomCoach, ScriptedCoach, build_coach_prompt, parse_reply
from arena.pong.match import PongConfig, run_pong
from arena.pong.physics import Paddle, PointSim, simulate_point
from arena.pong.replay import verify_pong
from arena.pong.strategy import Strategy, StrategyError, parse_strategy

FAST = PongConfig(points_to_win=3, timeout_seconds=0.3, realtime=False)
GOOD = json.dumps({"aim": "corners", "angle": 0.8, "power": 0.1, "effort": 0.9, "anticipation": 0.7, "home": 0.5, "note": "모서리 공략"})


def run(left, right, config=FAST, seed="s", log=None):
    log = log or EventLog("pong-test")
    winner, score = asyncio.run(run_pong(left, right, log, config, seed=seed))
    return winner, score, log


def of(log, type_):
    return [e for e in log.events if e["type"] == type_]


# ---- strategy ----

def test_parse_strategy_clamps_and_truncates():
    s = parse_strategy({"aim": "away", "angle": 2, "power": -1, "effort": 0.5, "anticipation": 1, "home": 0, "note": "x" * 500})
    assert (s.angle, s.power) == (1.0, 0.0)
    assert len(s.note) == 200


@pytest.mark.parametrize(
    "data",
    [[], {"aim": "smash"}, {"aim": "away", "angle": "high", "power": 0, "effort": 0, "anticipation": 0, "home": 0},
     {"aim": "away", "angle": True, "power": 0, "effort": 0, "anticipation": 0, "home": 0}],
)
def test_parse_strategy_rejects_bad_input(data):
    with pytest.raises(StrategyError):
        parse_strategy(data)


@pytest.mark.parametrize("text", [GOOD, f"작전입니다:\n```json\n{GOOD}\n```"])
def test_parse_reply_accepts_prose_and_fences(text):
    assert parse_reply(text).strategy.aim == "corners"


def test_parse_reply_reports_errors():
    assert parse_reply("no idea").error
    assert parse_reply('{"aim": "nope"}').error


# ---- physics ----

def test_point_is_deterministic():
    a = simulate_point("seed:1", 0, (Strategy(), Strategy()), (Paddle(), Paddle()))
    b = simulate_point("seed:1", 0, (Strategy(), Strategy()), (Paddle(), Paddle()))
    assert (a.result.winner, a.result.hits, a.result.ticks) == (b.result.winner, b.result.hits, b.result.ticks)
    assert a.frame() == b.frame()


def test_different_strategies_change_the_point():
    results = {
        simulate_point("seed:2", 0, (Strategy(aim=aim, angle=angle), Strategy()), (Paddle(), Paddle())).result.ticks
        for aim in ("away", "center", "corners") for angle in (0.1, 0.9)
    }
    assert len(results) > 1


def test_ball_stays_inside_walls_and_points_end():
    sim = PointSim("seed:3", 1, (Strategy(), Strategy()), (Paddle(), Paddle()))
    while sim.step() is None:
        assert physics.BALL_R - 1e-9 <= sim.by <= physics.HEIGHT - physics.BALL_R + 1e-9
        for p in sim.paddles:
            assert physics.PADDLE_HALF <= p.y <= physics.HEIGHT - physics.PADDLE_HALF
            assert 0 <= p.stamina <= physics.STAMINA_MAX
    assert sim.result.winner in (0, 1)
    assert sim.result.reason == "miss"


def test_serve_goes_to_receiver():
    assert PointSim("x", 0, (Strategy(), Strategy()), (Paddle(), Paddle())).vx > 0
    assert PointSim("x", 1, (Strategy(), Strategy()), (Paddle(), Paddle())).vx < 0


def test_prediction_accounts_for_wall_bounces():
    sim = PointSim("x", 0, (Strategy(), Strategy()), (Paddle(), Paddle()))
    sim.bx, sim.by, sim.vx, sim.vy = 80.0, 80.0, 100.0, 100.0
    t = (physics.PADDLE_X[1] - 80.0) / 100.0
    unfolded = 80.0 + 100.0 * t  # 157: well past the bottom wall
    assert unfolded > physics.HEIGHT
    predicted = sim._predict_y(1)
    assert physics.BALL_R <= predicted <= physics.HEIGHT - physics.BALL_R
    # Step the real ball to the paddle line and compare.
    while sim.bx < physics.PADDLE_X[1] - 1:
        sim.bx += sim.vx * physics.DT
        sim.by += sim.vy * physics.DT
        if sim.by > physics.HEIGHT - physics.BALL_R:
            sim.by, sim.vy = 2 * (physics.HEIGHT - physics.BALL_R) - sim.by, -sim.vy
        if sim.by < physics.BALL_R:
            sim.by, sim.vy = 2 * physics.BALL_R - sim.by, -sim.vy
    assert math.isclose(sim.by, predicted, abs_tol=2.0)


def test_low_effort_drains_less_stamina():
    tired = []
    for effort in (0.2, 1.0):
        pads = (Paddle(), Paddle())
        simulate_point("seed:4", 0, (Strategy(effort=effort, anticipation=1.0), Strategy()), pads)
        tired.append(pads[0].stamina)
    assert tired[0] >= tired[1]


# ---- match ----

def test_match_completes_and_replays():
    winner, score, log = run(RandomCoach(seed=1), FixedCoach())
    assert max(score) == 3 and score[winner] == 3
    assert log.events[0]["type"] == "match_started" and log.events[-1]["type"] == "match_ended"
    assert len(of(log, "point_started")) == len(of(log, "point_ended"))
    assert of(log, "hit")
    check = verify_pong(log.events)
    assert check.ok, check.problems


def test_replay_detects_tampering():
    _, _, log = run(RandomCoach(seed=2), FixedCoach())
    events = json.loads(json.dumps(log.events))
    first = next(e for e in events if e["type"] == "point_started")
    first["strategies"][0]["power"] = 1.0 - first["strategies"][0]["power"]
    first["paddles"][0]["y"] += 5
    assert not verify_pong(events).ok


def test_both_coaches_asked_in_parallel_each_timeout():
    config = PongConfig(points_to_win=2, timeout_seconds=2.0, realtime=False)
    t0 = time.monotonic()
    _, score, log = run(FixedCoach(delay=0.2, name="a"), FixedCoach(delay=0.2, name="b"), config=config)
    elapsed = time.monotonic() - t0
    timeouts = of(log, "timeout_started")
    assert elapsed < len(timeouts) * 0.2 * 1.6  # sequential calls would take 2 x 0.2s per timeout
    assert timeouts[0]["phase"] == "pregame"
    assert len(timeouts) == sum(score)  # pregame + one per point except the last
    for started in timeouts:
        i = log.events.index(started)
        sides = {e["side"] for e in log.events[i:i + 3] if e["type"] == "strategy_set"}
        assert sides == {0, 1}


def test_slow_invalid_or_crashing_coach_keeps_previous_strategy():
    left = ScriptedCoach([GOOD, 1.0, "garbage", RuntimeError("cli crashed")], name="left")
    _, _, log = run(left, FixedCoach(), config=PongConfig(points_to_win=4, timeout_seconds=0.3, realtime=False))
    kept = [e for e in of(log, "strategy_kept") if e["side"] == 0]
    reasons = [k["reason"] for k in kept[:3]]
    assert "no answer" in reasons[0]
    assert "no JSON" in reasons[1]
    assert "cli crashed" in reasons[2]
    # The strategy set in the pregame stays in force for every point after the failures.
    for e in of(log, "point_started"):
        assert e["strategies"][0]["aim"] == "corners"


def test_briefing_hides_opponent_strategy():
    left = ScriptedCoach([], name="left")
    secret = Strategy(aim="center", angle=0.123, power=0.456, note="비밀 작전")
    run(left, FixedCoach(secret))
    later = left.briefings[-1]
    text = json.dumps(later.__dict__, ensure_ascii=False)
    assert "비밀 작전" not in text and "0.456" not in text and "0.123" not in text
    assert later.rallies and set(later.rallies[0]) >= {"won", "end", "hits", "you", "opponent"}
    assert later.current["aim"] == "away"  # its own (default) strategy


def test_briefing_perspective_is_mirrored():
    left, right = ScriptedCoach([], name="l"), ScriptedCoach([], name="r")
    run(left, right)
    lb, rb = left.briefings[-1], right.briefings[-1]
    assert lb.score == rb.score[::-1]
    assert [r["won"] for r in lb.rallies] == [None if w is None else not w for w in (r["won"] for r in rb.rallies)]


def test_coach_prompt_contents():
    b = Briefing(side=1, phase="timeout", point=3, score=(1, 2), points_to_win=7, seconds=20,
                 current=Strategy().to_dict(), rallies=[], owner_notes="수비적으로")
    text = build_coach_prompt(b)
    assert "First to 7 points" in text and '"your_side": "right"' in text
    assert "수비적으로" in text and "ONLY a JSON object" in text


class FakeCompleter:
    name = "fake"

    def __init__(self, text, error=None):
        self.text, self.error = text, error
        self.prompts = []

    async def complete(self, prompt, schema=None):
        from arena.agents.cli import Completion

        self.prompts.append((prompt, schema))
        return Completion("" if self.error else self.text, self.text, self.error)


def test_llm_coach_uses_agent_completion_with_schema():
    agent = FakeCompleter(GOOD)
    b = Briefing(0, "pregame", 0, (0, 0), 7, 20, Strategy().to_dict(), [])
    reply = asyncio.run(LLMCoach(agent).plan(b))
    assert reply.strategy.aim == "corners"
    assert agent.prompts[0][1]["required"][0] == "aim"
    failed = asyncio.run(LLMCoach(FakeCompleter("usage limit", error="claude reported an error")).plan(b))
    assert failed.strategy is None and failed.error


def test_realtime_mode_broadcasts_frames_but_does_not_log_them():
    frames = []
    log = EventLog("rt")
    log.subscribe(lambda e: frames.append(e) if e["type"] == "frame" else None)
    config = PongConfig(points_to_win=1, timeout_seconds=0.2, realtime=True, serve_pause=0.0)
    # Patch a short point: right coach never moves so it misses quickly.
    lazy = Strategy(effort=0.0, anticipation=0.0, home=0.0)
    run(FixedCoach(), FixedCoach(lazy), config=config, log=log)
    assert frames and "ball" in frames[0]
    assert not of(log, "frame")
