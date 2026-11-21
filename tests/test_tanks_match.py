import asyncio
import json
from dataclasses import replace

from arena.events import EventLog
from arena.tanks import rules as R
from arena.tanks.coach import Briefing, FixedCoach, RandomCoach, ScriptedCoach, build_coach_prompt
from arena.tanks.match import TanksConfig, run_tanks
from arena.tanks.replay import verify_tanks
from arena.tanks.strategy import Strategy

FAST = TanksConfig(timeout_seconds=0.5, realtime=False)
AGGRO = replace(Strategy(), aggression=0.9, note="공격")


def run(a, b, config=FAST, seed="match", log=None):
    log = log or EventLog("tanks-test")
    winner, reason = asyncio.run(run_tanks(a, b, log, config, seed=seed))
    return winner, reason, log


def of(log, type_):
    return [e for e in log.events if e["type"] == type_]


def test_match_completes_and_replays():
    winner, reason, log = run(RandomCoach(seed=3), FixedCoach())
    end = log.events[-1]
    assert end["type"] == "match_ended" and end["winner"] == winner
    assert reason in ("kills", "score", "score_tie")
    if reason == "kills":
        assert end["kills"][winner] == R.KILLS_TO_WIN
    check = verify_tanks(log.events)
    assert check.ok, check.problems


def test_replay_detects_tampering():
    _, _, log = run(RandomCoach(seed=4), FixedCoach())
    events = json.loads(json.dumps(log.events))
    first = next(e for e in events if e["type"] == "strategy_applied")
    first["strategies"][0]["aggression"] = 1.0 - first["strategies"][0]["aggression"]
    first["strategies"][0]["build"] = ["body_damage"] * 10
    assert not verify_tanks(events).ok


def test_time_limit_decides_by_score():
    config = replace(FAST, match_seconds=20, quiet_seconds=999)
    winner, reason, log = run(FixedCoach(), FixedCoach(), config=config)
    end = log.events[-1]
    assert reason in ("score", "score_tie")
    assert end["tick"] == 20 * R.TICK_RATE
    if reason == "score":
        assert end["scores"][winner] > end["scores"][1 - winner]


def test_deaths_trigger_timeouts_and_strategies_apply_at_that_tick():
    _, _, log = run(FixedCoach(AGGRO), FixedCoach(AGGRO), seed="brawl")
    phases = [e["phase"] for e in of(log, "timeout_started")]
    assert phases[0] == "pregame"
    assert "death" in phases
    kills = [e for e in of(log, "kill") if e["victim_ai"]]
    applied_ticks = {e["tick"] for e in of(log, "strategy_applied")}
    assert all(k["tick"] in applied_ticks for k in kills[:-1])  # the last kill may end the match


def test_quiet_timeout_fires():
    config = replace(FAST, quiet_seconds=10, match_seconds=35)
    passive = replace(Strategy(), aggression=0.0)
    _, _, log = run(FixedCoach(passive), FixedCoach(passive), config=config)
    assert "quiet" in [e["phase"] for e in of(log, "timeout_started")]


def test_timeouts_are_capped():
    config = replace(FAST, quiet_seconds=2, max_timeouts=4, match_seconds=60)
    _, _, log = run(FixedCoach(), FixedCoach(), config=config)
    assert len(of(log, "timeout_started")) == 4
    assert log.events[-1]["timeouts"] == 4


def test_level15_timeout_lets_the_coach_pick_the_class():
    sniper = '{"build": ["reload"], "class_choice": "sniper", "farm_target": "valuable", "npc_policy": "ignore", ' \
             '"aggression": 0.0, "engage_range": 0.5, "retreat_hp": 0.5, "strafe": 0.5, "aim_lead": 0.5, ' \
             '"center_bias": 0.7, "note": "스나이퍼"}'

    class Picker(ScriptedCoach):
        class_briefings = []

        async def plan(self, briefing):
            if briefing.choose_class_now:
                self.class_briefings.append(briefing)
                from arena import coaching
                from arena.tanks.strategy import parse_strategy
                return coaching.reply_from_text(sniper, parse_strategy)
            return await super().plan(briefing)

    passive = replace(Strategy(), aggression=0.0, class_choice="flank")
    picker = Picker([json.dumps(passive.to_dict())] * 30, name="picker")
    config = replace(FAST, quiet_seconds=999)
    _, _, log = run(picker, FixedCoach(passive), config=config, seed="farm")
    changes = [e for e in of(log, "class_changed") if e["tank"] == 0]
    assert picker.class_briefings, "tank 0 should reach level 15 while farming"
    assert picker.class_briefings[0].you["level"] >= R.CLASS_LEVEL
    assert changes and changes[0]["cls"] == "sniper"


def test_failing_coach_keeps_previous_strategy():
    good = json.dumps(AGGRO.to_dict())
    coach = ScriptedCoach([good, 2.0, "nonsense", RuntimeError("cli crashed")], name="flaky")
    config = replace(FAST, quiet_seconds=5, match_seconds=40)
    _, _, log = run(coach, FixedCoach(), config=config)
    kept = [e["reason"] for e in of(log, "strategy_kept") if e["side"] == 0]
    assert "no answer" in kept[0] and "no JSON" in kept[1] and "cli crashed" in kept[2]
    applied = of(log, "strategy_applied")
    assert all(a["strategies"][0]["note"] == "공격" for a in applied[:4])


def test_briefing_hides_opponent_plan():
    spy = ScriptedCoach([], name="spy")
    secret = replace(Strategy(), aggression=0.123, build=("body_damage",) * 30, note="비밀 작전")
    run(spy, FixedCoach(secret), config=replace(FAST, quiet_seconds=10, match_seconds=40))
    text = json.dumps([b.__dict__ for b in spy.briefings], ensure_ascii=False)
    assert "비밀 작전" not in text and "0.123" not in text
    later = spy.briefings[-1]
    assert set(later.opponent) == {"kills", "deaths", "score"}
    assert later.you["stats"] and "xp_by_source" in later.since_last


def test_briefing_triggers_are_perspective_aware():
    a, b = ScriptedCoach([], name="a"), ScriptedCoach([], name="b")
    run(a, b, seed="brawl2")
    for ba, bb in zip(a.briefings, b.briefings):
        if ba.phase == "death":
            assert ("you were killed" in ba.trigger) != ("you were killed" in bb.trigger) or "shape" in ba.trigger \
                or "NPC" in ba.trigger


def test_prompt_contents():
    b = Briefing(side=0, phase="level15", trigger="you reached level 15", seconds=20, match_time=100, time_left=260,
                 kills_to_win=3, you={"level": 15}, opponent={"kills": 0, "deaths": 1, "score": 900},
                 current=Strategy().to_dict(), since_last={}, last_seen_opponent=None, choose_class_now=True,
                 owner_notes="스나이퍼로 가")
    text = build_coach_prompt(b)
    assert "diep.io" in text and "class_choice you give now is applied immediately" in text
    assert "스나이퍼로 가" in text and "ONLY a JSON object" in text


def test_realtime_broadcasts_frames_without_logging_them():
    frames = []
    log = EventLog("rt")
    log.subscribe(lambda e: frames.append(e) if e["type"] == "frame" else None)
    config = replace(FAST, realtime=True, match_seconds=0.5, start_pause=0.0)
    run(FixedCoach(), FixedCoach(), config=config, log=log)
    assert frames and "tanks" in frames[0] and "status" in frames[-1]
    assert not of(log, "frame")
