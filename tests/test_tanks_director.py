import asyncio
import json
from dataclasses import replace
from pathlib import Path

import pytest

from arena.agents.cli import Completion
from arena.events import EventLog, load_events
from arena.tanks import rules as R
from arena.tanks.analysis import director_effects
from arena.tanks.coach import FixedCoach, ScriptedCoach
from arena.tanks.director import QUICK, Director, classify
from arena.tanks.match import TanksConfig, run_tanks
from arena.tanks.replay import verify_tanks
from arena.tanks.strategy import Strategy

FAST = TanksConfig(timeout_seconds=0.5, realtime=False, yield_every=1, quiet_seconds=999, match_seconds=40)


class FakeAgent:
    """Stands in for a CLI agent: answers after `delay` event-loop turns (= ticks in fast mode)."""

    name = "fake"

    def __init__(self, answer, delay=5, error=None):
        self.answer, self.delay, self.error = answer, delay, error
        self.prompts = []
        self.cancelled = 0

    async def complete(self, prompt, schema=None):
        self.prompts.append(prompt)
        try:
            for _ in range(self.delay):
                await asyncio.sleep(0)
        except asyncio.CancelledError:
            self.cancelled += 1
            raise
        text = self.answer(prompt) if callable(self.answer) else self.answer
        return Completion("" if self.error else text, text, self.error)


def interpretation(**changes):
    strategy = replace(Strategy(), **changes).to_dict()
    return json.dumps({"strategy": strategy, "interpretation": "변경함", "opinion": "위험할 수 있습니다",
                       "feasible": True}, ensure_ascii=False)


def run(director, script, config=FAST, seed="dir", second=None):
    """Run a match; `script` is a list of (tick, text) the director submits when the world reaches that tick."""
    log = EventLog("dir-test")
    replies = []

    async def main():
        async def feed():
            for tick, text in script:
                while director._tick() < tick or not director.active:
                    await asyncio.sleep(0)
                replies.append(director.submit(text))

        feeder = asyncio.create_task(feed())
        result = await run_tanks(FixedCoach(name="me"), second or FixedCoach(name="opp"), log, config,
                                 seed=seed, director=director)
        feeder.cancel()
        return result

    asyncio.run(main())
    return log, replies


def of(log, type_, **match):
    return [e for e in log.events if e["type"] == type_ and all(e.get(k) == v for k, v in match.items())]


# ---- classification ----

@pytest.mark.parametrize("text,qid", [
    ("공격적으로", "attack"), ("압박해!", "attack"), ("거리 벌려", "distance"), ("후퇴해", "retreat"),
    ("중앙으로 가", "center"), ("농사해", "farm"), ("careful", "careful"),
])
def test_classify_quick(text, qid):
    assert classify(text) == qid


@pytest.mark.parametrize("text", [
    "상대 체력이 얼마 안 남았으니 마지막으로 한번 크게 압박해보고 안 되면 빠져",  # long -> LLM
    "공격하다가 위험하면 후퇴",  # two intents -> LLM
    "엄폐물 뒤에 숨어",  # unknown -> LLM
])
def test_classify_goes_to_llm(text):
    assert classify(text) is None


# ---- quick commands ----

def test_quick_command_applies_next_tick_without_pausing():
    d = Director(side=0)
    log, replies = run(d, [(100, "공격적으로")])
    assert replies[0]["kind"] == "quick"
    applied = of(log, "strategy_applied", source="director")
    assert applied[0]["tick"] == 100 and applied[0]["command_id"] == replies[0]["id"]
    s = applied[0]["strategies"][0]
    assert s["aggression"] >= 0.75 and s["seek"] >= 0.8
    before = [e for e in of(log, "strategy_applied") if e["tick"] < 100][-1]
    assert applied[0]["strategies"][1] == before["strategies"][1]  # the other side is untouched
    assert of(log, "director_result", ok=True)[0]["interpretation"] == QUICK["attack"][0]
    # The command itself never pauses the match: no timeout opens at the moment it is applied.
    assert all(round(e["match_time"] * R.TICK_RATE) != 100 for e in of(log, "timeout_started"))
    assert verify_tanks(log.events).ok


def test_temporary_retreat_reverts_after_ten_seconds():
    d = Director(side=0)
    log, _ = run(d, [(60, "후퇴해")])
    applied = of(log, "strategy_applied", source="director")
    assert applied[0]["strategies"][0]["retreat_hp"] == 1.0 and not applied[0]["revert"]
    assert applied[1]["revert"] and applied[1]["tick"] == 60 + round(10 * R.TICK_RATE)
    assert applied[1]["strategies"][0]["retreat_hp"] == Strategy().retreat_hp
    assert verify_tanks(log.events).ok


# ---- LLM commands ----

def test_llm_command_runs_in_background_and_applies_only_changed_fields():
    agent = FakeAgent(interpretation(engage_range=0.95, note="멀리서"), delay=40)
    d = Director(side=0, agent=agent)
    log, replies = run(d, [(100, "엄폐물 뒤에 숨어서 멀리서 쏴")])
    assert replies[0]["kind"] == "llm"
    applied = of(log, "strategy_applied", source="director")[0]
    assert applied["tick"] > 100 + 30  # the match kept running while the coach thought
    assert applied["strategies"][0]["engage_range"] == 0.95
    result = of(log, "director_result", id=replies[0]["id"])[0]
    assert result["ok"] and result["opinion"] == "위험할 수 있습니다"
    assert set(result["changes"]) == {"engage_range", "note"}
    assert "엄폐물 뒤에 숨어서 멀리서 쏴" in agent.prompts[0]
    assert verify_tanks(log.events).ok


def test_new_llm_command_supersedes_the_pending_one():
    agent = FakeAgent(interpretation(strafe=1.0), delay=50)
    d = Director(side=0, agent=agent)
    log, replies = run(d, [(100, "첫 번째 긴 지시를 내린다 무엇이든 해봐"), (110, "두 번째 긴 지시로 바꾼다 좌우로 피해")])
    first = of(log, "director_result", id=replies[0]["id"])[0]
    assert not first["ok"] and "superseded" in first["error"]
    assert agent.cancelled == 1
    assert of(log, "director_result", id=replies[1]["id"])[0]["ok"]
    assert len(of(log, "strategy_applied", source="director")) == 1


@pytest.mark.parametrize("answer,error", [("모르겠어요", None), ('{"strategy": {"aim": 1}}', None), ("", "usage limit")])
def test_failed_interpretation_keeps_strategy(answer, error):
    d = Director(side=0, agent=FakeAgent(answer, delay=3, error=error))
    log, replies = run(d, [(50, "이상한 지시를 길게 내려본다 아무거나")])
    assert not of(log, "director_result", id=replies[0]["id"])[0]["ok"]
    assert not of(log, "strategy_applied", source="director")


def test_llm_limit_and_no_agent():
    d = Director(side=0, agent=FakeAgent(interpretation(), delay=1), max_llm=1)
    _, replies = run(d, [(30, "긴 자연어 지시 하나를 보낸다 첫번째"), (60, "긴 자연어 지시 하나를 보낸다 두번째")])
    assert replies[0]["ok"] and not replies[1]["ok"] and "limit" in replies[1]["error"]
    _, replies = run(Director(side=0), [(30, "긴 자연어 지시 하나를 보낸다 첫번째")])
    assert not replies[0]["ok"]


def test_infeasible_instruction_is_flagged():
    answer = json.dumps({"strategy": replace(Strategy(), engage_range=0.9).to_dict(),
                         "interpretation": "엄폐물이 없어 거리 유지로 대체", "opinion": "", "feasible": False},
                        ensure_ascii=False)
    d = Director(side=0, agent=FakeAgent(answer, delay=2))
    log, replies = run(d, [(30, "엄폐물 뒤로 숨어서 기다려라 지금")])
    result = of(log, "director_result", id=replies[0]["id"])[0]
    assert result["ok"] and result["feasible"] is False


def test_director_notes_reach_only_own_coach_timeouts():
    me, opp = ScriptedCoach([], name="me"), ScriptedCoach([], name="opp")
    d = Director(side=0)
    log = EventLog("notes")
    config = replace(FAST, quiet_seconds=5, match_seconds=12)

    async def main():
        async def feed():
            while d._tick() < 30 or not d.active:
                await asyncio.sleep(0)
            d.submit("공격적으로")
        task = asyncio.create_task(feed())
        await run_tanks(me, opp, log, config, seed="n", director=d)
        task.cancel()

    asyncio.run(main())
    assert me.briefings[0].owner_notes is None  # pregame, before any instruction
    assert "공격적으로" in me.briefings[-1].owner_notes
    assert all(b.owner_notes is None for b in opp.briefings)


def test_director_wins_over_a_timeout_that_happens_while_interpreting():
    agent = FakeAgent(interpretation(center_bias=0.0), delay=80)
    d = Director(side=0, agent=agent)
    config = replace(FAST, quiet_seconds=4)  # a quiet timeout at tick 120 lands mid-interpretation
    log, replies = run(d, [(100, "가장자리로 가서 버텨라 당분간 계속")], config=config)
    applied = [e for e in of(log, "strategy_applied") if e["tick"] >= 100]
    sources = [e["source"] for e in applied]
    assert "timeout" in sources and sources.index("director") > sources.index("timeout")
    assert verify_tanks(log.events).ok


# ---- analysis ----

def test_director_effects_compare_against_counterfactual():
    d = Director(side=0)
    config = replace(FAST, match_seconds=60)
    log, _ = run(d, [(900, "공격적으로")], config=config, seed="probe0")
    effects = director_effects(log.events)
    assert len(effects) == 1
    eff = effects[0]
    assert eff["text"] == "공격적으로" and eff["kind"] == "quick"
    assert "aggression" in eff["changes"] and "seek" in eff["changes"]
    assert 0 < eff["effect"]["seconds"] <= 30.0
    assert eff["effect"]["compared_seconds"] <= eff["effect"]["seconds"]
    assert set(eff["effect"]["delta"]) >= {"avg_distance", "damage_dealt", "xp_gained"}
    assert "events" in eff["effect"]["with"]


def test_compare_averages_distance_over_ticks_alive_in_both_branches():
    from arena.tanks.analysis import compare
    from arena.tanks.world import World

    w = World("cmp")
    while w.tick < 600:
        w.step()
    w.tanks[1].last_hit_by = 0
    w.tanks[1].hp = 0  # the opponent dies on the next tick in both branches
    result = compare(w, 0, replace(Strategy(), aggression=1.0), 300)
    dead_ticks = round(R.RESPAWN_SECONDS * R.TICK_RATE)
    assert result["compared_seconds"] <= round((300 - dead_ticks) * R.DT, 1)
    assert any(ev["event"] == "you killed the opponent" for ev in result["with"]["events"])


def test_quick_center_and_edge_move_the_tank():
    import copy
    import math

    from arena.tanks.director import QUICK
    from arena.tanks.world import World

    import statistics

    moved = {}
    for qid in ("center", "edge"):
        deltas = []  # median over several game states: a death in one branch can flip a single sample
        for seed in range(8):
            w = World(f"zone{seed}")
            while w.tick < 1200:
                w.step()
            a, b = copy.deepcopy(w), copy.deepcopy(w)
            a.set_strategy(0, replace(a.tanks[0].strategy, **QUICK[qid][2](a.tanks[0].strategy)))
            for _ in range(300):
                a.step()
                b.step()
            deltas.append(math.hypot(a.tanks[0].x - 600, a.tanks[0].y - 600)
                          - math.hypot(b.tanks[0].x - 600, b.tanks[0].y - 600))
        moved[qid] = statistics.median(deltas)
    assert moved["center"] < -30 and moved["edge"] > 30


def test_published_tank_records_still_replay():
    for path in Path("matches/tanks").glob("*.jsonl"):
        assert verify_tanks(load_events(path)).ok, path
