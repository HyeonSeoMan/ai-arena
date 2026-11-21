import asyncio
import json

import pytest

from arena.agents import HeuristicAgent, RandomAgent, ScriptedAgent, parse_move
from arena.engine import BLACK, WHITE, EndReason, Status
from arena.events import EventLog, load_events, verify
from arena.runner import MatchConfig, run_match

FAST = MatchConfig(turn_timeout=0.5, max_retries=2, match_timeout=30)


def run(black, white, config=FAST, log=None):
    log = log or EventLog("test")
    game = asyncio.run(run_match(black, white, log, config))
    return game, log


def types(log):
    return [e["type"] for e in log.events]


def move(r, c):
    return json.dumps({"row": r, "col": c})


# Black plays row 0, white plays row 2; black completes five on its 5th move.
BLACK_WINS = [move(0, c) for c in range(5)]
WHITE_FILLER = [move(2, c) for c in (0, 2, 4, 6, 8)]


@pytest.mark.parametrize("seed", range(5))
def test_dummy_agents_complete_a_match(seed):
    game, log = run(HeuristicAgent(seed=seed), RandomAgent(seed=seed))
    assert game.is_over
    assert types(log)[0] == "match_started"
    assert types(log)[-1] == "match_ended"
    assert verify(log.events).ok


def test_heuristic_beats_random_most_of_the_time():
    wins = 0
    for seed in range(10):
        game, _ = run(HeuristicAgent(seed=seed), RandomAgent(seed=seed))
        wins += game.winner == BLACK
    assert wins >= 8


def test_scripted_win_logs_every_move():
    game, log = run(ScriptedAgent(BLACK_WINS), ScriptedAgent(WHITE_FILLER))
    assert game.winner == BLACK
    assert game.end_reason == EndReason.FIVE_IN_ROW
    applied = [e for e in log.events if e["type"] == "move_applied"]
    assert [(e["player"], e["row"], e["col"]) for e in applied][:2] == [(BLACK, 0, 0), (WHITE, 2, 0)]
    assert len(applied) == 9


def test_invalid_response_is_retried_with_error_feedback():
    black = ScriptedAgent(["I think the centre is best", *BLACK_WINS])
    game, log = run(black, ScriptedAgent(WHITE_FILLER))
    assert game.winner == BLACK
    rejected = [e for e in log.events if e["type"] == "move_rejected"]
    assert len(rejected) == 1 and rejected[0]["reason"] == "invalid_response"
    assert black.observations[0].last_error is None
    assert black.observations[1].attempt == 2
    assert black.observations[1].last_error.startswith("invalid_response")


def test_illegal_move_is_retried():
    white = ScriptedAgent([move(0, 0), *WHITE_FILLER])  # (0, 0) is taken by black's first move
    game, log = run(ScriptedAgent(BLACK_WINS), white)
    assert game.winner == BLACK
    rejected = [e for e in log.events if e["type"] == "move_rejected"]
    assert rejected[0]["reason"] == "occupied"
    assert white.observations[1].last_error.startswith("occupied")


def test_too_many_invalid_attempts_forfeits():
    game, log = run(ScriptedAgent(["nope", "still no", "{}"]), ScriptedAgent([]))
    assert game.status == Status.WON
    assert game.winner == WHITE
    assert game.end_reason == EndReason.FORFEIT
    assert "3 invalid attempts" in game.end_detail
    assert verify(log.events).ok


def test_agent_exception_counts_as_invalid_attempt():
    black = ScriptedAgent([RuntimeError("cli crashed"), *BLACK_WINS])
    game, log = run(black, ScriptedAgent(WHITE_FILLER))
    assert game.winner == BLACK
    responded = [e for e in log.events if e["type"] == "agent_responded"]
    assert "cli crashed" in responded[0]["error"]


def test_turn_timeout_forfeits_without_retry():
    game, log = run(ScriptedAgent([move(4, 4)]), ScriptedAgent([5.0]))
    assert game.winner == BLACK
    assert game.end_reason == EndReason.FORFEIT
    assert "no response" in game.end_detail
    assert "agent_timeout" in types(log)
    assert verify(log.events).ok


def test_match_timeout_ends_in_draw():
    config = MatchConfig(turn_timeout=5, max_retries=2, match_timeout=0.3)
    game, log = run(ScriptedAgent([move(4, 4)]), ScriptedAgent([5.0]), config=config)
    assert game.status == Status.DRAW
    assert game.end_reason == EndReason.TIME_LIMIT
    assert verify(log.events).ok


def test_observation_contains_only_public_state():
    black, white = ScriptedAgent(BLACK_WINS), ScriptedAgent(WHITE_FILLER)
    run(black, white)
    obs = white.observations[0]
    assert obs.color == WHITE
    assert obs.moves == [(BLACK, 0, 0)]
    assert obs.board[0][0] == BLACK
    assert obs.owner_notes is None


def test_log_file_roundtrip_and_tamper_detection(tmp_path):
    path = tmp_path / "m.jsonl"
    log = EventLog("file-test", path)
    run(HeuristicAgent(seed=1), RandomAgent(seed=1), log=log)
    log.close()

    events = load_events(path)
    assert events == json.loads(json.dumps(log.events))
    assert all(e["audience"] == "public" for e in events)
    assert [e["seq"] for e in events] == list(range(len(events)))
    assert verify(events).ok

    events[-1]["winner"] = 3 - (events[-1]["winner"] or 1)
    assert not verify(events).ok


def test_listener_receives_events_live():
    seen = []
    log = EventLog("live")
    log.subscribe(lambda e: seen.append(e["type"]))
    run(ScriptedAgent(BLACK_WINS), ScriptedAgent(WHITE_FILLER), log=log)
    assert seen == types(log)


@pytest.mark.parametrize(
    "raw,expected",
    [
        ('{"row": 3, "col": 5}', (3, 5)),
        ('Sure! My move is {"row": 1, "col": 2}.', (1, 2)),
        ('```json\n{"col": 7, "row": 0}\n```', (0, 7)),
        ('{"thinking": "x"} then {"row": 4, "col": 4}', (4, 4)),
    ],
)
def test_parse_move_extracts_json(raw, expected):
    reply = parse_move(raw)
    assert reply.error is None
    assert (reply.row, reply.col) == expected


@pytest.mark.parametrize("raw", ["", "row 4 col 4", '{"row": 4}', "{broken"])
def test_parse_move_reports_error(raw):
    assert parse_move(raw).error is not None
