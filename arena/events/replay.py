"""Rebuild a game from its event log and check it matches the recorded result."""

from __future__ import annotations

from dataclasses import dataclass

from arena.engine import EndReason, Game
from arena.events.log import Event


@dataclass(frozen=True)
class ReplayCheck:
    ok: bool
    problems: list[str]
    game: Game


def replay(events: list[Event]) -> Game:
    """Re-apply recorded moves, then any non-move ending (forfeit / time limit)."""
    game = Game()
    for event in events:
        if event["type"] == "move_applied":
            game.play(event["row"], event["col"])
        elif event["type"] == "match_ended" and not game.is_over:
            reason = event["reason"]
            if reason == EndReason.FORFEIT:
                game.forfeit(event["loser"], event.get("detail") or "")
            elif reason == EndReason.TIME_LIMIT:
                game.expire()
    return game


def verify(events: list[Event]) -> ReplayCheck:
    ended = [e for e in events if e["type"] == "match_ended"]
    if not ended:
        return ReplayCheck(False, ["no match_ended event"], replay(events))
    record = ended[-1]
    game = replay(events)
    problems = []
    if game.board != record["board"]:
        problems.append("final board differs")
    if game.winner != record["winner"]:
        problems.append(f"winner differs: replay={game.winner} record={record['winner']}")
    if game.status != record["status"]:
        problems.append(f"status differs: replay={game.status} record={record['status']}")
    if game.end_reason != record["reason"]:
        problems.append(f"reason differs: replay={game.end_reason} record={record['reason']}")
    return ReplayCheck(not problems, problems, game)
