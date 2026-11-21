"""Turn loop: asks agents for moves, enforces time limits and retries, emits events."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass

from arena.agents.base import Agent, AgentReply, Observation
from arena.engine import BLACK, SIZE, WHITE, Game, IllegalMove, opponent
from arena.events.log import EventLog, truncate


@dataclass(frozen=True)
class MatchConfig:
    turn_timeout: float = 60.0
    max_retries: int = 2
    match_timeout: float = 3600.0


async def run_match(
    black: Agent,
    white: Agent,
    log: EventLog,
    config: MatchConfig | None = None,
    clock: Callable[[], float] = time.monotonic,
) -> Game:
    config = config or MatchConfig()
    players = {BLACK: black, WHITE: white}
    game = Game()
    start = clock()

    log.emit("match_started", black=black.name, white=white.name, board_size=SIZE, config=asdict(config))

    while not game.is_over:
        player = game.current
        agent = players[player]
        turn = game.turn
        log.emit("turn_started", turn=turn, player=player, agent=agent.name)

        last_error: str | None = None
        for attempt in range(1, config.max_retries + 2):
            remaining = config.match_timeout - (clock() - start)
            if remaining <= 0:
                game.expire()
                break

            obs = Observation(
                match_id=log.match_id,
                turn=turn,
                color=player,
                board=[row[:] for row in game.board],
                board_text=game.to_text(),
                moves=[(m.player, m.row, m.col) for m in game.moves],
                attempt=attempt,
                last_error=last_error,
            )
            limit = min(config.turn_timeout, remaining)
            t0 = clock()
            try:
                reply = await asyncio.wait_for(agent.act(obs), timeout=limit)
            except TimeoutError:
                elapsed_ms = round((clock() - t0) * 1000)
                log.emit("agent_timeout", turn=turn, player=player, attempt=attempt, elapsed_ms=elapsed_ms)
                if remaining <= config.turn_timeout:
                    game.expire()
                else:
                    game.forfeit(player, f"no response within {config.turn_timeout:g}s")
                break
            except Exception as exc:  # an agent crash counts as an invalid attempt
                reply = AgentReply(raw="", error=f"agent error: {type(exc).__name__}: {exc}")
            elapsed_ms = round((clock() - t0) * 1000)

            log.emit(
                "agent_responded",
                turn=turn,
                player=player,
                attempt=attempt,
                elapsed_ms=elapsed_ms,
                raw=truncate(reply.raw),
                row=reply.row,
                col=reply.col,
                error=reply.error,
                meta=reply.meta,
            )

            if reply.error:
                reason, message = "invalid_response", reply.error
            else:
                try:
                    game.play(reply.row, reply.col)
                except IllegalMove as exc:
                    reason, message = exc.reason, str(exc)
                else:
                    log.emit("move_applied", turn=turn, player=player, row=reply.row, col=reply.col)
                    break

            log.emit("move_rejected", turn=turn, player=player, attempt=attempt, reason=reason, message=message)
            last_error = f"{reason}: {message}"
        else:
            game.forfeit(player, f"{config.max_retries + 1} invalid attempts; last: {last_error}")

    log.emit(
        "match_ended",
        status=game.status,
        winner=game.winner,
        loser=opponent(game.winner) if game.winner else None,
        reason=game.end_reason,
        detail=game.end_detail,
        board=game.board,
        total_moves=len(game.moves),
        duration_ms=round((clock() - start) * 1000),
    )
    return game
