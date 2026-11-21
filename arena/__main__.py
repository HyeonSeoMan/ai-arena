"""Command-line entry point.

    uv run python -m arena play --black heuristic --white random --games 2
    uv run python -m arena replay matches/<match_id>.jsonl

Only offline dummy agents are available at this stage.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import uuid
from datetime import datetime
from pathlib import Path

from arena.agents import HeuristicAgent, RandomAgent
from arena.engine import BLACK, Game
from arena.events import EventLog, load_events, verify
from arena.runner import MatchConfig, run_match

MATCHES_DIR = Path("matches")
COLOR_NAMES = {1: "black", 2: "white"}

AGENTS = {
    "random": lambda name, seed: RandomAgent(name=name, seed=seed),
    "heuristic": lambda name, seed: HeuristicAgent(name=name, seed=seed),
}


def new_match_id() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]


def describe(game: Game, black: str, white: str) -> str:
    if game.winner is None:
        return f"draw ({game.end_reason})"
    winner = black if game.winner == BLACK else white
    detail = f": {game.end_detail}" if game.end_detail else ""
    return f"{winner} ({COLOR_NAMES[game.winner]}) wins by {game.end_reason}{detail}"


async def play(args: argparse.Namespace) -> int:
    config = MatchConfig(turn_timeout=args.turn_timeout, max_retries=args.max_retries, match_timeout=args.match_timeout)
    first = AGENTS[args.black](f"{args.black}-A", args.seed)
    second = AGENTS[args.white](f"{args.white}-B", None if args.seed is None else args.seed + 1)

    for i in range(args.games):
        # Rematches alternate colours.
        black, white = (first, second) if i % 2 == 0 else (second, first)
        match_id = new_match_id()
        path = MATCHES_DIR / f"{match_id}.jsonl"
        log = EventLog(match_id, path)
        try:
            game = await run_match(black, white, log, config)
        finally:
            log.close()
        print(f"\n=== game {i + 1}/{args.games}: {black.name} (X) vs {white.name} (O) ===")
        print(game.to_text())
        print(describe(game, black.name, white.name))
        print(f"log: {path}")
    return 0


def replay_cmd(args: argparse.Namespace) -> int:
    events = load_events(Path(args.log))
    check = verify(events)
    print(check.game.to_text())
    if check.ok:
        print(f"replay OK: {len(check.game.moves)} moves, status={check.game.status}, winner={check.game.winner}")
        return 0
    print("replay MISMATCH: " + "; ".join(check.problems))
    return 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="arena")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("play", help="run match(es) between agents")
    p.add_argument("--black", choices=AGENTS, default="heuristic")
    p.add_argument("--white", choices=AGENTS, default="random")
    p.add_argument("--games", type=int, default=1)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--turn-timeout", type=float, default=MatchConfig.turn_timeout)
    p.add_argument("--max-retries", type=int, default=MatchConfig.max_retries)
    p.add_argument("--match-timeout", type=float, default=MatchConfig.match_timeout)

    r = sub.add_parser("replay", help="verify a match log by replaying it")
    r.add_argument("log")

    args = parser.parse_args(argv)
    if args.command == "play":
        return asyncio.run(play(args))
    return replay_cmd(args)


if __name__ == "__main__":
    sys.exit(main())
