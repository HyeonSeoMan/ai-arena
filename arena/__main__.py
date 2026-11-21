"""Command-line entry point.

    uv run python -m arena serve --black claude --white codex     # watch live at http://127.0.0.1:8000
    uv run python -m arena play --black heuristic --white random --games 2
    uv run python -m arena replay matches/<match_id>.jsonl
    uv run python -m arena pong --left claude --right codex     # coach pong, watch live
    uv run python -m arena pong --left random --right fixed --fast   # offline, terminal only
    uv run python -m arena tanks --first claude --second codex   # coach tanks, watch live

`claude` and `codex` run the subscription-authenticated CLIs and require the
billing preflight to pass; the other agents are offline dummies.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import webbrowser
from pathlib import Path

from arena.agents import HeuristicAgent, RandomAgent
from arena.agents.billing import BillingGuardError, preflight
from arena.agents.cli import ClaudeCodeAgent, CodexAgent
from arena.engine import BLACK, Game
from arena.events import EventLog, load_events, verify
from arena.events.log import new_match_id
from arena.pong.coach import FixedCoach, LLMCoach, RandomCoach
from arena.pong.match import PongConfig, run_pong
from arena.pong.replay import verify_pong
from arena.runner import MatchConfig, run_match
from arena.tanks import coach as tanks_coach
from arena.tanks.match import TanksConfig, run_tanks
from arena.tanks.replay import verify_tanks

MATCHES_DIR = Path("matches")
PONG_DIR = MATCHES_DIR / "pong"
TANKS_DIR = MATCHES_DIR / "tanks"
COLOR_NAMES = {1: "black", 2: "white"}

AGENTS = {
    "random": lambda name, args: RandomAgent(name=name, seed=args.seed),
    "heuristic": lambda name, args: HeuristicAgent(name=name, seed=args.seed, delay=args.dummy_delay),
    "claude": lambda name, args: ClaudeCodeAgent(name=name),
    "codex": lambda name, args: CodexAgent(name=name),
}


def make_agents(args: argparse.Namespace):
    if args.black == args.white:
        names = (f"{args.black}-A", f"{args.white}-B")
    else:
        names = (args.black, args.white)
    preflight({args.black, args.white})  # no-op for dummy agents
    return AGENTS[args.black](names[0], args), AGENTS[args.white](names[1], args)


def make_config(args: argparse.Namespace) -> MatchConfig:
    return MatchConfig(turn_timeout=args.turn_timeout, max_retries=args.max_retries, match_timeout=args.match_timeout)


def describe(game: Game, black: str, white: str) -> str:
    if game.winner is None:
        return f"draw ({game.end_reason})"
    winner = black if game.winner == BLACK else white
    detail = f": {game.end_detail}" if game.end_detail else ""
    return f"{winner} ({COLOR_NAMES[game.winner]}) wins by {game.end_reason}{detail}"


async def play(args: argparse.Namespace) -> int:
    first, second = make_agents(args)
    config = make_config(args)
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


async def serve(args: argparse.Namespace) -> int:
    from arena.web.server import Arena

    first, second = make_agents(args)
    config = make_config(args)
    arena = Arena(first, second, lambda a, b, log: run_match(a, b, log, config), MATCHES_DIR)
    await run_server(arena, args)
    return 0


async def run_server(arena, args: argparse.Namespace) -> None:
    import uvicorn

    from arena.web.server import HOST, create_app

    server = uvicorn.Server(uvicorn.Config(create_app(arena), host=HOST, port=args.port, log_level="warning"))
    url = f"http://{HOST}:{args.port}"

    async def start_when_ready() -> None:
        while not server.started:
            await asyncio.sleep(0.05)
        print(f"spectate at {url}  (Ctrl+C to stop)", flush=True)
        if not args.no_browser:
            webbrowser.open(url)
        await asyncio.sleep(args.start_delay)  # give the browser a moment to connect
        arena.start_match()

    starter = asyncio.create_task(start_when_ready())
    await server.serve()
    starter.cancel()


PONG_COACHES = {
    "random": lambda name, args: RandomCoach(name=name, seed=args.seed, delay=args.dummy_delay),
    "fixed": lambda name, args: FixedCoach(name=name, delay=args.dummy_delay),
    "claude": lambda name, args: LLMCoach(ClaudeCodeAgent(name=name)),
    "codex": lambda name, args: LLMCoach(CodexAgent(name=name)),
}


async def pong(args: argparse.Namespace) -> int:
    names = (f"{args.left}-A", f"{args.right}-B") if args.left == args.right else (args.left, args.right)
    preflight({args.left, args.right})
    first = PONG_COACHES[args.left](names[0], args)
    second = PONG_COACHES[args.right](names[1], args)
    config = PongConfig(points_to_win=args.points, timeout_seconds=args.timeout, realtime=not args.fast)

    if args.fast:
        match_id = new_match_id()
        log = EventLog(match_id, PONG_DIR / f"{match_id}.jsonl")
        try:
            winner, score = await run_pong(first, second, log, config)
        finally:
            log.close()
        print(f"{first.name} {score[0]} : {score[1]} {second.name}  -> {(first, second)[winner].name} wins")
        print(f"log: {log.path}")
        return 0

    from arena.web.server import Arena

    arena = Arena(first, second, lambda a, b, log: run_pong(a, b, log, config), PONG_DIR, page="pong.html")
    await run_server(arena, args)
    return 0


TANK_COACHES = {
    "random": lambda name, args: tanks_coach.RandomCoach(name=name, seed=args.seed, delay=args.dummy_delay),
    "fixed": lambda name, args: tanks_coach.FixedCoach(name=name, delay=args.dummy_delay),
    "claude": lambda name, args: tanks_coach.LLMCoach(ClaudeCodeAgent(name=name)),
    "codex": lambda name, args: tanks_coach.LLMCoach(CodexAgent(name=name)),
}


async def tanks(args: argparse.Namespace) -> int:
    same = args.first == args.second
    names = (f"{args.first}-A", f"{args.second}-B") if same else (args.first, args.second)
    preflight({args.first, args.second})
    first = TANK_COACHES[args.first](names[0], args)
    second = TANK_COACHES[args.second](names[1], args)
    config = TanksConfig(timeout_seconds=args.timeout, realtime=not args.fast)

    if args.fast:
        match_id = new_match_id()
        log = EventLog(match_id, TANKS_DIR / f"{match_id}.jsonl")
        try:
            winner, reason = await run_tanks(first, second, log, config)
        finally:
            log.close()
        end = log.events[-1]
        print(f"{first.name} vs {second.name}: {(first, second)[winner].name} wins by {reason} "
              f"(kills {end['kills']}, scores {end['scores']}, levels {end['levels']}, timeouts {end['timeouts']})")
        print(f"log: {log.path}")
        return 0

    from arena.web.server import Arena

    arena = Arena(first, second, lambda a, b, log: run_tanks(a, b, log, config), TANKS_DIR, page="tanks.html")
    await run_server(arena, args)
    return 0


def replay_cmd(args: argparse.Namespace) -> int:
    events = load_events(Path(args.log))
    if events and events[0].get("game") == "tanks":
        check = verify_tanks(events)
        end = events[-1]
        print(f"replay {'OK' if check.ok else 'MISMATCH'}: kills={end.get('kills')} scores={end.get('scores')}")
        for problem in check.problems:
            print(f"  {problem}")
        return 0 if check.ok else 1
    if events and events[0].get("game") == "pong":
        check = verify_pong(events)
        end = events[-1]
        print(f"replay {'OK' if check.ok else 'MISMATCH'}: score={end.get('score')}")
        for problem in check.problems:
            print(f"  {problem}")
        return 0 if check.ok else 1
    check = verify(events)
    print(check.game.to_text())
    if check.ok:
        print(f"replay OK: {len(check.game.moves)} moves, status={check.game.status}, winner={check.game.winner}")
        return 0
    print("replay MISMATCH: " + "; ".join(check.problems))
    return 1


def add_match_options(p: argparse.ArgumentParser) -> None:
    p.add_argument("--black", choices=AGENTS, default="heuristic")
    p.add_argument("--white", choices=AGENTS, default="random")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--dummy-delay", type=float, default=0.0, help="thinking time for heuristic agents (seconds)")
    p.add_argument("--turn-timeout", type=float, default=MatchConfig.turn_timeout)
    p.add_argument("--max-retries", type=int, default=MatchConfig.max_retries)
    p.add_argument("--match-timeout", type=float, default=MatchConfig.match_timeout)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="arena")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("play", help="run match(es) in the terminal")
    add_match_options(p)
    p.add_argument("--games", type=int, default=1)

    s = sub.add_parser("serve", help="run a match and spectate it in the browser (127.0.0.1 only)")
    add_match_options(s)
    s.add_argument("--port", type=int, default=8000)
    s.add_argument("--no-browser", action="store_true")
    s.add_argument("--start-delay", type=float, default=2.0)

    g = sub.add_parser("pong", help="coach pong: AI coaches set strategies, bots play in real time")
    g.add_argument("--left", choices=PONG_COACHES, default="random")
    g.add_argument("--right", choices=PONG_COACHES, default="fixed")
    g.add_argument("--points", type=int, default=PongConfig.points_to_win)
    g.add_argument("--timeout", type=float, default=PongConfig.timeout_seconds, help="strategy timeout (seconds)")
    g.add_argument("--seed", type=int, default=None)
    g.add_argument("--dummy-delay", type=float, default=0.0, help="thinking time for dummy coaches (seconds)")
    g.add_argument("--fast", action="store_true", help="simulate without real-time pacing or browser")
    g.add_argument("--port", type=int, default=8000)
    g.add_argument("--no-browser", action="store_true")
    g.add_argument("--start-delay", type=float, default=2.0)

    t = sub.add_parser("tanks", help="coach tanks: AI coaches set builds and plans, bots fight in real time")
    t.add_argument("--first", choices=TANK_COACHES, default="random")
    t.add_argument("--second", choices=TANK_COACHES, default="fixed")
    t.add_argument("--timeout", type=float, default=TanksConfig.timeout_seconds, help="strategy timeout (seconds)")
    t.add_argument("--seed", type=int, default=None)
    t.add_argument("--dummy-delay", type=float, default=0.0, help="thinking time for dummy coaches (seconds)")
    t.add_argument("--fast", action="store_true", help="simulate without real-time pacing or browser")
    t.add_argument("--port", type=int, default=8000)
    t.add_argument("--no-browser", action="store_true")
    t.add_argument("--start-delay", type=float, default=2.0)

    r = sub.add_parser("replay", help="verify a match log by replaying it")
    r.add_argument("log")

    args = parser.parse_args(argv)
    try:
        if args.command == "play":
            return asyncio.run(play(args))
        if args.command == "serve":
            return asyncio.run(serve(args))
        if args.command == "pong":
            return asyncio.run(pong(args))
        if args.command == "tanks":
            return asyncio.run(tanks(args))
    except BillingGuardError as exc:
        print(f"[billing] {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        return 130
    return replay_cmd(args)


if __name__ == "__main__":
    sys.exit(main())
