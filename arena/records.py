"""Build matches/README.md: an index of every finished match log, newest first."""

from __future__ import annotations

from pathlib import Path

from arena.events.log import load_events

LLM_NAMES = ("claude", "codex")
GAME_TITLES = {"gomoku": "오목", "pong": "Coach Pong", "tanks": "Coach Tanks"}


def _players(start: dict) -> tuple[str, str]:
    return (start.get("black") or start.get("left") or "?", start.get("white") or start.get("right") or "?")


def _result(game: str, players: tuple[str, str], end: dict) -> str:
    if end.get("winner") is None:
        return "무승부"
    winner = players[end["winner"] - 1] if game == "gomoku" else players[end["winner"]]
    if game == "gomoku":
        return f"{winner} 승 ({end['reason']}, {end['total_moves']}수)"
    if game == "pong":
        return f"{winner} 승 ({end['score'][0]} : {end['score'][1]})"
    detail = f"{end['kills'][0]} : {end['kills'][1]} 킬" if end["reason"] == "kills" else \
        f"점수 {end['scores'][0]:,} : {end['scores'][1]:,}"
    return f"{winner} 승 ({detail})"


def summarize(path: Path, root: Path) -> dict | None:
    events = load_events(path)
    if not events or events[-1]["type"] != "match_ended":
        return None
    start, end = events[0], events[-1]
    game = start.get("game", "gomoku")
    players = _players(start)
    return {
        "id": path.stem,
        "path": path.relative_to(root).as_posix(),
        "game": game,
        "players": players,
        "ai": any(name.split("-")[0] in LLM_NAMES for name in players),
        "result": _result(game, players, end),
    }


def build_index(matches_dir: Path) -> tuple[str, list[dict]]:
    rows = [r for p in sorted(matches_dir.rglob("*.jsonl"), reverse=True) if (r := summarize(p, matches_dir))]
    rows.sort(key=lambda r: r["id"], reverse=True)
    lines = [
        "# 경기 기록",
        "",
        "완료된 경기 로그(JSONL) 목록입니다. `uv run python -m arena records`로 다시 만듭니다.",
        "각 로그는 `uv run python -m arena replay <파일>`로 재현 검증할 수 있습니다.",
        "",
        "| 경기 ID | 게임 | 참가자 | 실제 AI | 결과 | 로그 |",
        "|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(f"| {r['id']} | {GAME_TITLES.get(r['game'], r['game'])} | {r['players'][0]} vs {r['players'][1]} | "
                     f"{'예' if r['ai'] else '더미'} | {r['result']} | [{r['path']}]({r['path']}) |")
    return "\n".join(lines) + "\n", rows
