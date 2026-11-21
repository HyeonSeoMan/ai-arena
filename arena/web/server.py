"""Local spectator server. Binds to 127.0.0.1 only; never exposed to the network.

Live events go to browsers over a WebSocket; finished matches are served from
their JSONL logs for replay. Spectators are read-only except for starting a
rematch.
"""

from __future__ import annotations

import asyncio
import re
from pathlib import Path

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse

from arena.agents.base import Agent
from arena.events.log import Event, EventLog, load_events, new_match_id
from arena.runner import MatchConfig, run_match

HOST = "127.0.0.1"
STATIC = Path(__file__).parent / "static"
_MATCH_ID = re.compile(r"^[\w-]+$")


class Arena:
    """Owns the two agents and runs one match at a time, alternating colours on rematch."""

    def __init__(self, first: Agent, second: Agent, config: MatchConfig, matches_dir: Path):
        self.agents = (first, second)
        self.config = config
        self.matches_dir = matches_dir
        self.games_started = 0
        self.log: EventLog | None = None
        self.task: asyncio.Task | None = None
        self.clients: set[asyncio.Queue[Event]] = set()

    @property
    def running(self) -> bool:
        return self.task is not None and not self.task.done()

    def start_match(self) -> str:
        if self.running:
            raise RuntimeError("a match is already running")
        first, second = self.agents
        black, white = (first, second) if self.games_started % 2 == 0 else (second, first)
        self.games_started += 1
        match_id = new_match_id()
        self.log = EventLog(match_id, self.matches_dir / f"{match_id}.jsonl")
        self.log.subscribe(self._broadcast)
        self.task = asyncio.create_task(self._run(black, white, self.log))
        return match_id

    async def _run(self, black: Agent, white: Agent, log: EventLog) -> None:
        try:
            await run_match(black, white, log, self.config)
        finally:
            log.close()

    def _broadcast(self, event: Event) -> None:
        for queue in self.clients:
            queue.put_nowait(event)


def _summary(path: Path) -> dict:
    events = load_events(path)
    start = next((e for e in events if e["type"] == "match_started"), {})
    end = next((e for e in reversed(events) if e["type"] == "match_ended"), None)
    return {
        "match_id": path.stem,
        "black": start.get("black"),
        "white": start.get("white"),
        "winner": end and end["winner"],
        "reason": end and end["reason"],
        "finished": end is not None,
    }


def create_app(arena: Arena) -> FastAPI:
    app = FastAPI(title="AI Arena")

    @app.get("/")
    async def index() -> FileResponse:
        return FileResponse(STATIC / "index.html")

    @app.get("/api/state")
    async def state() -> dict:
        return {"running": arena.running, "match_id": arena.log and arena.log.match_id}

    @app.get("/api/matches")
    async def list_matches() -> list[dict]:
        paths = sorted(arena.matches_dir.glob("*.jsonl"), reverse=True)
        return [_summary(p) for p in paths]

    @app.get("/api/matches/{match_id}")
    async def get_match(match_id: str) -> list[Event]:
        path = arena.matches_dir / f"{match_id}.jsonl"
        if not _MATCH_ID.match(match_id) or not path.is_file():
            raise HTTPException(404, "match not found")
        return load_events(path)

    @app.post("/api/matches")
    async def rematch() -> dict:
        if arena.running:
            raise HTTPException(409, "a match is already running")
        return {"match_id": arena.start_match()}

    @app.websocket("/ws")
    async def live(ws: WebSocket) -> None:
        await ws.accept()
        queue: asyncio.Queue[Event] = asyncio.Queue()
        # Subscribe and snapshot with no await in between so no event is missed or duplicated.
        arena.clients.add(queue)
        backlog = list(arena.log.events) if arena.log else []
        try:
            for event in backlog:
                await ws.send_json(event)
            while True:
                await ws.send_json(await queue.get())
        except WebSocketDisconnect:
            pass
        finally:
            arena.clients.discard(queue)

    return app
