"""Match event stream.

Every state change becomes one event. The same stream is written to a JSONL
file (for replay) and fanned out to live listeners (for spectating).
"""

from __future__ import annotations

import json
import time
import uuid
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

Event = dict[str, Any]
Listener = Callable[[Event], None]

MAX_RAW_CHARS = 4000


PARTIAL_SUFFIX = ".part"


class EventLog:
    """Writes to `<path>.part` while the match runs; renamed to `path` only once it has ended.

    Aborted or crashed matches therefore stay as `.part` files, which git ignores.
    """

    def __init__(self, match_id: str, path: Path | None = None):
        self.match_id = match_id
        self.path = path
        self.events: list[Event] = []
        self._listeners: list[Listener] = []
        self._file = None
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            self._partial = path.with_name(path.name + PARTIAL_SUFFIX)
            self._file = self._partial.open("w", encoding="utf-8")

    def subscribe(self, listener: Listener) -> None:
        self._listeners.append(listener)

    def emit(self, type_: str, audience: str = "public", **data: Any) -> Event:
        event = {
            "match_id": self.match_id,
            "seq": len(self.events),
            "ts": time.time(),
            "type": type_,
            "audience": audience,
            **data,
        }
        self.events.append(event)
        if self._file is not None:
            self._file.write(json.dumps(event, ensure_ascii=False) + "\n")
            self._file.flush()
        for listener in self._listeners:
            listener(event)
        return event

    def broadcast(self, type_: str, **data: Any) -> None:
        """Send a transient event (e.g. an animation frame) to live listeners only; not logged."""
        event = {"match_id": self.match_id, "type": type_, "audience": "public", **data}
        for listener in self._listeners:
            listener(event)

    @property
    def finished(self) -> bool:
        return bool(self.events) and self.events[-1]["type"] == "match_ended"

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None
            if self.finished:
                self._partial.replace(self.path)


def new_match_id() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]


def truncate(text: str, limit: int = MAX_RAW_CHARS) -> str:
    return text if len(text) <= limit else text[:limit] + f"... [truncated {len(text) - limit} chars]"


def load_events(path: Path) -> list[Event]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]
