"""Turn prompt shared by all LLM agents, so both sides receive identical information."""

from __future__ import annotations

import json

from arena.agents.base import Observation
from arena.engine import BLACK, SIZE

RULES = f"""You are playing Gomoku (five in a row) on a {SIZE}x{SIZE} board.
- Black moves first; players alternate, one stone per turn.
- A stone may only be placed on an empty cell.
- Five or more of your stones in a row (horizontal, vertical or diagonal) wins.
- No renju restrictions. A full board with no winner is a draw.
- Rows and columns are numbered 0 to {SIZE - 1}."""


def build_prompt(obs: Observation) -> str:
    me = "Black (X)" if obs.color == BLACK else "White (O)"
    parts = [
        RULES,
        f"You play {me}. This is turn {obs.turn}.",
        "Board (X = black, O = white, . = empty; row index on the left, column index on top):",
        obs.board_text,
        "Board as JSON rows (0 = empty, 1 = black, 2 = white):",
        json.dumps(obs.board),
    ]
    if obs.owner_notes:
        parts.append(f"Strategy notes from your owner:\n{obs.owner_notes}")
    if obs.last_error:
        parts.append(f"Your previous answer this turn was rejected ({obs.last_error}). Choose a different, legal move.")
    parts.append(
        "Decide your move yourself. Do not run commands or read files; just think and answer.\n"
        'Reply with ONLY a JSON object of the form {"row": <int>, "col": <int>} and nothing else.'
    )
    return "\n\n".join(parts)
