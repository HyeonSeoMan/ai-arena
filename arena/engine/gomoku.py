"""9x9 Gomoku engine: board state, move validation, win/draw judgement.

Pure logic with no I/O. The engine is the only authority on rules;
agents only propose moves.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

SIZE = 9
WIN_LENGTH = 5

EMPTY = 0
BLACK = 1
WHITE = 2

_DIRECTIONS = ((0, 1), (1, 0), (1, 1), (1, -1))
_SYMBOLS = {EMPTY: ".", BLACK: "X", WHITE: "O"}


class Status(StrEnum):
    ONGOING = "ongoing"
    WON = "won"
    DRAW = "draw"


class EndReason(StrEnum):
    FIVE_IN_ROW = "five_in_row"
    BOARD_FULL = "board_full"
    FORFEIT = "forfeit"
    TIME_LIMIT = "time_limit"


class IllegalMove(Exception):
    """Raised when a proposed move is rejected. `reason` is a stable code."""

    def __init__(self, reason: str, message: str):
        super().__init__(message)
        self.reason = reason


@dataclass(frozen=True)
class Move:
    player: int
    row: int
    col: int


def opponent(player: int) -> int:
    return WHITE if player == BLACK else BLACK


class Game:
    def __init__(self) -> None:
        self.board: list[list[int]] = [[EMPTY] * SIZE for _ in range(SIZE)]
        self.current: int = BLACK
        self.moves: list[Move] = []
        self.status: Status = Status.ONGOING
        self.winner: int | None = None
        self.end_reason: EndReason | None = None
        self.end_detail: str | None = None

    @property
    def turn(self) -> int:
        """1-based number of the turn about to be played."""
        return len(self.moves) + 1

    @property
    def is_over(self) -> bool:
        return self.status != Status.ONGOING

    def play(self, row: object, col: object) -> Move:
        """Validate and apply a move for the current player."""
        if self.is_over:
            raise IllegalMove("game_over", "the game has already ended")
        if not _is_int(row) or not _is_int(col):
            raise IllegalMove("not_integer", f"row and col must be integers, got {row!r}, {col!r}")
        if not (0 <= row < SIZE and 0 <= col < SIZE):
            raise IllegalMove("out_of_bounds", f"({row}, {col}) is outside 0..{SIZE - 1}")
        if self.board[row][col] != EMPTY:
            raise IllegalMove("occupied", f"({row}, {col}) is already occupied")

        move = Move(self.current, row, col)
        self.board[row][col] = move.player
        self.moves.append(move)

        if self._makes_five(row, col):
            self._end(Status.WON, move.player, EndReason.FIVE_IN_ROW)
        elif len(self.moves) == SIZE * SIZE:
            self._end(Status.DRAW, None, EndReason.BOARD_FULL)
        else:
            self.current = opponent(move.player)
        return move

    def forfeit(self, player: int, detail: str) -> None:
        """End the game with `player` losing (timeout, invalid responses, crash)."""
        if self.is_over:
            raise IllegalMove("game_over", "the game has already ended")
        self._end(Status.WON, opponent(player), EndReason.FORFEIT, detail)

    def expire(self) -> None:
        """End the game as a draw because the match time limit was reached."""
        if self.is_over:
            raise IllegalMove("game_over", "the game has already ended")
        self._end(Status.DRAW, None, EndReason.TIME_LIMIT)

    def to_text(self) -> str:
        """Board with row/column indices, for agent prompts and debugging."""
        header = "   " + " ".join(str(c) for c in range(SIZE))
        rows = [f"{r:>2} " + " ".join(_SYMBOLS[v] for v in self.board[r]) for r in range(SIZE)]
        return "\n".join([header, *rows])

    @classmethod
    def from_moves(cls, moves: list[tuple[int, int]]) -> Game:
        """Replay a move list from an empty board. Raises IllegalMove on bad history."""
        game = cls()
        for row, col in moves:
            game.play(row, col)
        return game

    def _makes_five(self, row: int, col: int) -> bool:
        player = self.board[row][col]
        for dr, dc in _DIRECTIONS:
            count = 1 + self._count(row, col, dr, dc, player) + self._count(row, col, -dr, -dc, player)
            if count >= WIN_LENGTH:
                return True
        return False

    def _count(self, row: int, col: int, dr: int, dc: int, player: int) -> int:
        n = 0
        r, c = row + dr, col + dc
        while 0 <= r < SIZE and 0 <= c < SIZE and self.board[r][c] == player:
            n += 1
            r += dr
            c += dc
        return n

    def _end(self, status: Status, winner: int | None, reason: EndReason, detail: str | None = None) -> None:
        self.status = status
        self.winner = winner
        self.end_reason = reason
        self.end_detail = detail


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)
