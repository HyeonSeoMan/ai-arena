import pytest

from arena.engine import BLACK, EMPTY, SIZE, WHITE, EndReason, Game, IllegalMove, Status


def play_black_line(cells, white_cells=None):
    """Black plays `cells` in order; white plays harmless filler moves in between."""
    # Spaced out so white never connects five on its own.
    filler = iter(white_cells or [(r, c) for r in (8, 6) for c in range(0, SIZE, 2)])
    game = Game()
    for i, (r, c) in enumerate(cells):
        game.play(r, c)
        if i < len(cells) - 1:
            game.play(*next(filler))
    return game


def test_initial_state():
    game = Game()
    assert all(v == EMPTY for row in game.board for v in row)
    assert game.current == BLACK
    assert game.turn == 1
    assert game.status == Status.ONGOING


def test_players_alternate_starting_with_black():
    game = Game()
    assert game.play(0, 0).player == BLACK
    assert game.play(0, 1).player == WHITE
    assert game.current == BLACK
    assert game.turn == 3


@pytest.mark.parametrize(
    "cells",
    [
        [(4, c) for c in range(2, 7)],  # horizontal
        [(r, 4) for r in range(0, 5)],  # vertical
        [(i, i) for i in range(1, 6)],  # diagonal
        [(i, 6 - i) for i in range(0, 5)],  # anti-diagonal
    ],
    ids=["horizontal", "vertical", "diagonal", "anti-diagonal"],
)
def test_five_in_row_wins_in_all_directions(cells):
    game = play_black_line(cells)
    assert game.status == Status.WON
    assert game.winner == BLACK
    assert game.end_reason == EndReason.FIVE_IN_ROW


def test_win_detected_when_gap_is_filled_last():
    game = play_black_line([(4, 0), (4, 1), (4, 3), (4, 4), (4, 2)])
    assert game.winner == BLACK


def test_overline_of_six_wins():
    game = play_black_line([(2, 0), (2, 1), (2, 2), (2, 4), (2, 5), (2, 3)])
    assert game.status == Status.WON
    assert game.winner == BLACK


def test_four_in_row_does_not_win():
    game = play_black_line([(4, c) for c in range(4)])
    assert game.status == Status.ONGOING


def test_white_can_win():
    game = Game()
    black = [(8, 0), (8, 2), (8, 4), (8, 6), (7, 1)]
    white = [(0, c) for c in range(5)]
    for b, w in zip(black, white):
        game.play(*b)
        game.play(*w)
    assert game.winner == WHITE


def test_occupied_cell_rejected_without_changing_state():
    game = Game()
    game.play(4, 4)
    with pytest.raises(IllegalMove) as exc:
        game.play(4, 4)
    assert exc.value.reason == "occupied"
    assert game.current == WHITE
    assert len(game.moves) == 1


@pytest.mark.parametrize("row,col", [(-1, 0), (0, -1), (9, 0), (0, 9)])
def test_out_of_bounds_rejected(row, col):
    with pytest.raises(IllegalMove) as exc:
        Game().play(row, col)
    assert exc.value.reason == "out_of_bounds"


@pytest.mark.parametrize("row,col", [("4", 4), (4.0, 4), (True, 4), (None, 4)])
def test_non_integer_rejected(row, col):
    with pytest.raises(IllegalMove) as exc:
        Game().play(row, col)
    assert exc.value.reason == "not_integer"


def test_move_after_game_over_rejected():
    game = play_black_line([(0, c) for c in range(5)])
    with pytest.raises(IllegalMove) as exc:
        game.play(5, 5)
    assert exc.value.reason == "game_over"


# 41 black / 40 white with no five-in-row in any direction.
DRAW_PATTERN = [[BLACK if ((c // 2) + r) % 2 == 0 else WHITE for c in range(SIZE)] for r in range(SIZE)]


def draw_move_order():
    blacks = [(r, c) for r in range(SIZE) for c in range(SIZE) if DRAW_PATTERN[r][c] == BLACK]
    whites = [(r, c) for r in range(SIZE) for c in range(SIZE) if DRAW_PATTERN[r][c] == WHITE]
    order = []
    for i, b in enumerate(blacks):
        order.append(b)
        if i < len(whites):
            order.append(whites[i])
    return order


def test_full_board_without_five_is_draw():
    game = Game.from_moves(draw_move_order())
    assert game.status == Status.DRAW
    assert game.winner is None
    assert game.end_reason == EndReason.BOARD_FULL
    assert game.board == DRAW_PATTERN


def test_forfeit_awards_win_to_opponent():
    game = Game()
    game.play(4, 4)
    game.forfeit(WHITE, "timeout")
    assert game.winner == BLACK
    assert game.end_reason == EndReason.FORFEIT
    assert game.end_detail == "timeout"
    with pytest.raises(IllegalMove):
        game.play(0, 0)


def test_expire_ends_in_draw():
    game = Game()
    game.expire()
    assert game.status == Status.DRAW
    assert game.end_reason == EndReason.TIME_LIMIT


def test_replay_reproduces_final_state():
    original = play_black_line([(i, i) for i in range(5)])
    replayed = Game.from_moves([(m.row, m.col) for m in original.moves])
    assert replayed.board == original.board
    assert replayed.winner == original.winner
    assert replayed.end_reason == original.end_reason


def test_replay_rejects_illegal_history():
    with pytest.raises(IllegalMove):
        Game.from_moves([(0, 0), (0, 0)])


def test_to_text_has_indices_and_symbols():
    game = Game()
    game.play(0, 0)
    game.play(8, 8)
    lines = game.to_text().splitlines()
    assert lines[0] == "   0 1 2 3 4 5 6 7 8"
    assert lines[1].startswith(" 0 X")
    assert lines[9].endswith("O")
