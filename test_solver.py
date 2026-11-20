"""Unit tests for solver mechanics: rotation, multi-line clear, corrupted & fault blocks, anti-deadlock."""
import numpy as np

import solver
from vision import N, EMPTY, NORMAL, CORRUPTED, FAULT, TrayPiece


def _piece(cells, slot_idx=0, has_special=False):
    norm, r, c = solver.normalize_cells(cells)
    return TrayPiece(slot_idx=slot_idx, box=(0, 0, 100, 100), center=(50, 50), cells=norm, rows=r, cols=c, has_special=has_special)


def test_rotate_cw_L_piece():
    # Matches tutorial image 1: #. / ## rotated 90 CW -> ## / #.
    cells = ((0, 0), (1, 0), (1, 1))
    r1, rows, cols = solver.rotate_cw(cells, 1)
    assert (rows, cols) == (2, 2)
    assert r1 == ((0, 0), (0, 1), (1, 0))
    # 4 rotations returns original
    r4, _, _ = solver.rotate_cw(cells, 4)
    assert r4 == cells


def test_corrupted_block_survives_row_clear():
    g = np.zeros((N, N), dtype=np.int32)
    g[0, 0] = CORRUPTED
    g[0, 7] = CORRUPTED
    for c in range(1, 6):
        g[0, c] = NORMAL
    # Fill remaining cell (0, 6) -> row 0 clears, but (0,0) and (0,7) stay CORRUPTED!
    ng, lines, pts, faults_hit, _ = solver.apply_move(g, ((0, 6),))
    assert lines == 1
    assert pts == 200
    assert faults_hit == 0
    assert ng[0, 0] == CORRUPTED
    assert ng[0, 7] == CORRUPTED
    assert all(ng[0, c] == EMPTY for c in range(1, 7))


def test_fault_block_requires_extra_clear():
    g = np.zeros((N, N), dtype=np.int32)
    for c in range(7):
        g[2, c] = NORMAL
    g[2, 3] = FAULT
    # Place at (2, 7) -> row 2 clears: NORMAL cells vanish, FAULT downgrades to NORMAL (still occupied!)
    ng, lines, pts, faults_hit, _ = solver.apply_move(g, ((2, 7),))
    assert lines == 1
    assert faults_hit == 1
    assert ng[2, 3] == NORMAL
    assert ng[2, 0] == EMPTY
    assert ng[2, 7] == EMPTY


def test_quadruple_clear_live_tutorial_puzzle():
    # Exact board from live_cap.png (tutorial 1)
    g = np.zeros((N, N), dtype=np.int32)
    for r, c in [
        (0, 0), (0, 3), (0, 4), (0, 7),
        (1, 3), (1, 4), (2, 3), (2, 4),
        (3, 0), (3, 1), (3, 2), (3, 5), (3, 6), (3, 7),
        (4, 0), (4, 1), (4, 2), (4, 4), (4, 5), (4, 6), (4, 7),
        (5, 3), (5, 4), (6, 3), (6, 4), (7, 3), (7, 4),
    ]:
        g[r, c] = NORMAL

    p0 = _piece([(0, 0), (1, 0), (1, 1)], slot_idx=0)
    p1 = _piece([(0, 0), (0, 1), (1, 0), (1, 1)], slot_idx=1)

    plan = solver.solve_batch(g, [p0, p1], allow_rotate=True)
    assert plan is not None
    assert plan.complete_batch
    assert plan.total_lines == 4
    assert plan.total_points == 1500
    # Must clear immediately on Move 1 (early clear tie-breaker + full accumulated bonus)
    assert plan.moves[0].slot_idx == 0
    assert plan.moves[0].rot == 1
    assert plan.moves[0].board_cells == ((3, 3), (3, 4), (4, 3))


def test_batch_order_avoids_deadlock():
    # Board full except row 0 cols 0..3 and a 3x3 area at rows 1..3 cols 0..2 blocked by row 0 col 0..2
    g = np.full((N, N), NORMAL, dtype=np.int32)
    # Leave only (0,0) empty so placing 1x1 clears row 0 AND col 0, making room for a 1x4 bar!
    g[0, 0] = EMPTY
    p_bar = _piece([(0, 0), (0, 1), (0, 2), (0, 3)], slot_idx=0)
    p_dot = _piece([(0, 0)], slot_idx=1)

    plan = solver.solve_batch(g, [p_bar, p_dot], allow_rotate=True)
    assert plan is not None
    assert plan.complete_batch
    assert plan.moves[0].slot_idx == 1  # MUST place 1x1 first to clear space for 1x4!
    assert plan.moves[1].slot_idx == 0


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("PASS:", name)
    print("ALL SOLVER TESTS PASSED")
