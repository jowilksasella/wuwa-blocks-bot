"""Offline vision tests using synthetic frames and saved real game captures (if present)."""
import os
import cv2
import numpy as np

import vision
import solver


def _synthetic_slot(cells, slot_size=114, cell_px=24):
    """Render a crisp dark tray slot with a colored piece centered inside."""
    img = np.full((slot_size, slot_size, 3), (41, 18, 22), dtype=np.uint8)
    rows = max(r for r, _ in cells) + 1
    cols = max(c for _, c in cells) + 1
    pw = cols * cell_px
    ph = rows * cell_px
    ox = (slot_size - pw) // 2
    oy = (slot_size - ph) // 2
    for r, c in cells:
        x0 = ox + c * cell_px
        y0 = oy + r * cell_px
        cv2.rectangle(img, (x0, y0), (x0 + cell_px - 2, y0 + cell_px - 2), (80, 160, 240), -1)
    return img


def test_decode_various_polyominoes():
    cases = [
        ("1x1", [(0, 0)], 24),
        ("2x2", [(0, 0), (0, 1), (1, 0), (1, 1)], 24),
        ("3x3", [(r, c) for r in range(3) for c in range(3)], 24),
        ("L3", [(0, 0), (1, 0), (1, 1)], 24),
        ("T5", [(0, 0), (0, 1), (0, 2), (1, 1), (2, 1)], 24),
        ("bar4_h", [(0, 0), (0, 1), (0, 2), (0, 3)], 24),
        ("bar5_h", [(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)], 21),
        ("bar5_v", [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0)], 21),
        ("Z4", [(0, 0), (0, 1), (1, 1), (1, 2)], 24),
        ("diag2", [(0, 0), (1, 1)], 24),
    ]
    for name, cells, cpx in cases:
        crop = _synthetic_slot(cells, slot_size=114, cell_px=cpx)
        p = vision.decode_slot_piece(crop, 0, (1059, 253, 1173, 366))
        assert p is not None, f"{name} returned None"
        assert p.cells == tuple(sorted(cells)), f"{name}: got {p.cells}, expected {tuple(sorted(cells))}"


def test_empty_slot_returns_none():
    crop = np.full((114, 114, 3), (41, 18, 22), dtype=np.uint8)
    assert vision.decode_slot_piece(crop, 0, (1059, 253, 1173, 366)) is None


def test_saved_live_capture_if_available():
    sample = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tests", "live_sample.png")
    if not os.path.exists(sample):
        return
    frame = cv2.imdecode(np.fromfile(sample, np.uint8), cv2.IMREAD_COLOR)
    assert vision.in_level(frame)
    grid, pieces = vision.read_state(frame)
    assert int((grid == vision.NORMAL).sum()) == 27
    assert len(pieces) == 2
    assert pieces[0].cells == ((0, 0), (1, 0), (1, 1))
    assert pieces[1].cells == ((0, 0), (0, 1), (1, 0), (1, 1))
    plan = solver.solve_batch(grid, pieces, allow_rotate=True)
    assert plan is not None and plan.total_lines == 4


def test_saved_lvl2_3slot_with_corrupted_and_5x1():
    sample = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tests", "live_lvl2_3slot.png")
    if not os.path.exists(sample):
        return
    frame = cv2.imdecode(np.fromfile(sample, np.uint8), cv2.IMREAD_COLOR)
    assert vision.in_level(frame)
    grid, pieces = vision.read_state(frame)
    # Four corner corrupted blocks
    for r, c in ((0, 0), (0, 7), (7, 0), (7, 7)):
        assert grid[r, c] == vision.CORRUPTED
    assert len(pieces) == 3
    assert pieces[0].cells == ((0, 0), (0, 1), (1, 0), (2, 0))
    assert pieces[1].cells == tuple((r, c) for r in range(3) for c in range(3))
    assert pieces[2].cells == ((0, 0), (0, 1), (0, 2), (0, 3), (0, 4))


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("PASS:", name)
    print("ALL VISION TESTS PASSED")
