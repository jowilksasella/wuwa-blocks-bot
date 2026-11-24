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


def _load(name):
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tests", name)
    if not os.path.exists(path):
        return None
    return cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_COLOR)


def test_tile_crops_bomb_crystal_medal():
    expect = {
        "tile_bomb_big.png": (False, vision.MARK_BOMB_BIG),
        "tile_bomb.png": (False, vision.MARK_BOMB),
        "tile_bomb_teal.png": (False, vision.MARK_BOMB),  # tutorial scale (~50px tile)
        "tile_crystal_snow.png": (True, vision.MARK_NONE),
        "tile_crystal_emblem.png": (True, vision.MARK_NONE),
        "tile_medal_snow.png": (False, vision.MARK_MEDAL),
        "tile_medal_emblem.png": (False, vision.MARK_MEDAL),
        "tile_plain.png": (False, vision.MARK_NONE),
    }
    for name, want in expect.items():
        img = _load(name)
        if img is None:
            continue
        h, w = img.shape[:2]
        if name != "tile_bomb_teal.png":
            img = img[1 : h - 1, 1 : w - 1]
        assert vision.classify_tile(img) == want, name
        # Any element colour: bombs / crystals / medals can sit on tiles of every hue
        # (incl. wind, not captured yet). Rotate hue in 15-degree steps; result must not change.
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        for d in range(15, 180, 15):
            shifted = hsv.copy()
            shifted[..., 0] = (shifted[..., 0].astype(np.int32) + d) % 180
            got = vision.classify_tile(cv2.cvtColor(shifted, cv2.COLOR_HSV2BGR))
            assert got == want, (name, d * 2, got)


def test_large_line_icon_is_medal_not_bomb():
    # Synthetic green tile with a thick hollow star/ring icon covering >30% of the centre,
    # standing in for an unseen medal (e.g. wind). It has no solid round body -> medal.
    tile = np.full((74, 74, 3), (90, 170, 40), dtype=np.uint8)
    for ang in range(0, 360, 45):  # 8 thick petals around an open centre, inside the icon area
        a = np.deg2rad(ang)
        p0 = (int(37 + 8 * np.cos(a)), int(38 + 8 * np.sin(a)))
        p1 = (int(37 + 19 * np.cos(a)), int(38 + 19 * np.sin(a)))
        cv2.line(tile, p0, p1, (245, 250, 245), 6)
    f = vision.tile_features(tile)
    assert f["icon"] > 0.30, f
    assert vision.classify_tile(tile) == (False, vision.MARK_MEDAL), f


def test_saved_medal_mode_with_bombs_and_crystals():
    frame = _load("live_medal_bombs.png")
    if frame is None:
        return
    grid, marks, pieces = vision.read_state_full(frame)
    crystals = {(1, 3), (2, 2), (3, 1), (3, 4), (4, 3), (4, 6), (5, 5), (6, 4)}
    bombs = {(2, 3), (3, 2), (4, 5), (5, 4)}
    plain = {(3, 3), (4, 4)}
    for r in range(vision.N):
        for c in range(vision.N):
            if (r, c) in crystals:
                assert grid[r, c] == vision.FAULT, (r, c)
            elif (r, c) in bombs:
                assert grid[r, c] == vision.NORMAL and marks[r, c] == vision.MARK_BOMB, (r, c)
            elif (r, c) in plain:
                assert grid[r, c] == vision.NORMAL and marks[r, c] == vision.MARK_NONE, (r, c)
            else:
                assert grid[r, c] == vision.EMPTY, (r, c)
    assert [p.specials for p in pieces] == [((1, 0),), ((2, 0),), ((1, 2),)]
    plan = solver.solve_batch(grid, pieces, allow_rotate=True, marks=marks)
    assert plan is not None and plan.complete_batch


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("PASS:", name)
    print("ALL VISION TESTS PASSED")
