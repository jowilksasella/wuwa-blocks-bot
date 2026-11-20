"""Computer-vision reader for Wuthering Waves「拉海洛方块 · 逆转」(8x8 board + dynamic 2/3-slot tray).

All base coordinates are measured on a 1280x720 client area and scaled via
layout(w, h) (following btw-helper's resolution-independent convention).
"""
from dataclasses import dataclass
import cv2
import numpy as np

N = 8
BASE_W, BASE_H = 1280.0, 720.0

# Exact 8x8 grid geometry measured on 1280x720 client area:
# cell(0,0) occupies x=[337,410], y=[57,130]; pitch = 76.0px (74px tile + 2px gap).
BX0, BY0, PITCH = 336.0, 56.0, 76.0

# Right-hand tray dark box horizontal bounds on 1280x720: x=[1059, 1173] (width 115)
TRAY_X0, TRAY_X1 = 1059.0, 1173.0
TRAY_Y_MIN, TRAY_Y_MAX = 60.0, 660.0

# Cell types on the 8x8 board
EMPTY = 0
NORMAL = 1
CORRUPTED = 2  # 灰色「数据损坏方块」：不可消除，但同行/列满时其他方块照常消除
FAULT = 3      # 1级/2级「故障方块」：高光裂纹/冰晶表面，消除时先解除故障层

CELL_NAMES = {EMPTY: ".", NORMAL: "#", CORRUPTED: "X", FAULT: "F"}


@dataclass(frozen=True)
class TrayPiece:
    slot_idx: int
    box: tuple  # (x0, y0, x1, y1) in frame pixels
    center: tuple  # (cx, cy) in frame pixels
    cells: tuple  # sorted tuple of (r, c) relative to (0, 0)
    rows: int
    cols: int
    has_special: bool = False  # bomb / medal icon inside piece

    @property
    def size(self):
        return len(self.cells)


def layout(w, h):
    """Return (scale, xoff, yoff) mapping 1280x720 base coords onto a w x h frame."""
    s = h / BASE_H
    return s, (w - BASE_W * s) / 2.0, 0.0


def to_frame(x, y, w=int(BASE_W), h=int(BASE_H)):
    s, ox, oy = layout(w, h)
    return ox + x * s, oy + y * s


def cell_rect(r, c, w=int(BASE_W), h=int(BASE_H)):
    """Pixel rect (x0, y0, x1, y1) of board cell (r, c) in a w x h frame."""
    s, ox, oy = layout(w, h)
    x0 = ox + (BX0 + c * PITCH + 1.0) * s
    y0 = oy + (BY0 + r * PITCH + 1.0) * s
    size = (PITCH - 2.0) * s
    return x0, y0, x0 + size, y0 + size


def cell_center(r, c, w=int(BASE_W), h=int(BASE_H)):
    x0, y0, x1, y1 = cell_rect(r, c, w, h)
    return 0.5 * (x0 + x1), 0.5 * (y0 + y1)


def _crop_base(arr, x0, y0, x1, y1, w, h):
    s, ox, oy = layout(w, h)
    ix0 = max(0, int(round(ox + x0 * s)))
    iy0 = max(0, int(round(oy + y0 * s)))
    ix1 = min(w, int(round(ox + x1 * s)))
    iy1 = min(h, int(round(oy + y1 * s)))
    return arr[iy0:iy1, ix0:ix1]


def detect_tray_slots(frame, hsv=None):
    """Dynamically find all 1/2/3 dark square tray boxes on the right side.

    Uses the four inner corners of each ~115x115 dark slot box (no polyomino—not even
    5x1, 1x5, or 3x3—ever covers the corners), so 2-slot and 3-slot layouts are detected
    reliably even when a 5x1 bar touches the left/right mid-edges.
    Returns list of (x0, y0, x1, y1) tuples in frame pixel coordinates, top to bottom.
    """
    h, w = frame.shape[:2]
    s, ox, oy = layout(w, h)
    if hsv is None:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    y_start = max(0, int(round(oy + TRAY_Y_MIN * s)))
    y_end = min(h, int(round(oy + TRAY_Y_MAX * s)))

    x0 = int(round(ox + TRAY_X0 * s))
    x1 = int(round(ox + TRAY_X1 * s))
    l_out = max(0, min(w - 1, int(round(ox + (TRAY_X0 - 9) * s))))
    r_out = max(0, min(w - 1, int(round(ox + (TRAY_X1 + 9) * s))))
    box_h = int(round(115 * s))

    v = hsv[:, :, 2]
    # Scan vertical strips near left & right inner edges (x0+2..x0+7 and x1-7..x1-2).
    # For a 5x1 horizontal bar, only the middle ~26px rows of the 115px slot are bright,
    # while the top ~44px and bottom ~44px are completely dark (V < 70).
    # Morphological closing with a 36*s kernel bridges any 5x1 bar gap cleanly!
    l0, l1 = max(0, x0 + int(round(2 * s))), min(w, x0 + int(round(8 * s)))
    r0, r1 = max(0, x1 - int(round(8 * s))), min(w, x1 - int(round(2 * s)))
    if l1 <= l0 or r1 <= r0 or y_end <= y_start:
        return []

    v_slice = v[y_start:y_end]
    in_dark = (v_slice[:, l0:l1].mean(axis=1) < 70) & (v_slice[:, r0:r1].mean(axis=1) < 70)
    out_bright = (v_slice[:, l_out] > 110) & (v_slice[:, r_out] > 110)
    row_ok = (in_dark & out_bright).astype(np.uint8).reshape(-1, 1)

    # Bridge vertical interruptions up to 27*s pixels (a 5x1 bar is ~23px tall,
    # whereas the gap between adjacent tray slots is ~34-36px — never merges slots!)
    k_h = max(3, int(round(27 * s))) | 1
    closed = cv2.morphologyEx(row_ok, cv2.MORPH_CLOSE, np.ones((k_h, 1), dtype=np.uint8)).ravel() > 0

    min_h = int(round(88 * s))
    max_h = int(round(136 * s))
    slots = []
    in_seg = False
    seg0 = 0
    for idx, ok in enumerate(closed):
        if ok and not in_seg:
            seg0 = idx
            in_seg = True
        elif not ok and in_seg:
            seg_h = idx - seg0
            if min_h <= seg_h <= max_h:
                slots.append((x0, y_start + seg0, x1, y_start + idx - 1))
            in_seg = False
    if in_seg:
        seg_h = len(closed) - seg0
        if min_h <= seg_h <= max_h:
            slots.append((x0, y_start + seg0, x1, y_start + len(closed) - 1))

    # Sanity-verify that the 4 inner corners of each detected slot are actually dark
    verified = []
    c_off = max(4, int(round(8 * s)))
    for sx0, sy0, sx1, sy1 in slots:
        if sy1 - sy0 < box_h * 0.7:
            continue
        corners = [
            v[sy0 + c_off, sx0 + c_off],
            v[sy0 + c_off, sx1 - c_off],
            v[sy1 - c_off, sx0 + c_off],
            v[sy1 - c_off, sx1 - c_off],
        ]
        if sum(1 for c in corners if c < 75) >= 3:
            verified.append((sx0, sy0, sx1, sy1))
    return verified


def in_level(frame):
    """True only when an active playable 拉海洛方块·逆转 board is unobstructed on screen."""
    if frame is None or frame.size == 0:
        return False
    h, w = frame.shape[:2]
    if h < 300 or w < 500:
        return False
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # 1. Top-right (i) white circle button visible
    info_roi = _crop_base(hsv, 1132, 28, 1162, 58, w, h)
    if info_roi.size == 0:
        return False
    white_frac = ((info_roi[..., 2] > 215) & (info_roi[..., 1] < 50)).mean()
    if white_frac < 0.14:
        return False

    # 2. Board outer top lavender frame bright, inner top rim dark
    top_frame = _crop_base(hsv, 420, 41, 860, 50, w, h)
    top_inner = _crop_base(hsv, 420, 52, 860, 56, w, h)
    if top_frame[..., 2].mean() < 160 or top_inner[..., 2].mean() > 135:
        return False

    # 3. Left frame mid-band bright (rejects dark tutorial/dialog overlays covering the center!)
    left_frame = _crop_base(hsv, 320, 290, 329, 430, w, h)
    if left_frame[..., 2].mean() < 140:
        return False

    # 4. At least one valid dark square tray slot detected on the right
    slots = detect_tray_slots(frame, hsv)
    if not slots:
        return False

    return True


def _classify_cell(bgr_patch, hsv_patch):
    """Classify a single board cell interior into EMPTY / NORMAL / CORRUPTED / FAULT."""
    mean_v = float(hsv_patch[..., 2].mean())
    mean_s = float(hsv_patch[..., 1].mean())
    mean_b, mean_g, mean_r = [float(bgr_patch[..., i].mean()) for i in range(3)]

    # Empty cell: dark navy-blue background (even rows 0-1 with purple top-glow have R<55, G<55, V<108)
    if mean_v < 110 and mean_r < 62 and mean_g < 62 and mean_s > 55:
        return EMPTY

    # Grey 数据损坏方块 (corrupted/indestructible): neutral grey (low saturation, mid brightness)
    rgb_spread = max(mean_r, mean_g, mean_b) - min(mean_r, mean_g, mean_b)
    if mean_s < 38 and rgb_spread < 26 and 70 <= mean_v <= 185:
        return CORRUPTED

    # Extra fallback for very dark cells
    if mean_v < 95:
        return EMPTY

    # Distinguish FAULT (1级/2级故障方块: bright frosted white inner highlight / sharp glitch cracks)
    # from flat NORMAL matte tiles.
    # On flat normal tiles (including bright yellow), pixels with V>242 & S<55 are <1%,
    # and inner V standard deviation is typically < 22.
    bright_frost = float(((hsv_patch[..., 2] > 238) & (hsv_patch[..., 1] < 65)).mean())
    v_std = float(hsv_patch[..., 2].std())
    if bright_frost > 0.08 or (bright_frost > 0.03 and v_std > 30.0):
        return FAULT

    return NORMAL


def read_board(frame):
    """Return 8x8 int ndarray of cell types (EMPTY/NORMAL/CORRUPTED/FAULT)."""
    h, w = frame.shape[:2]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    grid = np.zeros((N, N), dtype=np.int32)
    for r in range(N):
        for c in range(N):
            x0, y0, x1, y1 = cell_rect(r, c, w, h)
            mx = (x1 - x0) * 0.14
            my = (y1 - y0) * 0.14
            ix0, iy0 = int(round(x0 + mx)), int(round(y0 + my))
            ix1, iy1 = int(round(x1 - mx)), int(round(y1 - my))
            grid[r, c] = _classify_cell(frame[iy0:iy1, ix0:ix1], hsv[iy0:iy1, ix0:ix1])
    return grid


def _is_connected_8(cells):
    if not cells:
        return False
    s = set(cells)
    stack = [cells[0]]
    seen = {cells[0]}
    while stack:
        r, c = stack.pop()
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                nb = (r + dr, c + dc)
                if nb in s and nb not in seen:
                    seen.add(nb)
                    stack.append(nb)
    return len(seen) == len(s)


def decode_slot_piece(slot_bgr, slot_idx, box):
    """Decode the polyomino shape inside a single tray slot crop. Returns TrayPiece or None."""
    sh, sw = slot_bgr.shape[:2]
    if sh < 30 or sw < 30:
        return None
    pad = max(2, int(round(min(sw, sh) * 0.025)))
    inner = slot_bgr[pad : sh - pad, pad : sw - pad]
    ih, iw = inner.shape[:2]
    hsv = cv2.cvtColor(inner, cv2.COLOR_BGR2HSV)

    # Dark slot background is uniform V~41; any piece pixel has V > 78
    fg = hsv[..., 2] > 78
    min_fg_area = max(40, int(0.015 * iw * ih))
    if int(fg.sum()) < min_fg_area:
        return None

    # Morphological cleanup against 1px specular specks
    fg_u8 = fg.astype(np.uint8)
    ys, xs = np.where(fg_u8 > 0)
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    pw = x1 - x0 + 1
    ph = y1 - y0 + 1

    # Search integer grid dimensions (rows, cols) in 1..5 x 1..5
    # Empirically, mini-cell pitch is ~0.209 * sw for 1..4 cells and ~0.184 * sw for 5 cells.
    best_score = -1e9
    best_cells = None
    best_dims = None

    for rows in range(1, 6):
        for cols in range(1, 6):
            cw = pw / cols
            ch = ph / rows
            aspect = min(cw, ch) / max(cw, ch)
            if aspect < 0.76:
                continue
            c_avg = 0.5 * (cw + ch)
            ratio = c_avg / sw
            if ratio < 0.145 or ratio > 0.275:
                continue

            # Measure inner fill ratio of every (r, c) cell
            fills = np.zeros((rows, cols), dtype=np.float32)
            for r in range(rows):
                cy0 = y0 + int(round((r + 0.18) * ch))
                cy1 = y0 + int(round((r + 0.82) * ch))
                for c in range(cols):
                    cx0 = x0 + int(round((c + 0.18) * cw))
                    cx1 = x0 + int(round((c + 0.82) * cw))
                    patch = fg_u8[max(0, cy0) : max(cy0 + 1, cy1), max(0, cx0) : max(cx0 + 1, cx1)]
                    fills[r, c] = float(patch.mean()) if patch.size else 0.0

            occupied = fills > 0.45
            if not occupied.any():
                continue
            # Tight bounding box check: every edge row/col must contain at least one cell
            if not (occupied[0, :].any() and occupied[-1, :].any() and occupied[:, 0].any() and occupied[:, -1].any()):
                continue

            # Cleanliness: cells should be strongly near 0 or near 1 (bimodal)
            ambiguity = float(np.minimum(fills, 1.0 - fills).mean())
            max_amb = float(np.minimum(fills, 1.0 - fills).max())

            # Pitch prior: ~0.209 for all pieces (5x1 bar uses full 120px = 5*24px -> ~0.205)
            expected_ratio = 0.205 if max(rows, cols) == 5 else 0.209
            pitch_err = abs(ratio - expected_ratio)

            # Internal 1px gap check: between adjacent occupied cells, the seam should dip slightly
            cells = tuple(sorted((r, c) for r in range(rows) for c in range(cols) if occupied[r, c]))
            conn_bonus = 0.5 if _is_connected_8(cells) else -2.0

            score = (
                -8.0 * ambiguity
                - 3.0 * max_amb
                - 12.0 * pitch_err
                - 3.0 * (1.0 - aspect)
                + conn_bonus
            )
            if score > best_score:
                best_score = score
                best_cells = cells
                best_dims = (rows, cols)

    if best_cells is None:
        return None

    # Detect special icon (bomb / medal badge) on piece cells: crisp bright white/silver mark
    piece_hsv = hsv[y0 : y1 + 1, x0 : x1 + 1]
    special_frac = float(((piece_hsv[..., 2] > 235) & (piece_hsv[..., 1] < 55)).mean())
    has_special = special_frac > 0.05

    bx0, by0, bx1, by1 = box
    center = (0.5 * (bx0 + bx1), 0.5 * (by0 + by1))
    return TrayPiece(
        slot_idx=slot_idx,
        box=box,
        center=center,
        cells=best_cells,
        rows=best_dims[0],
        cols=best_dims[1],
        has_special=has_special,
    )


def read_tray(frame):
    """Detect all tray slots and decode each piece. Returns list of (TrayPiece or None)."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    slots = detect_tray_slots(frame, hsv)
    pieces = []
    for i, box in enumerate(slots):
        x0, y0, x1, y1 = box
        crop = frame[y0 : y1 + 1, x0 : x1 + 1]
        pieces.append(decode_slot_piece(crop, i, box))
    return pieces


def read_state(frame):
    """Return (board_8x8, pieces_list) or None if not currently inside a playable level."""
    if not in_level(frame):
        return None
    return read_board(frame), read_tray(frame)


def format_board(grid):
    lines = ["    " + " ".join(str(c) for c in range(N))]
    for r in range(N):
        lines.append(f"r{r}: " + " ".join(CELL_NAMES[int(grid[r, c])] for c in range(N)))
    return "\n".join(lines)


def format_piece(piece):
    if piece is None:
        return "(empty)"
    grid = [["." for _ in range(piece.cols)] for _ in range(piece.rows)]
    for r, c in piece.cells:
        grid[r][c] = "#"
    tag = " [SPECIAL]" if piece.has_special else ""
    return f"Slot {piece.slot_idx} ({piece.rows}x{piece.cols}, {piece.size} cells){tag}:\n" + "\n".join(
        "  " + "".join(row) for row in grid
    )


def render_preview(frame, grid, pieces, plan, out_path):
    """Save an annotated visual debug image showing detected board, tray pieces, and planned moves."""
    img = frame.copy()
    h, w = img.shape[:2]

    # 1. Draw detected board cells
    colors = {
        EMPTY: (70, 70, 70),
        NORMAL: (80, 210, 80),
        CORRUPTED: (180, 180, 180),
        FAULT: (255, 190, 60),
    }
    for r in range(N):
        for c in range(N):
            x0, y0, x1, y1 = [int(round(v)) for v in cell_rect(r, c, w, h)]
            kind = int(grid[r, c])
            cv2.rectangle(img, (x0 + 2, y0 + 2), (x1 - 2, y1 - 2), colors[kind], 1 if kind == EMPTY else 2)
            if kind != EMPTY:
                cv2.putText(img, CELL_NAMES[kind], (x0 + 6, y0 + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[kind], 1)

    # 2. Draw detected tray boxes and step badges
    step_colors = [(50, 60, 255), (0, 165, 255), (255, 220, 40)]  # Step 1 Red, Step 2 Orange, Step 3 Cyan
    step_by_slot = {}
    if plan:
        for step_idx, mv in enumerate(plan.moves):
            step_by_slot[mv.slot_idx] = (step_idx, mv)

    for p in pieces:
        if p is None:
            continue
        x0, y0, x1, y1 = p.box
        info = step_by_slot.get(p.slot_idx)
        col = step_colors[info[0] % len(step_colors)] if info else (200, 200, 200)
        cv2.rectangle(img, (x0, y0), (x1, y1), col, 3)
        label = f"S{p.slot_idx} {p.rows}x{p.cols}"
        if info:
            step_idx, mv = info
            label = f"#{step_idx + 1} rot={mv.rot * 90}"
        cv2.putText(img, label, (x0, max(18, y0 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 2)

    # 3. Draw planned move target footprints on the 8x8 board
    if plan:
        for step_idx, mv in enumerate(plan.moves):
            col = step_colors[step_idx % len(step_colors)]
            inset = 6 + step_idx * 4
            for r, c in mv.board_cells:
                x0, y0, x1, y1 = [int(round(v)) for v in cell_rect(r, c, w, h)]
                cv2.rectangle(img, (x0 + inset, y0 + inset), (x1 - inset, y1 - inset), col, 3)
                cv2.putText(
                    img,
                    f"{step_idx + 1}",
                    (int((x0 + x1) * 0.5) - 6, int((y0 + y1) * 0.5) + 6),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.75,
                    col,
                    2,
                )

    cv2.imencode(".png", img)[1].tofile(out_path)
