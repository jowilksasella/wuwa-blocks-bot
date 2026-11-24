"""Full-batch permutation + 4-rotation search solver for「拉海洛方块 · 逆转」(8x8).

Faithfully models:
  - 90-degree clockwise piece rotation (0/90/180/270, deduplicated per shape)
  - Multi-line clear scoring: 1->200, 2->500, 3->1000, 4->1500, 5->2500 (LINE_SCORES)
  - Consecutive clear combo bonus
  - CORRUPTED (grey 数据损坏方块): indestructible, stays occupied when row/col clears
  - FAULT (1级/2级故障方块): stays occupied after 1st clear (fault layer stripped), prioritized
  - SPECIAL pieces (bomb/medal): bonus when cleared in a completed line
  - Bombs (marks): a cleared bomb tile also clears its 3x3 (无角) / 5x5 (有角) area; chains into
    other bombs. A bomb on a FAULT (crystal) tile only loses the crystal and does not detonate.
  - Medals (收集奖章模式): counted when a medal tile is actually removed (crystal medals first
    turn NORMAL). The level is won by medal count, so medals dominate the ranking score.
  - Full-batch deadlock avoidance: always prefers plans that place every piece in the current batch
"""
from dataclasses import dataclass
from itertools import permutations
import numpy as np

from vision import N, EMPTY, NORMAL, CORRUPTED, FAULT, MARK_NONE, MARK_MEDAL, BOMB_RADIUS

# Ranking weight per collected medal. Larger than a fault hit (28) and a 2-line bonus, so the
# search prefers collecting medals; board health still breaks ties and prevents deadlock.
MEDAL_WEIGHT = 60.0


@dataclass(frozen=True)
class Move:
    slot_idx: int
    rot: int  # 0, 1, 2, 3 clockwise 90-degree turns
    top_r: int
    top_c: int
    board_cells: tuple  # tuple of (r, c) on the 8x8 board
    center_rc: tuple    # (center_r, center_c) of piece bounding box on 8x8 board
    lines_cleared: int
    points: int
    faults_hit: int
    special_cleared: int
    medals: int = 0      # medal tiles removed by this move (lines + blasts)
    bombs: int = 0       # bombs detonated by this move (incl. chain)


@dataclass(frozen=True)
class Plan:
    moves: tuple  # tuple of Move (1..3 moves covering current batch)
    score: float
    total_points: int
    total_lines: int
    total_faults_hit: int
    complete_batch: bool
    final_grid: np.ndarray
    total_medals: int = 0
    final_marks: np.ndarray = None


def normalize_cells(cells):
    min_r = min(r for r, _ in cells)
    min_c = min(c for _, c in cells)
    norm = tuple(sorted((r - min_r, c - min_c) for r, c in cells))
    rows = max(r for r, _ in norm) + 1
    cols = max(c for _, c in norm) + 1
    return norm, rows, cols


def rotate_cw(cells, k=1):
    """Rotate relative (r, c) cells clockwise by k * 90 degrees."""
    cur, rows, cols = normalize_cells(cells)
    for _ in range(k % 4):
        cur = tuple((c, rows - 1 - r) for r, c in cur)
        cur, rows, cols = normalize_cells(cur)
    return cur, rows, cols


def rotate_marked_cw(cells, mark_map=None, k=1):
    """Rotate cells clockwise k*90 deg, carrying per-cell marks {(r, c): MARK_*} along.

    Returns (norm_cells, rows, cols, rotated_mark_map)."""
    mark_map = mark_map or {}
    cur = [(r, c, mark_map.get((r, c), MARK_NONE)) for r, c in cells]
    for _ in range(k % 4):
        rows = max(r for r, _, _ in cur) + 1
        cur = [(c, rows - 1 - r, t) for r, c, t in cur]
    min_r = min(r for r, _, _ in cur)
    min_c = min(c for _, c, _ in cur)
    cur = [(r - min_r, c - min_c, t) for r, c, t in cur]
    norm = tuple(sorted((r, c) for r, c, _ in cur))
    rows = max(r for r, _ in norm) + 1
    cols = max(c for _, c in norm) + 1
    return norm, rows, cols, {(r, c): t for r, c, t in cur if t != MARK_NONE}


def rotate_tagged_cw(cells, specials=(), k=1):
    """Rotate cells clockwise k*90 deg, carrying the subset `specials` along.

    Returns (norm_cells, rows, cols, norm_specials)."""
    norm, rows, cols, mm = rotate_marked_cw(cells, {rc: MARK_MEDAL for rc in specials}, k)
    return norm, rows, cols, tuple(sorted(mm))


def piece_mark_map(piece):
    """{(r, c): MARK_*} of a TrayPiece. Icons not identified by a peek are assumed to be medals."""
    specials = getattr(piece, "specials", ())
    kinds = getattr(piece, "special_marks", ()) or (MARK_MEDAL,) * len(specials)
    return dict(zip(specials, kinds))


def unique_rotations(cells, allow_rotate=True, specials=(), mark_map=None):
    """Return list of (rot_count, norm_cells, rows, cols, rotated_mark_map) deduplicated by shape+icons."""
    if mark_map is None:
        mark_map = {rc: MARK_MEDAL for rc in specials}
    seen = set()
    out = []
    limit = 4 if allow_rotate else 1
    for rot in range(limit):
        norm, rows, cols, mm = rotate_marked_cw(cells, mark_map, rot)
        key = (norm, tuple(sorted(mm.items())))
        if key not in seen:
            seen.add(key)
            out.append((rot, norm, rows, cols, mm))
    return out


LINE_SCORES = {0: 0, 1: 200, 2: 500, 3: 1000, 4: 1500, 5: 2500}


def line_points(lines):
    if lines <= 0:
        return 0
    return LINE_SCORES.get(lines, 100 * lines * lines)


def _blast(clear_mask, ng, nm):
    """Extend clear_mask with bomb blasts (chained). Only bombs on NORMAL tiles detonate.

    Returns number of bombs detonated."""
    queue = [(r, c) for r, c in zip(*np.nonzero(clear_mask & (ng == NORMAL) & (nm >= 2)))]
    done = set()
    while queue:
        r, c = queue.pop()
        if (r, c) in done:
            continue
        done.add((r, c))
        rad = BOMB_RADIUS.get(int(nm[r, c]), 0)
        for rr in range(max(0, r - rad), min(N, r + rad + 1)):
            for cc in range(max(0, c - rad), min(N, c + rad + 1)):
                if not clear_mask[rr, cc]:
                    clear_mask[rr, cc] = True
                    if ng[rr, cc] == NORMAL and int(nm[rr, cc]) in BOMB_RADIUS:
                        queue.append((rr, cc))
    return len(done)


def apply_move_ex(grid, marks, board_cells, cell_marks=None):
    """Apply a placement with per-cell marks. Returns
    (next_grid, next_marks, lines, points, faults_hit, medals, bombs, cleared_mask)."""
    ng = grid.copy()
    nm = marks.copy() if marks is not None else np.zeros((N, N), dtype=np.int32)
    for i, (r, c) in enumerate(board_cells):
        ng[r, c] = NORMAL
        nm[r, c] = cell_marks[i] if cell_marks else MARK_NONE

    full_rows = [r for r in range(N) if np.all(ng[r, :] != EMPTY)]
    full_cols = [c for c in range(N) if np.all(ng[:, c] != EMPTY)]
    lines = len(full_rows) + len(full_cols)

    clear_mask = np.zeros((N, N), dtype=bool)
    if lines == 0:
        return ng, nm, 0, 0, 0, 0, 0, clear_mask
    for r in full_rows:
        clear_mask[r, :] = True
    for c in full_cols:
        clear_mask[:, c] = True
    bombs = _blast(clear_mask, ng, nm)

    # Each hit cell reacts once per move (assumption: a line + blast overlap is still one hit):
    # NORMAL -> EMPTY (mark collected / consumed), FAULT -> NORMAL (crystal stripped, mark kept),
    # CORRUPTED -> unchanged (assumed blast-proof as well).
    fault_cleared = clear_mask & (ng == FAULT)
    normal_cleared = clear_mask & (ng == NORMAL)
    medals = int((normal_cleared & (nm == MARK_MEDAL)).sum())
    ng[normal_cleared] = EMPTY
    nm[normal_cleared] = MARK_NONE
    ng[fault_cleared] = NORMAL
    return ng, nm, lines, line_points(lines), int(fault_cleared.sum()), medals, bombs, normal_cleared


def apply_move(grid, board_cells, has_special=False):
    """Apply a legal placement onto a copy of grid (no marks; legacy API).

    Returns (next_grid, lines_cleared, points, faults_hit, special_cleared).
    """
    ng, _, lines, pts, faults_hit, _, _, cleared = apply_move_ex(grid, None, board_cells)
    special_cleared = 1 if has_special and any(cleared[r, c] for r, c in board_cells) else 0
    return ng, lines, pts, faults_hit, special_cleared


def _hole_penalty(empty_mask):
    """Penalize tiny disconnected empty pockets (size 1, 2, 3) via fast flood fill."""
    visited = ~empty_mask
    pen = 0.0
    for r in range(N):
        for c in range(N):
            if visited[r, c]:
                continue
            # BFS pocket
            stack = [(r, c)]
            visited[r, c] = True
            size = 0
            while stack:
                cr, cc = stack.pop()
                size += 1
                if cr > 0 and not visited[cr - 1, cc]:
                    visited[cr - 1, cc] = True
                    stack.append((cr - 1, cc))
                if cr + 1 < N and not visited[cr + 1, cc]:
                    visited[cr + 1, cc] = True
                    stack.append((cr + 1, cc))
                if cc > 0 and not visited[cr, cc - 1]:
                    visited[cr, cc - 1] = True
                    stack.append((cr, cc - 1))
                if cc + 1 < N and not visited[cr, cc + 1]:
                    visited[cr, cc + 1] = True
                    stack.append((cr, cc + 1))
            if size == 1:
                pen += 18.0
            elif size == 2:
                pen += 7.0
            elif size == 3:
                pen += 2.5
    return pen


def evaluate_board(grid):
    """Heuristic board-health score (higher = healthier, easier to clear & survive)."""
    occ = grid != EMPTY
    emp = ~occ
    empty_cnt = int(emp.sum())

    # 1. Row & column smoothness (fewer empty<->occupied transitions = cleaner open space)
    h_trans = int((occ[:, :-1] != occ[:, 1:]).sum())
    v_trans = int((occ[:-1, :] != occ[1:, :]).sum())
    transitions = h_trans + v_trans

    # 2. Isolated tiny holes (1x1 / 1x2 / 3-cell pockets are lethal in Block Blast)
    hole_pen = _hole_penalty(emp)

    # 3. Room for giant pieces (3x3 squares & 5-long bars)
    emp_i = emp.astype(np.int32)
    # 3x3 sliding sum via integral shifts
    s3_h = emp_i[:, :-2] + emp_i[:, 1:-1] + emp_i[:, 2:]
    s3 = s3_h[:-2, :] + s3_h[1:-1, :] + s3_h[2:, :]
    fit_3x3 = int((s3 == 9).sum())

    s5_h = emp_i[:, :-4] + emp_i[:, 1:-3] + emp_i[:, 2:-2] + emp_i[:, 3:-1] + emp_i[:, 4:]
    s5_v = emp_i[:-4, :] + emp_i[1:-3, :] + emp_i[2:-2, :] + emp_i[3:-1, :] + emp_i[4:, :]
    fit_5 = int((s5_h == 5).sum()) + int((s5_v == 5).sum())

    # 4. Almost-full lines (6/8 or 7/8 filled) set up future multi-line combos
    row_sums = occ.sum(axis=1)
    col_sums = occ.sum(axis=0)
    near_full = int(((row_sums == 6) | (row_sums == 7)).sum() + ((col_sums == 6) | (col_sums == 7)).sum())

    # 5. Remaining fault blocks penalty (strongly drives clearing lines across fault blocks)
    faults_left = int((grid == FAULT).sum())

    # 6. Bonus for completely clean board (all clear!)
    clean_bonus = 40.0 if empty_cnt == N * N else 0.0

    return (
        1.5 * empty_cnt
        - 1.35 * transitions
        - hole_pen
        + 4.5 * min(fit_3x3, 3)
        + 1.0 * max(0, fit_3x3 - 3)
        + 2.5 * min(fit_5, 4)
        + 1.2 * near_full
        - 14.0 * faults_left
        + clean_bonus
    )


def legal_placements(grid, piece, allow_rotate=True):
    """Generate all legal (rot, top_r, top_c, board_cells, center_rc, cell_marks) for a TrayPiece."""
    occ = grid != EMPTY
    out = []
    for rot, norm, rows, cols, mm in unique_rotations(piece.cells, allow_rotate=allow_rotate, mark_map=piece_mark_map(piece)):
        cell_marks = tuple(mm.get(rc, MARK_NONE) for rc in norm)
        for tr in range(N - rows + 1):
            for tc in range(N - cols + 1):
                ok = True
                for dr, dc in norm:
                    if occ[tr + dr, tc + dc]:
                        ok = False
                        break
                if ok:
                    bcells = tuple((tr + dr, tc + dc) for dr, dc in norm)
                    center_rc = (tr + (rows - 1) * 0.5, tc + (cols - 1) * 0.5)
                    out.append((rot, tr, tc, bcells, center_rc, cell_marks))
    return out


def solve_batch(grid, pieces, allow_rotate=True, beam_width=20, marks=None):
    """Find the optimal ordered sequence of rotated placements for the current tray batch.

    pieces: list of (TrayPiece or None)
    marks: optional 8x8 MARK_* array (medals / bombs) from vision.read_board_marks
    Returns Plan or None if no legal placement exists for any piece.
    """
    if marks is None:
        marks = np.zeros((N, N), dtype=np.int32)
    active = [p for p in pieces if p is not None]
    if not active:
        return None

    best_complete = None
    best_partial = None

    # Deduplicate identical piece shapes across permutations (only if same cells & special)
    seen_orders = set()
    for order in permutations(active):
        sig = tuple((p.cells, tuple(sorted(piece_mark_map(p).items()))) for p in order)
        if sig in seen_orders:
            continue
        seen_orders.add(sig)

        # Beam entries: (rank_score, cur_grid, moves_tuple, cum_points, cum_lines, cum_faults, cum_spec, cum_bonus, streak, cur_marks, cum_medals)
        beam = [(0.0, grid, (), 0, 0, 0, 0, 0.0, 0, marks, 0)]

        for depth, piece in enumerate(order):
            next_beam = []
            for _, cur_g, mvs, pts, lines_tot, faults_tot, spec_tot, bonus_tot, streak, cur_m, med_tot in beam:
                placements = legal_placements(cur_g, piece, allow_rotate=allow_rotate)
                for rot, tr, tc, bcells, center_rc, cmarks in placements:
                    ng, nm, ln, gain, fh, med, nb, cleared = apply_move_ex(cur_g, cur_m, bcells, cmarks)
                    sp = 1 if piece.has_special and any(cleared[r, c] for r, c in bcells) else 0
                    new_streak = streak + 1 if ln > 0 else 0
                    combo_bonus = 18.0 * (new_streak - 1) if new_streak >= 2 else 0.0
                    multi_bonus = 22.0 * max(0, ln - 1) ** 2
                    # Slight preference for fewer rotations & clearing earlier when tied
                    rot_tie = -0.05 * rot
                    early_tie = 0.04 * ln * (len(order) - depth)
                    new_bonus = bonus_tot + multi_bonus + combo_bonus + rot_tie + early_tie
                    h_val = evaluate_board(ng)

                    mv = Move(
                        slot_idx=piece.slot_idx,
                        rot=rot,
                        top_r=tr,
                        top_c=tc,
                        board_cells=bcells,
                        center_rc=center_rc,
                        lines_cleared=ln,
                        points=gain,
                        faults_hit=fh,
                        special_cleared=sp,
                        medals=med,
                        bombs=nb,
                    )
                    new_pts = pts + gain
                    new_lines = lines_tot + ln
                    new_faults = faults_tot + fh
                    new_spec = spec_tot + sp
                    new_med = med_tot + med

                    total_score = (
                        new_pts * 0.14
                        + new_faults * 28.0
                        + new_spec * 32.0
                        + new_med * MEDAL_WEIGHT
                        + new_bonus
                        + h_val
                    )
                    next_beam.append(
                        (total_score, ng, mvs + (mv,), new_pts, new_lines, new_faults, new_spec, new_bonus, new_streak, nm, new_med)
                    )

            if not next_beam:
                # Record longest partial plan in case all full orders deadlock
                for sc, cur_g, mvs, pts, lines_tot, faults_tot, spec_tot, _, _, cur_m, med_tot in beam:
                    if mvs:
                        cand = Plan(mvs, sc, pts, lines_tot, faults_tot, False, cur_g, med_tot, cur_m)
                        if best_partial is None or (len(cand.moves), cand.score) > (
                            len(best_partial.moves),
                            best_partial.score,
                        ):
                            best_partial = cand
                beam = []
                break

            next_beam.sort(key=lambda x: x[0], reverse=True)
            # Keep wider beam on 2-piece batches (essentially exact), fast beam on 3-piece
            width = max(beam_width, 32) if len(order) <= 2 else beam_width
            beam = next_beam[:width]

        if beam:
            sc, cur_g, mvs, pts, lines_tot, faults_tot, _, _, _, cur_m, med_tot = beam[0]
            cand = Plan(mvs, sc, pts, lines_tot, faults_tot, True, cur_g, med_tot, cur_m)
            if best_complete is None or cand.score > best_complete.score:
                best_complete = cand

    return best_complete if best_complete is not None else best_partial
