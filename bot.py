"""Auto-play bot for Wuthering Waves「拉海洛方块 · 逆转」(Client-Win64-Shipping.exe).

Runs 100% locally (PrintWindow capture + OpenCV thresholds + batch permutation/rotation
solver + Win32 SendInput drag). No network calls, no game memory read/write.

Usage:
    python bot.py                  full auto-play (foreground-only, F8 to stop)
    python bot.py --dry            capture once, print board/tray/plan, save preview.png
    python bot.py --moves 1        play 1 piece and stop (recommended for first test)
    python bot.py --overlay        launch zero-input click-through overlay (btw-helper style)
    python bot.py --no-rotate      solve & play without rotating pieces
    python bot.py --rotate r       override rotation input method (rmb/r/space/wheel/click_tray/none)
    python bot.py --probe          hold a piece over center of board & save probe_mid.png / probe_rot.png
"""
import argparse
import json
import os
import sys
import time

import cv2
import numpy as np
import win32api
import win32gui

import inputs
import solver
import vision
from capture import grab
from win import find_window, geometry

HERE = os.path.dirname(os.path.abspath(__file__))
VK_F8 = 0x77
_last_status = None

DEFAULT_CFG = {
    "input_mode": "foreground",
    "idle_wait_s": 2.0,
    "allow_rotate": True,
    "rotate_method": "rmb",
    "drag_offset_x": 0.0,
    "drag_offset_y": 0.0,
    "drag_steps": 12,
    "step_ms": 14,
    "hold_ms": 130,
    "rotate_wait_ms": 150,
    "post_up_ms": 110,
    "post_drop_wait_s": 0.50,
    "batch_refill_wait_s": 0.85,
    "beam_width": 20,
    # 托盤方塊預覽：把方塊拖到棋盤左側、畫面垂直中間（不放下）看清每格是炸彈還是獎章，再拖回原槽。
    # "off"（預設）= 不拖出預覽，只看形狀；小圖上看到的圖示一律當獎章（不額外拖動，不拖慢節奏）
    # "icons" = 只預覽小圖有圖示的方塊；"all" = 每塊都預覽
    "peek_tray": "off",
    "peek_hold_s": 0.25,
}


def load_config():
    cfg = dict(DEFAULT_CFG)
    try:
        with open(os.path.join(HERE, "config.json"), encoding="utf-8") as f:
            cfg.update(json.load(f))
    except FileNotFoundError:
        pass
    return cfg


def status(text):
    global _last_status
    if text != _last_status:
        print(time.strftime("[%H:%M:%S]"), text, flush=True)
        _last_status = text


def f8_pressed():
    return bool(win32api.GetAsyncKeyState(VK_F8) & 0x8001)


def wait(seconds):
    end = time.time() + seconds
    while time.time() < end:
        if f8_pressed():
            return True
        time.sleep(0.04)
    return False


def state_sig(grid, pieces, marks=None):
    return (
        grid.tobytes(),
        marks.tobytes() if marks is not None else b"",
        tuple((p.slot_idx, p.cells, p.specials) if p else None for p in pieces),
    )


def stable_state(hwnd, tries=8, gap=0.14):
    """Wait until two consecutive frames agree on board + tray shapes."""
    prev_sig = None
    for _ in range(tries):
        if f8_pressed():
            return None
        frame = grab(hwnd)
        res = vision.read_state_full(frame)
        if res is None:
            return None
        grid, marks, pieces = res
        sig = state_sig(grid, pieces, marks)
        if sig == prev_sig:
            return frame, grid, pieces, marks
        prev_sig = sig
        if wait(gap):
            return None
    return None


def ready(hwnd, input_mode="post_message"):
    if not win32gui.IsWindow(hwnd):
        return "遊戲視窗已關閉"
    if win32gui.IsIconic(hwnd):
        return "暫停：鳴潮視窗已最小化（請還原視窗，可留在背景被其他視窗遮住）"
    if input_mode == "foreground" and win32gui.GetForegroundWindow() != hwnd:
        return "暫停：鳴潮視窗不在前景（切回遊戲自動繼續）"
    frame = grab(hwnd)
    if not vision.in_level(frame):
        return "等待進入「拉海洛方塊 · 逆轉」關卡畫面"
    return None


def target_frame_xy(mv, w, h, cfg):
    s, ox, oy = vision.layout(w, h)
    cr, cc = mv.center_rc
    tx = ox + (vision.BX0 + (cc + 0.5) * vision.PITCH + float(cfg["drag_offset_x"])) * s
    ty = oy + (vision.BY0 + (cr + 0.5) * vision.PITCH + float(cfg["drag_offset_y"])) * s
    return tx, ty


def run_probe(hwnd, cfg):
    """Safe mid-drag calibration probe (works in background via PostMessage): grabs a piece,
    hovers over board center, tests rotate, captures probe_mid.png & probe_rot.png, then drags back."""
    import win32con

    res = stable_state(hwnd)
    if res is None:
        sys.exit("未偵測到關卡畫面，請先進入拉海洛方塊關卡。")
    frame, grid, pieces, marks = res
    active = [p for p in pieces if p is not None]
    if not active:
        sys.exit("右側沒有可用方塊。")
    piece = next((p for p in active if len(solver.unique_rotations(p.cells)) > 1), active[0])
    h, w = frame.shape[:2]
    gx, gy, _, _ = geometry(hwnd)
    sx, sy = piece.center
    cx, cy = vision.cell_center(3, 3, w, h)
    is_bg = win32gui.GetForegroundWindow() != hwnd
    saved = win32api.GetCursorPos() if is_bg else None

    def move_c(x, y, btn=0):
        win32api.SetCursorPos((int(round(gx + x)), int(round(gy + y))))
        inputs._post(hwnd, win32con.WM_MOUSEMOVE, btn, inputs._lp(x, y))

    print(f"正在測試背景拖曳槽位 {piece.slot_idx + 1} ({piece.rows}x{piece.cols}) 並執行 rotate_method={cfg['rotate_method']}...", flush=True)
    try:
        if is_bg:
            inputs._post(hwnd, win32con.WM_ACTIVATE, win32con.WA_ACTIVE, 0)
            time.sleep(0.03)
        move_c(sx, sy, 0)
        time.sleep(0.05)
        inputs._post(hwnd, win32con.WM_LBUTTONDOWN, win32con.MK_LBUTTON, inputs._lp(sx, sy))
        time.sleep(0.08)
        for i in range(1, 13):
            move_c(sx + (cx - sx) * i / 12, sy + (cy - sy) * i / 12, win32con.MK_LBUTTON)
            time.sleep(0.015)
        time.sleep(0.18)
        mid_img = grab(hwnd)
        cv2.imencode(".png", mid_img)[1].tofile(os.path.join(HERE, "probe_mid.png"))

        inputs._post_rotate(hwnd, cx, cy, method=cfg["rotate_method"], wait_s=0.20)
        rot_img = grab(hwnd)
        cv2.imencode(".png", rot_img)[1].tofile(os.path.join(HERE, "probe_rot.png"))

        for i in range(1, 10):
            move_c(cx + (sx - cx) * i / 9, cy + (sy - cy) * i / 9, win32con.MK_LBUTTON)
            time.sleep(0.012)
        inputs._post(hwnd, win32con.WM_LBUTTONUP, 0, inputs._lp(sx, sy))
    finally:
        if is_bg:
            inputs._post(hwnd, win32con.WM_ACTIVATE, win32con.WA_INACTIVE, 0)
            if saved is not None:
                win32api.SetCursorPos(saved)

    diff = float(np.abs(rot_img.astype(np.int16) - mid_img.astype(np.int16)).mean())
    print(f"已儲存 probe_mid.png 與 probe_rot.png（旋轉前後像素平均差異: {diff:.2f}）。")
    if diff < 0.8:
        print("⚠ 差異極小，代表目前 rotate_method 可能無效，建議試 --rotate rmb 或 --rotate r。")
    else:
        print("✅ 偵測到明顯畫面變化，請檢視 probe_mid.png 與 probe_rot.png 確認方向與落點偏移。")


def wait_mouse_idle(idle_s, poll=0.10):
    """In background mode, yield while the user is actively moving their mouse.

    Returns True if F8 was pressed while waiting, False once cursor stayed still for idle_s.
    """
    if idle_s <= 0:
        return False
    last_pos = win32api.GetCursorPos()
    still_since = time.time()
    while time.time() - still_since < idle_s:
        if f8_pressed():
            return True
        cur = win32api.GetCursorPos()
        if abs(cur[0] - last_pos[0]) > 3 or abs(cur[1] - last_pos[1]) > 3:
            last_pos = cur
            still_since = time.time()
            status(f"偵測到你正在移動滑鼠，暫讓控制權（停手 {idle_s:.1f} 秒後自動背景繼續）")
        time.sleep(poll)
    return False


def peek_tray(hwnd, frame, todo, cfg, cache):
    """Drag each piece in `todo` out to the peek spot, screenshot it at tile scale, drag it back.

    Fills cache[(slot_idx, cells)] = {(r, c): MARK_*} for all 4 orientations (the game keeps a
    rotation made while dragging). Returns True if F8 was pressed."""
    allow_rotate = bool(cfg["allow_rotate"])
    input_mode = cfg.get("input_mode", "foreground")
    h, w = frame.shape[:2]
    hx, hy = vision.to_frame(vision.PEEK_X, vision.PEEK_Y, w, h)
    os.makedirs(os.path.join(HERE, "captures"), exist_ok=True)
    for p in todo:
        if f8_pressed():
            return True
        if ready(hwnd, input_mode=input_mode):
            return False
        k = vision.peek_rotation(p, allow_rotate)
        held_cells, rows, cols, _ = solver.rotate_marked_cw(p.cells, None, k)
        shot = {}

        def on_hold():
            time.sleep(float(cfg.get("peek_hold_s", 0.25)))
            shot["f"] = grab(hwnd)

        sx, sy = p.center
        inputs.drag_and_place(
            hwnd, sx, sy, hx, hy,
            rot=k,
            rotate_method=cfg["rotate_method"],
            input_mode=input_mode,
            steps=int(cfg["drag_steps"]),
            step_ms=int(cfg["step_ms"]),
            hold_ms=int(cfg["hold_ms"]),
            rotate_wait_ms=int(cfg["rotate_wait_ms"]),
            post_up_ms=int(cfg.get("post_up_ms", 110)),
            on_hold=on_hold,
            return_xy=(sx, sy),
        )
        key = (p.slot_idx, p.cells)
        loc = vision.locate_held_piece(frame, shot["f"], rows, cols) if "f" in shot else None
        if "f" in shot:
            cv2.imwrite(os.path.join(HERE, "captures", f"peek_slot{p.slot_idx + 1}.png"), shot["f"])
        if loc is None:
            print(f"  預覽槽位 {p.slot_idx + 1}：找不到被拖出的方塊，圖示暫當獎章", flush=True)
            cache[key] = None
            continue
        held = vision.read_held_marks(shot["f"], loc, held_cells)
        for j in (1, 2, 3, 0):  # held orientation (j=0) last so it wins for symmetric shapes
            cj, _, _, mj = solver.rotate_marked_cw(held_cells, held, j)
            cache[(p.slot_idx, cj)] = mj
        names = {vision.MARK_MEDAL: "獎章", vision.MARK_BOMB: "炸彈3x3", vision.MARK_BOMB_BIG: "炸彈5x5"}
        found = "、".join(f"{names[m]}@{rc}" for rc, m in sorted(held.items())) or "無圖示"
        print(f"  預覽槽位 {p.slot_idx + 1}（{rows}x{cols}{'，已順轉90°' if k else ''}）：{found}", flush=True)
    return False


def apply_peek(pieces, cache):
    out = []
    for p in pieces:
        if p is not None:
            mm = cache.get((p.slot_idx, p.cells))
            if mm is not None:
                p = vision.with_marks(p, mm)
        out.append(p)
    return out


def play_loop(hwnd, cfg, max_moves=0, trace=False):
    played = 0
    consecutive_misses = 0
    allow_rotate = bool(cfg["allow_rotate"])
    rot_method = cfg["rotate_method"]
    input_mode = cfg.get("input_mode", "foreground")
    idle_wait_s = float(cfg.get("idle_wait_s", 2.0))
    beam_width = int(cfg["beam_width"])
    peek_mode = str(cfg.get("peek_tray", "off")).lower()
    peek_cache = {}
    prev_count = 0

    while not max_moves or played < max_moves:
        if f8_pressed():
            return played

        blocked = ready(hwnd, input_mode=input_mode)
        if blocked:
            status(blocked)
            if wait(0.35):
                return played
            continue

        # In background mode, never steal cursor while the user is actively moving their mouse
        if input_mode == "post_message" and win32gui.GetForegroundWindow() != hwnd:
            if wait_mouse_idle(idle_wait_s):
                return played
            if ready(hwnd, input_mode=input_mode):
                continue

        res = stable_state(hwnd)
        if res is None:
            if wait(0.2):
                return played
            continue

        frame, grid, pieces, marks = res
        active = [p for p in pieces if p is not None]
        if not active:
            status("等待右側刷新新一輪方塊...")
            if wait(float(cfg["batch_refill_wait_s"])):
                return played
            continue

        # Tray refilled -> new pieces may reuse a slot + shape with different icons
        if len(active) > prev_count:
            peek_cache.clear()
        prev_count = len(active)
        if peek_mode != "off":
            todo = [
                p for p in active
                if (p.slot_idx, p.cells) not in peek_cache and (p.specials or peek_mode == "all")
            ]
            if todo:
                status("預覽托盤方塊圖示中...")
                if peek_tray(hwnd, frame, todo, cfg, peek_cache):
                    return played
                if wait(0.15):
                    return played
                continue  # re-read: a peeked 5-wide piece now sits upright in its slot
            pieces = apply_peek(pieces, peek_cache)
            active = [p for p in pieces if p is not None]

        t0 = time.perf_counter()
        plan = solver.solve_batch(grid, pieces, allow_rotate=allow_rotate, beam_width=beam_width, marks=marks)
        solve_ms = (time.perf_counter() - t0) * 1000.0

        if plan is None or not plan.moves:
            status("目前找不到合法放置位置（可能已卡死或關卡結算中）")
            if wait(1.0):
                return played
            continue

        # Re-check readiness right before sending input
        if ready(hwnd, input_mode=input_mode):
            continue

        mv = plan.moves[0]
        piece = next((p for p in active if p.slot_idx == mv.slot_idx), None)
        if piece is None:
            continue

        status(f"自動遊玩進行中（模式={input_mode}，按 F8 隨時停止）")
        if trace:
            print(vision.format_board(grid, marks), flush=True)
            for p in active:
                print(vision.format_piece(p), flush=True)

        h, w = frame.shape[:2]
        sx, sy = piece.center
        tx, ty = target_frame_xy(mv, w, h, cfg)

        print(
            f"  第 {played + 1} 步 | 槽位 {mv.slot_idx + 1} | 順轉 {mv.rot * 90}° | "
            f"落點 ({mv.top_r + 1},{mv.top_c + 1}) | 消 {mv.lines_cleared} 行/列 (+{mv.points}) | "
            f"本輪總計 {plan.total_lines} 行/列"
            + (f" | 獎章 +{mv.medals}（本輪 {plan.total_medals}）" if plan.total_medals or mv.medals else "")
            + (f" | 引爆 {mv.bombs} 個炸彈" if mv.bombs else "")
            + f" | 耗時 {solve_ms:.0f}ms",
            flush=True,
        )

        before_sig = state_sig(grid, pieces, marks)
        inputs.drag_and_place(
            hwnd,
            sx,
            sy,
            tx,
            ty,
            rot=mv.rot,
            rotate_method=rot_method,
            input_mode=input_mode,
            steps=int(cfg["drag_steps"]),
            step_ms=int(cfg["step_ms"]),
            hold_ms=int(cfg["hold_ms"]),
            rotate_wait_ms=int(cfg["rotate_wait_ms"]),
            post_up_ms=int(cfg.get("post_up_ms", 110)),
        )

        # Wait for placement & clear animation to settle and verify state changed
        if wait(float(cfg["post_drop_wait_s"])):
            return played

        changed = False
        deadline = time.time() + 1.4
        while time.time() < deadline:
            if f8_pressed():
                return played
            cur = grab(hwnd)
            if not vision.in_level(cur):
                changed = True
                break
            cur_g, cur_m = vision.read_board_marks(cur)
            cur_p = vision.read_tray(cur)
            if state_sig(cur_g, cur_p, cur_m) != before_sig:
                changed = True
                break
            if wait(0.12):
                return played

        if changed:
            played += 1
            consecutive_misses = 0
            # Extra settle time after a line clear animation or batch refill
            extra = 0.55 if mv.lines_cleared > 0 else 0.0
            if len(active) == 1:
                extra = max(extra, float(cfg["batch_refill_wait_s"]))
            if extra > 0 and wait(extra):
                return played
        else:
            consecutive_misses += 1
            print(
                f"  ⚠ 第 {consecutive_misses} 次放置未生效（棋盤與右側槽無變化）。"
                + (f" 若方塊未旋轉，請調整 config.json 的 rotate_method（目前={rot_method}）或 drag_offset_y。" if mv.rot > 0 else " 請檢查遊戲是否以管理員權限執行，或用 --probe 校準 drag_offset_y。"),
                flush=True,
            )
            if consecutive_misses >= 3:
                status("連續 3 次放置失敗，已暫停等待手動調整（切走再切回可重試，按 F8 退出）")
                if wait(2.0):
                    return played
                consecutive_misses = 0

    return played


def main():
    ap = argparse.ArgumentParser(description="鳴潮「拉海洛方塊 · 逆轉」自動遊玩腳本")
    ap.add_argument("--dry", action="store_true", help="只讀盤計算並輸出 preview.png，不發送任何滑鼠輸入")
    ap.add_argument("--overlay", action="store_true", help="啟動透明點擊穿透提示層（不操作滑鼠，參考 btw-helper）")
    ap.add_argument("--probe", action="store_true", help="拖曳校準模式：抓起方塊測旋轉與落點偏移並存圖，不放下")
    ap.add_argument("--moves", type=int, default=0, help="最多執行 N 步後停止（0 = 無限）")
    ap.add_argument("--no-rotate", action="store_true", help="停用方塊旋轉")
    ap.add_argument("--rotate", type=str, default=None, help="覆寫旋轉方式: rmb / r / space / wheel / none")
    ap.add_argument("--foreground", action="store_true", help="前景模式（預設：切走鳴潮絕對不碰滑鼠）")
    ap.add_argument("--bg", action="store_true", help="背景自動模式（使用 PostMessage＋停手防搶滑鼠，停手數秒才動）")
    ap.add_argument("--idle", type=float, default=None, help="背景模式下偵測滑鼠停手幾秒才動作（預設 2.0）")
    ap.add_argument("--trace", action="store_true", help="每步列印詳細棋盤與方塊形狀")
    args = ap.parse_args()

    if args.overlay:
        import overlay

        ov = overlay.Overlay(allow_rotate=not args.no_rotate)
        print("覆蓋層已啟動（純視覺提示模式）：切回鳴潮即顯示，Ctrl+C 退出。", flush=True)
        try:
            ov.root.mainloop()
        except KeyboardInterrupt:
            pass
        return

    cfg = load_config()
    if args.no_rotate:
        cfg["allow_rotate"] = False
    if args.rotate is not None:
        cfg["rotate_method"] = args.rotate
    if args.foreground:
        cfg["input_mode"] = "foreground"
    if args.bg:
        cfg["input_mode"] = "post_message"
    if args.idle is not None:
        cfg["idle_wait_s"] = args.idle

    hwnd = find_window()
    while not hwnd:
        status("等待鳴潮視窗 (Client-Win64-Shipping.exe)...")
        if wait(1.0):
            return
        hwnd = find_window()

    x, y, w, h = geometry(hwnd)
    print(f"已找到鳴潮視窗 HWND={hwnd} 客戶區={w}x{h} (位置 {x},{y})", flush=True)

    if args.dry:
        frame = grab(hwnd)
        ok = vision.in_level(frame)
        print("in_level:", ok)
        if not ok:
            sys.exit("目前不在「拉海洛方塊 · 逆轉」關卡畫面（或有教學彈窗遮擋）。")
        grid, marks, pieces = vision.read_state_full(frame)
        print(vision.format_board(grid, marks))
        for p in pieces:
            print(vision.format_piece(p))
        t0 = time.perf_counter()
        plan = solver.solve_batch(
            grid, pieces, allow_rotate=bool(cfg["allow_rotate"]), beam_width=int(cfg["beam_width"]), marks=marks
        )
        dt = (time.perf_counter() - t0) * 1000.0
        if plan:
            print(
                f"\n求解完成 ({dt:.1f} ms) | 完整放完={plan.complete_batch} | 預計消 {plan.total_lines} 行/列 "
                f"(+{plan.total_points}分) | 預計收集獎章 {plan.total_medals}:"
            )
            for i, mv in enumerate(plan.moves):
                print(
                    f"  #{i + 1}: 槽位{mv.slot_idx + 1} 順轉{mv.rot * 90}° -> "
                    f"({mv.top_r + 1},{mv.top_c + 1}) 格={mv.board_cells} 消{mv.lines_cleared} 獎章{mv.medals} 炸彈{mv.bombs}"
                )
        else:
            print("找不到合法解。")
        out = os.path.join(HERE, "preview.png")
        vision.render_preview(frame, grid, pieces, plan, out, marks=marks)
        print("已輸出標註預覽圖:", out)
        return

    if args.probe:
        run_probe(hwnd, cfg)
        return

    print(
        f"自動遊玩已啟動（mode={cfg['input_mode']}, rotate={cfg['allow_rotate']}, method={cfg['rotate_method']}, "
        f"offset=({cfg['drag_offset_x']},{cfg['drag_offset_y']})）。按 F8 隨時結束。",
        flush=True,
    )
    played = play_loop(hwnd, cfg, max_moves=args.moves, trace=args.trace)
    print(f"已結束，共成功放置 {played} 塊方塊。", flush=True)


if __name__ == "__main__":
    main()
