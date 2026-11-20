"""Click-through transparent overlay for「拉海洛方块 · 逆转」(inspired by btw-helper).

Reads the live 8x8 board and right-hand tray from Client-Win64-Shipping.exe,
computes the optimal batch placement + rotation plan, and draws colour-coded
step footprints directly over the game window WITHOUT sending any input.

    python overlay.py            run live click-through overlay (Ctrl+C to quit)
    python overlay.py --always   also draw when the game window is not foreground
    python overlay.py --once     read once, print plan, save preview.png
    python overlay.py --no-rotate  disable rotation in recommendations
"""
import json
import os
import sys
import tkinter as tk

import win32con
import win32gui

import solver
import vision
from capture import grab
from win import find_window, geometry

HERE = os.path.dirname(os.path.abspath(__file__))
KEY = "#010101"  # transparent colour key
STEP_COLORS = ["#ff3b30", "#ff9500", "#00d4ff"]
ROT_LABELS = {0: "不轉", 1: "順轉90°", 2: "轉180°", 3: "順轉270°"}


def load_config():
    cfg = {"allow_rotate": True, "beam_width": 20, "overlay_interval_ms": 250}
    try:
        with open(os.path.join(HERE, "config.json"), encoding="utf-8") as f:
            cfg.update(json.load(f))
    except FileNotFoundError:
        pass
    return cfg


class Overlay:
    def __init__(self, allow_rotate=True, always=False):
        self.cfg = load_config()
        self.allow_rotate = allow_rotate and bool(self.cfg.get("allow_rotate", True))
        self.beam_width = int(self.cfg.get("beam_width", 20))
        self.interval_ms = int(self.cfg.get("overlay_interval_ms", 250))
        self.always = always
        self.hwnd = None
        self.last_sig = None

        self.root = tk.Tk()
        self.root.overrideredirect(True)
        self.root.attributes("-topmost", True)
        self.root.attributes("-transparentcolor", KEY)
        self.root.configure(bg=KEY)
        self.root.geometry("1x1+0+0")
        self.canvas = tk.Canvas(self.root, bg=KEY, highlightthickness=0, bd=0)
        self.canvas.pack(fill="both", expand=True)
        self.root.update_idletasks()
        self._click_through()
        self.root.report_callback_exception = self._on_error
        self.root.after(100, self.tick)

    def _click_through(self):
        h = int(self.root.wm_frame(), 16)
        ex = win32gui.GetWindowLong(h, win32con.GWL_EXSTYLE)
        ex |= (
            win32con.WS_EX_LAYERED
            | win32con.WS_EX_TRANSPARENT
            | win32con.WS_EX_TOOLWINDOW
            | win32con.WS_EX_NOACTIVATE
            | win32con.WS_EX_TOPMOST
        )
        win32gui.SetWindowLong(h, win32con.GWL_EXSTYLE, ex)

    def _on_error(self, exc, val, tb):
        if issubclass(exc, KeyboardInterrupt):
            self.root.destroy()
            return
        import traceback

        traceback.print_exception(exc, val, tb)

    def clear(self):
        if self.last_sig is not None:
            self.canvas.delete("all")
            self.last_sig = None

    def tick(self):
        try:
            self.update()
        finally:
            self.root.after(self.interval_ms, self.tick)

    def update(self):
        if not (self.hwnd and win32gui.IsWindow(self.hwnd)):
            self.hwnd = find_window()
        if not self.hwnd or win32gui.IsIconic(self.hwnd):
            return self.clear()
        if not self.always and win32gui.GetForegroundWindow() != self.hwnd:
            return self.clear()

        x, y, w, h = geometry(self.hwnd)
        geo = f"{w}x{h}+{x}+{y}"
        if self.root.geometry() != geo:
            self.root.geometry(geo)

        frame = grab(self.hwnd)
        state = vision.read_state(frame)
        if state is None:
            return self.clear()
        grid, pieces = state
        active_sig = tuple((p.slot_idx, p.cells, p.has_special) if p else None for p in pieces)
        sig = (geo, grid.tobytes(), active_sig, self.allow_rotate)
        if sig == self.last_sig:
            return

        plan = solver.solve_batch(grid, pieces, allow_rotate=self.allow_rotate, beam_width=self.beam_width)
        self.canvas.delete("all")
        self.draw(w, h, grid, pieces, plan)
        self.last_sig = sig

    def draw(self, w, h, grid, pieces, plan):
        cv = self.canvas
        s, ox, _ = vision.layout(w, h)
        fs = max(9, int(11 * s))

        # Highlight corrupted & fault cells subtly
        for r in range(vision.N):
            for c in range(vision.N):
                kind = int(grid[r, c])
                if kind in (vision.CORRUPTED, vision.FAULT):
                    x0, y0, x1, y1 = vision.cell_rect(r, c, w, h)
                    col = "#b0b0b0" if kind == vision.CORRUPTED else "#ffd60a"
                    tag = "損" if kind == vision.CORRUPTED else "故"
                    cv.create_rectangle(x0 + 2 * s, y0 + 2 * s, x1 - 2 * s, y1 - 2 * s, outline=col, width=max(1, int(2 * s)))
                    cv.create_text(x0 + 6 * s, y0 + 4 * s, text=tag, anchor="nw", fill=col, font=("Microsoft YaHei", fs, "bold"))

        step_by_slot = {}
        if plan:
            for idx, mv in enumerate(plan.moves):
                step_by_slot[mv.slot_idx] = (idx, mv)
                col = STEP_COLORS[idx % len(STEP_COLORS)]
                inset = (5 + idx * 4) * s
                for r, c in mv.board_cells:
                    x0, y0, x1, y1 = vision.cell_rect(r, c, w, h)
                    cv.create_rectangle(
                        x0 + inset,
                        y0 + inset,
                        x1 - inset,
                        y1 - inset,
                        outline=col,
                        width=max(2, int(4 * s)),
                    )
                    cv.create_text(
                        0.5 * (x0 + x1),
                        0.5 * (y0 + y1),
                        text=str(idx + 1),
                        fill=col,
                        font=("Microsoft YaHei", int(fs * 1.8), "bold"),
                    )

        # Highlight tray slots with matching step colours & rotation label
        for p in pieces:
            if p is None:
                continue
            x0, y0, x1, y1 = p.box
            info = step_by_slot.get(p.slot_idx)
            if info:
                idx, mv = info
                col = STEP_COLORS[idx % len(STEP_COLORS)]
                cv.create_rectangle(x0, y0, x1, y1, outline=col, width=max(2, int(4 * s)))
                badge = f"第{idx + 1}步 · {ROT_LABELS[mv.rot]}"
                cv.create_rectangle(x0, y0 - 24 * s, x1 + 20 * s, y0 - 2 * s, fill="#1c1c1e", outline=col, width=1)
                cv.create_text(x0 + 6 * s, y0 - 22 * s, text=badge, anchor="nw", fill=col, font=("Microsoft YaHei", fs, "bold"))

        # Left status box (same visual style as btw-helper)
        bx, by = ox + 22 * s, 210 * s
        lines = ["拉海洛方塊 · 提示模式"]
        if plan:
            lines.append(f"本輪預計消除: {plan.total_lines} 行/列 (+{plan.total_points}分)")
            for idx, mv in enumerate(plan.moves):
                lines.append(
                    f"#{idx + 1} 槽{mv.slot_idx + 1} {ROT_LABELS[mv.rot]} → ({mv.top_r + 1},{mv.top_c + 1})"
                    + (f" 消{mv.lines_cleared}" if mv.lines_cleared else "")
                )
            if not plan.complete_batch:
                lines.append("⚠ 本輪無全放解，已選最優存活步")
        else:
            lines.append("等待方塊刷新 / 無可放位置")

        box_w = 275 * s
        box_h = (16 + 26 * len(lines)) * s
        cv.create_rectangle(bx, by, bx + box_w, by + box_h, fill="#1c1c1e", outline="#00d4ff", width=max(1, int(2 * s)))
        for i, text in enumerate(lines):
            fill = "#ffffff" if i == 0 else (STEP_COLORS[(i - 2) % len(STEP_COLORS)] if 2 <= i < 2 + (len(plan.moves) if plan else 0) else "#d8d8dc")
            cv.create_text(bx + 12 * s, by + (10 + 26 * i) * s, text=text, anchor="nw", fill=fill, font=("Microsoft YaHei", fs, "bold"))


def main():
    allow_rotate = "--no-rotate" not in sys.argv
    always = "--always" in sys.argv
    if "--once" in sys.argv:
        hwnd = find_window()
        if not hwnd:
            sys.exit("找不到鳴潮視窗 (Client-Win64-Shipping.exe)")
        frame = grab(hwnd)
        state = vision.read_state(frame)
        if state is None:
            sys.exit("畫面上沒有識別到拉海洛方塊 8x8 棋盤")
        grid, pieces = state
        plan = solver.solve_batch(grid, pieces, allow_rotate=allow_rotate)
        print(vision.format_board(grid))
        for p in pieces:
            print(vision.format_piece(p))
        if plan:
            print(f"\n推薦方案 (總消 {plan.total_lines} 行/列, +{plan.total_points}分):")
            for i, mv in enumerate(plan.moves):
                print(f"  #{i + 1}: 槽位{mv.slot_idx + 1} {ROT_LABELS[mv.rot]} -> 第{mv.top_r + 1}行第{mv.top_c + 1}列 (消{mv.lines_cleared})")
        out = os.path.join(HERE, "preview.png")
        vision.render_preview(frame, grid, pieces, plan, out)
        print("已儲存預覽圖:", out)
        return

    ov = Overlay(allow_rotate=allow_rotate, always=always)
    print("拉海洛方塊覆蓋層已啟動（純視覺提示、不發送任何滑鼠輸入）：切回鳴潮即顯示，Ctrl+C 退出。")
    try:
        ov.root.mainloop()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
