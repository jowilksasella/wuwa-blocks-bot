"""Mouse & keyboard input for Wuthering Waves (Client-Win64-Shipping.exe).

Supports two control backends:
  1. 'post_message' (default, inspired by ok-ww PostMessageInteraction + GenshinInteraction):
     - Works even while the game window is in the BACKGROUND (not foreground, not minimized).
     - Sends WM_ACTIVATE(WA_ACTIVE) + SetCursorPos(exact client->screen) + PostMessage
       (WM_MOUSEMOVE / WM_LBUTTONDOWN / WM_RBUTTONDOWN / WM_KEYDOWN / WM_LBUTTONUP)
       + WM_ACTIVATE(WA_INACTIVE) and restores the user's cursor position immediately.
     - Zero multi-monitor virtual-desktop coordinate drift (verified <1px accuracy on dual displays).
  2. 'foreground' (inspired by match3-bot):
     - Acts only when the game is the foreground window, using SetCursorPos + SendInput.
"""
import ctypes
import time
from ctypes import wintypes
import win32api
import win32con
import win32gui

from win import geometry

user32 = ctypes.windll.user32

INPUT_MOUSE = 0
INPUT_KEYBOARD = 1

MOUSEEVENTF_LEFTDOWN = 0x0002
MOUSEEVENTF_LEFTUP = 0x0004
MOUSEEVENTF_RIGHTDOWN = 0x0008
MOUSEEVENTF_RIGHTUP = 0x0010
MOUSEEVENTF_WHEEL = 0x0800

KEYEVENTF_KEYUP = 0x0002
KEYEVENTF_SCANCODE = 0x0008

VK_MAP = {
    "r": (0x52, 0x13),
    "space": (0x20, 0x39),
    "q": (0x51, 0x10),
    "e": (0x45, 0x12),
}


class MOUSEINPUT(ctypes.Structure):
    _fields_ = [
        ("dx", wintypes.LONG),
        ("dy", wintypes.LONG),
        ("mouseData", wintypes.DWORD),
        ("dwFlags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("dwExtraInfo", ctypes.POINTER(wintypes.ULONG)),
    ]


class KEYBDINPUT(ctypes.Structure):
    _fields_ = [
        ("wVk", wintypes.WORD),
        ("wScan", wintypes.WORD),
        ("dwFlags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("dwExtraInfo", ctypes.POINTER(wintypes.ULONG)),
    ]


class _U(ctypes.Union):
    _fields_ = [("mi", MOUSEINPUT), ("ki", KEYBDINPUT), ("pad", ctypes.c_byte * 32)]


class INPUT(ctypes.Structure):
    _fields_ = [("type", wintypes.DWORD), ("u", _U)]


def _send_mouse(flags, data=0):
    inp = INPUT(type=INPUT_MOUSE)
    inp.u.mi = MOUSEINPUT(0, 0, data & 0xFFFFFFFF, flags, 0, None)
    return user32.SendInput(1, ctypes.byref(inp), ctypes.sizeof(INPUT))


def _send_key(vk, scan, keyup=False):
    inp = INPUT(type=INPUT_KEYBOARD)
    flags = KEYEVENTF_SCANCODE | (KEYEVENTF_KEYUP if keyup else 0)
    inp.u.ki = KEYBDINPUT(vk, scan, flags, 0, None)
    return user32.SendInput(1, ctypes.byref(inp), ctypes.sizeof(INPUT))


def to_screen(hwnd, x, y):
    gx, gy, _, _ = geometry(hwnd)
    return int(round(gx + x)), int(round(gy + y))


def move_screen(sx, sy):
    """Move OS cursor using exact physical desktop coordinates (immune to mixed-DPI virtual-desk drift)."""
    win32api.SetCursorPos((int(round(sx)), int(round(sy))))


def focus(hwnd):
    """Bring the game window forward; returns True when it is the foreground window."""
    if win32gui.GetForegroundWindow() == hwnd:
        return True
    win32api.keybd_event(win32con.VK_MENU, 0, 0, 0)
    win32api.keybd_event(win32con.VK_MENU, 0, win32con.KEYEVENTF_KEYUP, 0)
    try:
        win32gui.SetForegroundWindow(hwnd)
    except Exception:
        pass
    time.sleep(0.25)
    return win32gui.GetForegroundWindow() == hwnd


def _lp(x, y):
    return win32api.MAKELONG(int(round(x)), int(round(y)))


def _key_lparam(vk, is_up=False):
    sc = win32api.MapVirtualKey(vk, 0)
    lp = (sc << 16) | 1
    if is_up:
        lp |= (1 << 30) | (1 << 31)
    return lp


def _post(hwnd, msg, wp=0, lp=0):
    win32gui.PostMessage(hwnd, msg, wp, lp)


def _post_rotate(hwnd, x, y, method="rmb", wait_s=0.14):
    m = (method or "none").lower()
    if m == "none":
        return
    pos = _lp(x, y)
    if m == "rmb":
        _post(hwnd, win32con.WM_RBUTTONDOWN, win32con.MK_LBUTTON | win32con.MK_RBUTTON, pos)
        time.sleep(0.045)
        _post(hwnd, win32con.WM_RBUTTONUP, win32con.MK_LBUTTON, pos)
    elif m == "wheel":
        wp = win32api.MAKELONG(win32con.MK_LBUTTON, (-120) & 0xFFFF)
        _post(hwnd, win32con.WM_MOUSEWHEEL, wp, pos)
    elif m in VK_MAP:
        vk, _ = VK_MAP[m]
        _post(hwnd, win32con.WM_KEYDOWN, vk, _key_lparam(vk, is_up=False))
        time.sleep(0.045)
        _post(hwnd, win32con.WM_KEYUP, vk, _key_lparam(vk, is_up=True))
    else:
        raise ValueError(f"Unknown rotate_method: {method!r}")
    time.sleep(wait_s)


def _fg_rotate(method="rmb", wait_s=0.14):
    m = (method or "none").lower()
    if m == "none":
        return
    if m == "rmb":
        _send_mouse(MOUSEEVENTF_RIGHTDOWN)
        time.sleep(0.045)
        _send_mouse(MOUSEEVENTF_RIGHTUP)
    elif m == "wheel":
        _send_mouse(MOUSEEVENTF_WHEEL, data=-120)
    elif m in VK_MAP:
        vk, scan = VK_MAP[m]
        _send_key(vk, scan, keyup=False)
        time.sleep(0.045)
        _send_key(vk, scan, keyup=True)
    else:
        raise ValueError(f"Unknown rotate_method: {method!r}")
    time.sleep(wait_s)


def drag_and_place_post_message(
    hwnd,
    x0,
    y0,
    x1,
    y1,
    rot=0,
    rotate_method="rmb",
    steps=12,
    step_ms=14,
    hold_ms=130,
    rotate_wait_ms=150,
    post_up_ms=110,
    on_hold=None,
    return_xy=None,
):
    """Background-capable drag + rotate via PostMessage + SetCursorPos (ok-ww style).

    Works even when Client-Win64-Shipping.exe is covered by other windows (not minimized).
    Because UE4 throttles background FPS (~15-20 FPS = 50-66ms/frame), hold_ms and post_up_ms
    keep the cursor steady for >=2 frames at grab, drop, and release before restoring the cursor.
    """
    gx, gy, _, _ = geometry(hwnd)
    is_bg = win32gui.GetForegroundWindow() != hwnd
    saved_pos = win32api.GetCursorPos() if is_bg else None
    hold_s = hold_ms / 1000.0
    rot_s = rotate_wait_ms / 1000.0
    post_up_s = post_up_ms / 1000.0
    method = (rotate_method or "none").lower()

    def move_client(cx, cy, down_btn=0):
        win32api.SetCursorPos((int(round(gx + cx)), int(round(gy + cy))))
        _post(hwnd, win32con.WM_MOUSEMOVE, down_btn, _lp(cx, cy))

    try:
        if is_bg:
            _post(hwnd, win32con.WM_ACTIVATE, win32con.WA_ACTIVE, 0)
            time.sleep(0.04)

        # Grab piece at tray slot center
        move_client(x0, y0, 0)
        time.sleep(0.06)
        _post(hwnd, win32con.WM_LBUTTONDOWN, win32con.MK_LBUTTON, _lp(x0, y0))
        time.sleep(max(0.09, hold_s * 0.8))

        # Interpolate drag to target board bounding-box center
        for i in range(1, steps + 1):
            t = i / steps
            cx = x0 + (x1 - x0) * t
            cy = y0 + (y1 - y0) * t
            move_client(cx, cy, win32con.MK_LBUTTON)
            time.sleep(step_ms / 1000.0)

        # Hold at target so UE4 background frame registers exact cursor pos & snap preview
        move_client(x1, y1, win32con.MK_LBUTTON)
        time.sleep(hold_s)

        # Rotate clockwise while hovering over target cells
        if rot > 0 and method != "none":
            for _ in range(rot % 4):
                _post_rotate(hwnd, x1, y1, method=method, wait_s=rot_s)
            # Small 1px nudge so Unreal refreshes snap preview for the new orientation
            move_client(x1 + 1, y1, win32con.MK_LBUTTON)
            time.sleep(0.04)
            move_client(x1, y1, win32con.MK_LBUTTON)
            time.sleep(hold_s)

        if on_hold is not None:
            on_hold()
        if return_xy is not None:
            rx, ry = return_xy
            for i in range(1, steps + 1):
                t = i / steps
                move_client(x1 + (rx - x1) * t, y1 + (ry - y1) * t, win32con.MK_LBUTTON)
                time.sleep(step_ms / 1000.0)
            move_client(rx, ry, win32con.MK_LBUTTON)
            time.sleep(hold_s)
            x1, y1 = rx, ry

        # Release and wait >=2 background frames before restoring cursor
        _post(hwnd, win32con.WM_LBUTTONUP, 0, _lp(x1, y1))
        time.sleep(post_up_s)
    finally:
        if is_bg:
            _post(hwnd, win32con.WM_ACTIVATE, win32con.WA_INACTIVE, 0)
            if saved_pos is not None:
                win32api.SetCursorPos(saved_pos)


def drag_and_place_foreground(
    hwnd,
    x0,
    y0,
    x1,
    y1,
    rot=0,
    rotate_method="rmb",
    steps=12,
    step_ms=12,
    hold_ms=70,
    rotate_wait_ms=140,
    on_hold=None,
    return_xy=None,
):
    """Foreground drag + rotate via SetCursorPos + SendInput (match3-bot style).

    on_hold: called while the piece is still held at (x1, y1) (used to screenshot a peek).
    return_xy: if given, drag back there before releasing (peek: drop back on the tray slot).
    """
    sx, sy = to_screen(hwnd, x0, y0)
    ex, ey = to_screen(hwnd, x1, y1)
    hold_s = hold_ms / 1000.0
    rot_s = rotate_wait_ms / 1000.0
    method = (rotate_method or "none").lower()

    move_screen(sx, sy)
    time.sleep(0.045)
    _send_mouse(MOUSEEVENTF_LEFTDOWN)
    time.sleep(hold_s)

    for i in range(1, steps + 1):
        t = i / steps
        move_screen(sx + (ex - sx) * t, sy + (ey - sy) * t)
        time.sleep(step_ms / 1000.0)

    time.sleep(hold_s)

    if rot > 0 and method != "none":
        for _ in range(rot % 4):
            _fg_rotate(method=method, wait_s=rot_s)
        move_screen(ex + 1, ey)
        time.sleep(0.025)
        move_screen(ex, ey)
        time.sleep(hold_s)

    if on_hold is not None:
        on_hold()
    if return_xy is not None:
        rx, ry = to_screen(hwnd, *return_xy)
        for i in range(1, steps + 1):
            t = i / steps
            move_screen(ex + (rx - ex) * t, ey + (ry - ey) * t)
            time.sleep(step_ms / 1000.0)
        time.sleep(hold_s)

    _send_mouse(MOUSEEVENTF_LEFTUP)
    time.sleep(0.04)


def drag_and_place(
    hwnd,
    x0,
    y0,
    x1,
    y1,
    rot=0,
    rotate_method="rmb",
    input_mode="post_message",
    steps=12,
    step_ms=14,
    hold_ms=130,
    rotate_wait_ms=150,
    post_up_ms=110,
    on_hold=None,
    return_xy=None,
):
    if (input_mode or "post_message").lower() == "foreground":
        return drag_and_place_foreground(
            hwnd, x0, y0, x1, y1, rot=rot, rotate_method=rotate_method,
            steps=steps, step_ms=step_ms, hold_ms=hold_ms, rotate_wait_ms=rotate_wait_ms,
            on_hold=on_hold, return_xy=return_xy,
        )
    return drag_and_place_post_message(
        hwnd, x0, y0, x1, y1, rot=rot, rotate_method=rotate_method,
        steps=steps, step_ms=step_ms, hold_ms=hold_ms, rotate_wait_ms=rotate_wait_ms,
        post_up_ms=post_up_ms, on_hold=on_hold, return_xy=return_xy,
    )
