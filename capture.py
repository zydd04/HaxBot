import ctypes
import numpy as np
import win32gui
import win32ui

PW_RENDERFULLCONTENT = 2


def find_window_handle(title_substring: str):
    """Return the hwnd of the first visible window whose title contains
    title_substring (case-insensitive), or None if not found."""
    matches = []

    def _enum_handler(hwnd, _):
        if win32gui.IsWindowVisible(hwnd):
            title = win32gui.GetWindowText(hwnd)
            if title_substring.lower() in title.lower():
                matches.append(hwnd)

    win32gui.EnumWindows(_enum_handler, None)
    return matches[0] if matches else None


class WindowCapture:
    """Captures a window's client area as a BGR numpy array on demand."""

    def __init__(self, title_substring: str):
        self.title_substring = title_substring
        self.hwnd = find_window_handle(title_substring)
        if self.hwnd is None:
            raise RuntimeError(
                f"No window found matching '{title_substring}'. "
                "Please open it first, then run the program."
            )

    def get_frame(self):
        """Return the specified (485, 260) to (1430, 700) sub-box from the window."""
        bx, by = 485, 260
        bwidth = 1450 - 485  # 945 pixels
        bheight = 720 - 260  # 440 pixels

        left, top, right, bottom = win32gui.GetClientRect(self.hwnd)
        win_width, win_height = right - left, bottom - top
        if win_width <= 0 or win_height <= 0:
            return None

        hwnd_dc = win32gui.GetWindowDC(self.hwnd)
        mfc_dc = win32ui.CreateDCFromHandle(hwnd_dc)
        save_dc = mfc_dc.CreateCompatibleDC()

        save_bitmap = win32ui.CreateBitmap()
        save_bitmap.CreateCompatibleBitmap(mfc_dc, win_width, win_height)
        save_dc.SelectObject(save_bitmap)

        ctypes.windll.user32.PrintWindow(self.hwnd, save_dc.GetSafeHdc(), PW_RENDERFULLCONTENT)

        bmp_info = save_bitmap.GetInfo()
        bmp_bits = save_bitmap.GetBitmapBits(True)

        frame = np.frombuffer(bmp_bits, dtype="uint8")
        frame.shape = (bmp_info["bmHeight"], bmp_info["bmWidth"], 4)
        frame = np.ascontiguousarray(frame[:, :, :3])  # BGRA -> BGR

        win32gui.DeleteObject(save_bitmap.GetHandle())
        save_dc.DeleteDC()
        mfc_dc.DeleteDC()
        win32gui.ReleaseDC(self.hwnd, hwnd_dc)

        bx = max(0, min(bx, frame.shape[1]))
        by = max(0, min(by, frame.shape[0]))
        bx1 = min(bx + bwidth, frame.shape[1])
        by1 = min(by + bheight, frame.shape[0])
        frame = frame[by:by1, bx:bx1]

        return frame