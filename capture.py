import ctypes
import cv2
import numpy as np
import win32gui
import win32ui

PW_RENDERFULLCONTENT = 2


def find_window_handle(title_substring: str):
    matches = []
    def enum_handler(hwnd, _):
        if win32gui.IsWindowVisible(hwnd):
            title = win32gui.GetWindowText(hwnd)
            if title_substring.lower() in title.lower():
                matches.append(hwnd)

    win32gui.EnumWindows(enum_handler, None)

    return matches[0] if matches else None


class WindowCapture:

    def __init__(self,window_title="HaxBall",crop=(485, 260, 1450, 720),resize=None,):
        self.window_title = window_title
        self.crop = crop
        self.resize = resize

        self.hwnd = None
        self.refresh_window()

    def refresh_window(self):

        self.hwnd = find_window_handle(self.window_title)

        if self.hwnd is None:
            raise RuntimeError(
                f"Cannot find window containing '{self.window_title}'"
            )

    def get_frame(self):

        if self.hwnd is None or not win32gui.IsWindow(self.hwnd):
            self.refresh_window()
        left, top, right, bottom = win32gui.GetClientRect(self.hwnd)

        width = right - left
        height = bottom - top

        if width <= 0 or height <= 0:
            return None

        hwnd_dc = None
        mfc_dc = None
        save_dc = None
        bitmap = None

        try:

            hwnd_dc = win32gui.GetWindowDC(self.hwnd)
            mfc_dc = win32ui.CreateDCFromHandle(hwnd_dc)
            save_dc = mfc_dc.CreateCompatibleDC()
            bitmap = win32ui.CreateBitmap()
            bitmap.CreateCompatibleBitmap(mfc_dc,width,height)
            save_dc.SelectObject(bitmap)

            ctypes.windll.user32.PrintWindow(
                self.hwnd,
                save_dc.GetSafeHdc(),
                PW_RENDERFULLCONTENT
            )

            bmp_info = bitmap.GetInfo()

            bmp_bits = bitmap.GetBitmapBits(True)

            frame = np.frombuffer(bmp_bits,dtype=np.uint8)
            frame.shape = (bmp_info["bmHeight"],bmp_info["bmWidth"],4)
            frame = frame[:, :, :3]
            x0, y0, x1, y1 = self.crop
            frame = frame[y0:y1, x0:x1]
            frame = np.ascontiguousarray(frame)

            if self.resize is not None:
                frame = cv2.resize(frame,self.resize,interpolation=cv2.INTER_AREA)
            return frame

        except Exception:

            return None

        finally:

            if bitmap is not None:
                win32gui.DeleteObject(bitmap.GetHandle())

            if save_dc is not None:
                save_dc.DeleteDC()

            if mfc_dc is not None:
                mfc_dc.DeleteDC()

            if hwnd_dc is not None:
                win32gui.ReleaseDC(self.hwnd, hwnd_dc)
