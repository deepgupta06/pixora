"""Removing a file the way a person would: to the Recycle Bin, so it can be got back."""

from __future__ import annotations

import os
import sys


def discard(path: str) -> None:
    """Send a file to the Recycle Bin. Where there is none, delete it."""
    if not os.path.isfile(path):
        return
    if sys.platform == "win32" and _recycle(os.path.abspath(path)):
        return
    os.remove(path)


def _recycle(path: str) -> bool:
    import ctypes
    from ctypes import wintypes

    class Operation(ctypes.Structure):  # SHFILEOPSTRUCTW
        _fields_ = [
            ("hwnd", wintypes.HWND),
            ("wFunc", wintypes.UINT),
            ("pFrom", wintypes.LPCWSTR),
            ("pTo", wintypes.LPCWSTR),
            ("fFlags", ctypes.c_ushort),
            ("fAnyOperationsAborted", wintypes.BOOL),
            ("hNameMappings", wintypes.LPVOID),
            ("lpszProgressTitle", wintypes.LPCWSTR),
        ]

    delete, allow_undo, no_confirmation, silent, no_error_ui = 0x0003, 0x0040, 0x0010, 0x0004, 0x0400
    operation = Operation(
        wFunc=delete,
        pFrom=path + "\0",  # the list of paths ends with a second null
        fFlags=allow_undo | no_confirmation | silent | no_error_ui,
    )
    failed = ctypes.windll.shell32.SHFileOperationW(ctypes.byref(operation))
    return failed == 0 and not os.path.exists(path)
