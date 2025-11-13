"""Utility helpers for routing log messages and exceptions through the ViewModel.

These helpers centralize how we surface log output to the GUI. They attempt to
emit messages via a viewmodel's ``log_message`` signal when available and fall
back to ``print`` so logs are never lost silently.
"""
from __future__ import annotations

import sys
import traceback
from typing import Optional


def log_message(message: str, vm: Optional[object] = None) -> None:
    """Emit *message* through ``vm.log_message`` when possible, else print it."""
    text = str(message)
    try:
        if vm is not None and hasattr(vm, "log_message"):
            signal = getattr(vm, "log_message", None)
            if signal is not None:
                try:
                    signal.emit(text)
                    return
                except Exception:
                    # fall through to printing below
                    pass
        print(text)
    except Exception:
        # never raise from the logger itself; last-resort stdout fallback
        try:
            print(text)
        except Exception:
            pass


def log_exception(context: str, exc: Optional[BaseException] = None, vm: Optional[object] = None) -> None:
    """Format *exc* with traceback and delegate to :func:`log_message`."""
    try:
        if exc is None:
            exc = sys.exc_info()[1]
        tb = traceback.format_exc()
        if exc is None and (tb is None or tb.strip() == "NoneType: None"):
            payload = f"{context}: (no exception details available)"
        else:
            payload = f"{context}: {exc}\n{tb}"
        log_message(payload, vm=vm)
    except Exception:
        # Avoid raising from logging even if formatting fails
        try:
            print(f"{context}: logging failed for exception {exc}")
        except Exception:
            pass
