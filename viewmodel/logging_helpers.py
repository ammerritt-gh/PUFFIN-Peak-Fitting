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


def safe_call(func, *args, default=None, context: str = "operation", vm: Optional[object] = None, **kwargs):
    """Safely call a function, logging exceptions and returning default on failure.
    
    Args:
        func: Callable to execute
        *args: Positional arguments for func
        default: Value to return on exception (default: None)
        context: Description for error logging
        vm: ViewModel instance for logging
        **kwargs: Keyword arguments for func
        
    Returns:
        Result of func(*args, **kwargs) or default on exception
    """
    try:
        return func(*args, **kwargs)
    except Exception as e:
        log_exception(f"Failed during {context}", e, vm=vm)
        return default


def safe_emit(signal, *args, vm: Optional[object] = None, signal_name: str = "signal"):
    """Safely emit a Qt signal, catching and logging any exceptions.
    
    Args:
        signal: Qt signal to emit
        *args: Arguments to pass to signal.emit()
        vm: ViewModel instance for logging
        signal_name: Name of signal for error messages
    """
    try:
        signal.emit(*args)
    except Exception as e:
        log_exception(f"Failed to emit {signal_name}", e, vm=vm)
