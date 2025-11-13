"""
Dock widgets package for the BigFit application.

Each dock is a self-contained module that can be developed independently.
"""

from .controls_dock import ControlsDock
from .parameters_dock import ParametersDock
from .elements_dock import ElementsDock
from .log_dock import LogDock

__all__ = ["ControlsDock", "ParametersDock", "ElementsDock", "LogDock"]
