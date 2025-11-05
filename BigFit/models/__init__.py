# models/__init__.py

"""Public API for the models package.

This module programmatically discovers ModelSpec implementations defined in
`model_specs.py` and exports them automatically so adding a new spec does
not require editing this file.

Exports provided:
  - get_model_spec(model_name)
  - Parameter, BaseModelSpec
  - ModelState
  - <Each ModelSpec class found in model_specs as a module-level name>
  - get_available_model_spec_classes() -> dict(name -> class)
  - get_available_model_names() -> list[str]

Note: model spec classes are detected by finding classes that subclass
`BaseModelSpec` (excluding the BaseModelSpec itself).
"""

from typing import Dict

from . import model_specs as _model_specs
from .model_state import ModelState

# Core helpers forwarded from model_specs
get_model_spec = _model_specs.get_model_spec
Parameter = _model_specs.Parameter
BaseModelSpec = _model_specs.BaseModelSpec

# Discover all ModelSpec subclasses in model_specs module
_available_model_specs: Dict[str, type] = {}
for _name, _obj in vars(_model_specs).items():
    try:
        if isinstance(_obj, type) and issubclass(_obj, BaseModelSpec) and _obj is not BaseModelSpec:
            _available_model_specs[_name] = _obj
    except Exception:
        # defensive: ignore anything that can't be tested
        continue

# Inject discovered classes into this module's globals so callers can import them
globals().update(_available_model_specs)

def get_available_model_spec_classes() -> Dict[str, type]:
    """Return a shallow copy of discovered model spec classes as name -> class."""
    return dict(_available_model_specs)

def get_available_model_names() -> list:
    """Return sorted list of discovered model spec class names (e.g. 'VoigtModelSpec')."""
    return sorted(_available_model_specs.keys())

# Build __all__ for clean `from models import *` behavior.
__all__ = [
    "get_model_spec",
    "Parameter",
    "BaseModelSpec",
    "ModelState",
    "get_available_model_spec_classes",
    "get_available_model_names",
]
# Append discovered class names
__all__.extend(list(_available_model_specs.keys()))
