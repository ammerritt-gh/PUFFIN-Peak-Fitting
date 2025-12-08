# models/__init__.py

"""Public API for the models package.

This module programmatically discovers ModelSpec implementations defined in
`model_specs.py` and YAML files in `model_elements/`, exporting them 
automatically so adding a new spec requires only creating a YAML file.

Exports provided:
  - get_model_spec(model_name) - factory to get model specs by name
  - Parameter, BaseModelSpec, CompositeModelSpec - core classes
  - ModelState - runtime state container
  - get_available_model_spec_classes() -> dict(name -> class)
  - get_available_model_names() -> list[str]
  - get_atomic_component_names() -> list[str] - names usable in composites
  - list_available_elements() -> list[str] - YAML-defined element names
  - reload_model_elements() - reload YAML definitions from disk

Model Element System:
  Model elements are defined in human-readable YAML files in the 
  `model_elements/` subfolder. Each file defines a single model with
  its parameters, hints, and evaluation expression. This allows for:
  - Easy sharing of model definitions
  - Human-readable and editable format
  - Dynamic discovery without code changes
  - Graceful handling of missing or invalid models

Note: model spec classes are detected by finding classes that subclass
`BaseModelSpec` (excluding the BaseModelSpec itself).
"""

from typing import Dict, List

from . import model_specs as _model_specs
from .model_state import ModelState

# Core helpers forwarded from model_specs
get_model_spec = _model_specs.get_model_spec
Parameter = _model_specs.Parameter
BaseModelSpec = _model_specs.BaseModelSpec
CompositeModelSpec = _model_specs.CompositeModelSpec
get_atomic_component_names = _model_specs.get_atomic_component_names

# Import model element utilities with graceful handling
try:
    from .model_elements import (
        list_available_elements,
        reload_elements as reload_model_elements,
        ModelElementError,
        ModelElementNotFoundError,
        ModelElementValidationError,
    )
except ImportError:
    # Fallback if model_elements module is not available
    def list_available_elements() -> List[str]:
        """Return list of available model elements."""
        return get_atomic_component_names()
    
    def reload_model_elements() -> None:
        """Reload model elements from disk (no-op fallback)."""
        pass
    
    class ModelElementError(Exception):
        """Exception raised when a model element cannot be loaded."""
        pass
    
    class ModelElementNotFoundError(ModelElementError):
        """Exception raised when a requested model element does not exist."""
        pass
    
    class ModelElementValidationError(ModelElementError):
        """Exception raised when a model element fails validation."""
        pass

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
    """Return sorted list of discovered model spec class names (e.g. 'VoigtModelSpec').
    
    This includes both hardcoded ModelSpec classes and composite models loaded from YAML files.
    Composite models from YAML are included with a 'ModelSpec' suffix for consistency.
    """
    names = list(_available_model_specs.keys())
    
    # Add composite models from YAML elements
    try:
        elements = list_available_elements()
        # Check which elements are composite models by trying to load them
        for elem_name in elements:
            try:
                # Try to load the model spec
                spec = get_model_spec(elem_name)
                # Check if it's a composite model
                if isinstance(spec, CompositeModelSpec):
                    # Add with 'ModelSpec' suffix for consistency
                    class_name = elem_name.replace(' ', '') + 'ModelSpec'
                    if class_name not in names:
                        names.append(class_name)
            except Exception:
                # Skip elements that can't be loaded
                continue
    except Exception:
        # If we can't access composite models, just use hardcoded ones
        pass
    
    return sorted(names)

# Build __all__ for clean `from models import *` behavior.
__all__ = [
    "get_model_spec",
    "Parameter",
    "BaseModelSpec",
    "CompositeModelSpec",
    "ModelState",
    "get_available_model_spec_classes",
    "get_available_model_names",
    "get_atomic_component_names",
    "list_available_elements",
    "reload_model_elements",
    "ModelElementError",
    "ModelElementNotFoundError",
    "ModelElementValidationError",
]
# Append discovered class names
__all__.extend(list(_available_model_specs.keys()))
