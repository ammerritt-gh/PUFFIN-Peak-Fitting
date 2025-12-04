# models/model_elements/__init__.py
"""Model elements sub-package.

This package contains individual model element specifications that can be
used standalone or combined in composite models. Models are defined in
human-readable YAML files and loaded on startup.

Structure:
  - Each model element is defined in its own .yaml file
  - The loader discovers and validates these files automatically
  - Models contain parameter definitions, hints, and evaluation logic

Usage:
    from models.model_elements import get_element_spec, list_available_elements
    
    # Get a specific element spec
    gaussian_spec = get_element_spec("Gaussian")
    
    # List all available elements
    elements = list_available_elements()
"""

from .loader import (
    get_element_spec,
    list_available_elements,
    get_element_spec_class,
    reload_elements,
    ModelElementError,
    ModelElementNotFoundError,
    ModelElementValidationError,
)

__all__ = [
    "get_element_spec",
    "list_available_elements",
    "get_element_spec_class",
    "reload_elements",
    "ModelElementError",
    "ModelElementNotFoundError",
    "ModelElementValidationError",
]
