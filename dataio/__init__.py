from .data_loader import select_and_load_files, load_data_from_file
from .data_saver import save_dataset

# Lazy wrapper to avoid importing configuration at package import time (prevents circular imports)
def get_config(*args, **kwargs):
    from .configuration import get_config as _get_config
    return _get_config(*args, **kwargs)

# Lazy wrappers for fit persistence to avoid circular imports
def save_default_fit(*args, **kwargs):
    from .fit_persistence import save_default_fit as _save_default_fit
    return _save_default_fit(*args, **kwargs)

def load_default_fit(*args, **kwargs):
    from .fit_persistence import load_default_fit as _load_default_fit
    return _load_default_fit(*args, **kwargs)

def save_fit_for_file(*args, **kwargs):
    from .fit_persistence import save_fit_for_file as _save_fit_for_file
    return _save_fit_for_file(*args, **kwargs)

def load_fit_for_file(*args, **kwargs):
    from .fit_persistence import load_fit_for_file as _load_fit_for_file
    return _load_fit_for_file(*args, **kwargs)

def has_fit_for_file(*args, **kwargs):
    from .fit_persistence import has_fit_for_file as _has_fit_for_file
    return _has_fit_for_file(*args, **kwargs)

def reset_fit_for_file(*args, **kwargs):
    from .fit_persistence import reset_fit_for_file as _reset_fit_for_file
    return _reset_fit_for_file(*args, **kwargs)

def reset_default_fit(*args, **kwargs):
    from .fit_persistence import reset_default_fit as _reset_default_fit
    return _reset_default_fit(*args, **kwargs)

__all__ = [
    "select_and_load_files",
    "load_data_from_file",
    "save_dataset",
    "get_config",
    "save_default_fit",
    "load_default_fit",
    "save_fit_for_file",
    "load_fit_for_file",
    "has_fit_for_file",
    "reset_fit_for_file",
    "reset_default_fit",
]

