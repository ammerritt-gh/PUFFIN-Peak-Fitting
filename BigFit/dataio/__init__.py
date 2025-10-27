from .data_loader import select_and_load_files, load_data_from_file
from .data_saver import save_dataset

# Lazy wrapper to avoid importing configuration at package import time (prevents circular imports)
def get_config(*args, **kwargs):
    from .configuration import get_config as _get_config
    return _get_config(*args, **kwargs)

__all__ = ["select_and_load_files", "load_data_from_file", "save_dataset", "get_config"]

