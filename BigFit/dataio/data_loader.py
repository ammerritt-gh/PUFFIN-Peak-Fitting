# model/data_loader.py
import os
import numpy as np
import pandas as pd
from PySide6.QtWidgets import QFileDialog, QMessageBox

DATA_FILE_FILTER = "Data Files (*.dat *.txt *.csv)"


def resolve_default_input_dir():
    """Find a good default input directory."""
    env = os.environ.get("FMO_ANALYSIS_INPUT_DIR")
    if env and os.path.isdir(env):
        return env
    docs = os.path.join(os.path.expanduser("~"), "Documents")
    return docs


def load_data_from_file(filepath: str):
    """Load (x, y, error) data from a file. Returns (energy, counts, errors, file_info)."""
    try:
        # Detect format (dat, csv, etc.)
        _, ext = os.path.splitext(filepath.lower())
        if ext in (".csv", ".txt"):
            df = pd.read_csv(filepath, delim_whitespace=True, comment="#")
        else:
            df = pd.read_table(filepath, delim_whitespace=True, comment="#")

        # Flexible column detection
        cols = list(df.columns)
        if len(cols) >= 2:
            energy = df.iloc[:, 0].to_numpy(dtype=float)
            counts = df.iloc[:, 1].to_numpy(dtype=float)
            if len(cols) >= 3:
                errors = df.iloc[:, 2].to_numpy(dtype=float)
            else:
                errors = np.sqrt(np.clip(np.abs(counts), 1e-12, np.inf))
        else:
            raise ValueError("File must have at least two numeric columns.")

        file_info = {
            "path": filepath,
            "name": os.path.basename(filepath),
            "size": os.path.getsize(filepath),
        }

        return energy, counts, errors, file_info

    except Exception as e:
        raise RuntimeError(f"Failed to load {os.path.basename(filepath)}: {e}") from e


def select_and_load_files(parent=None):
    """Show a file dialog and load one or more data files."""
    input_dir = resolve_default_input_dir()
    filepaths, _ = QFileDialog.getOpenFileNames(parent, "Select Data Files", input_dir, DATA_FILE_FILTER)
    if not filepaths:
        return []

    data_list = []
    for fp in filepaths:
        try:
            data = load_data_from_file(fp)
            data_list.append(data)
        except Exception as e:
            if parent:
                QMessageBox.warning(parent, "Load Error", str(e))
            else:
                print(f"[WARN] {e}")
    return data_list

