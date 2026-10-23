# model/data_loader.py
import os
from typing import Optional
import numpy as np
import pandas as pd
from PySide6.QtWidgets import QFileDialog, QMessageBox
import csv
import io
import re

DATA_FILE_FILTER = "Data Files (*.dat *.txt *.csv);;All Files (*)"


def resolve_default_input_dir():
    """Find a good default input directory."""
    # Prefer the configured default_load_folder when available
    try:
        # import wrapper from package to avoid importing submodule at module-import time
        from dataio import get_config
        cfg = get_config()
        cfg_folder = cfg.default_load_folder or None
        if cfg_folder and os.path.isdir(cfg_folder):
            return cfg_folder
    except Exception:
        # fall back if config not available
        pass

    # Next try environment variable
    env = os.environ.get("FMO_ANALYSIS_INPUT_DIR")
    if env and os.path.isdir(env):
        return env

    # Finally, use ~/Documents
    docs = os.path.join(os.path.expanduser("~"), "Documents")
    return docs


def _load_bl43_analyzer(raw_lines):
    """SPring-8 BL43LXU per-analyzer file (`..._aNN`): return (meV_sft, y_i, ye_i), or None if the
    column-label line does not name those columns. Rows with npts == 0 or ye_i <= 0 are dropped."""
    labels = None
    for line in raw_lines:
        s = line.strip()
        if s.startswith("#"):
            labels = s.lstrip("#").split()
        elif s:
            break  # first data row: the label line, if any, was the last comment above it
    if not labels or not {"meV_sft", "y_i", "ye_i", "npts"} <= set(labels):
        return None
    ix, iy, ie, ipts = (labels.index(n) for n in ("meV_sft", "y_i", "ye_i", "npts"))
    rows = [l.split() for l in raw_lines if l.strip() and not l.lstrip().startswith("#")]
    bad = [i + 1 for i, r in enumerate(rows) if len(r) != len(labels)]
    if bad:
        raise ValueError(f"BL43LXU data rows {bad[:5]} do not have {len(labels)} columns")
    arr = np.array(rows, dtype=float)
    keep = (arr[:, ipts] > 0) & (arr[:, ie] > 0)
    return arr[keep, ix], arr[keep, iy], arr[keep, ie]


def load_data_from_file(filepath: str):
    """Load (x, y, error) data from a file. Returns (energy, counts, errors, file_info)."""
    try:
        # Read a chunk of the file to detect delimiter and find where numeric data starts
        with open(filepath, "r", encoding="utf-8", errors="replace") as fh:
            raw_lines = fh.readlines()

        bl43 = _load_bl43_analyzer(raw_lines)
        if bl43 is not None:
            energy, counts, errors = bl43
            file_info = {
                "path": filepath,
                "name": os.path.basename(filepath),
                "size": os.path.getsize(filepath),
            }
            return energy, counts, errors, file_info

        # Improved delimiter detection: try csv.Sniffer then score common delimiters over many lines
        sample = "".join(raw_lines[:50])
        delim = None
        try:
            sniff = csv.Sniffer().sniff(sample)
            delim = sniff.delimiter
        except Exception:
            # score candidate delimiters by consistency of token counts across the sample lines
            candidates = [",", "\t", ";", None]  # None means whitespace
            best = None
            best_score = -1.0
            for cand in candidates:
                counts = []
                for line in raw_lines[:50]:
                    s = line.strip()
                    if s == "" or s.lstrip().startswith("#"):
                        continue
                    if cand is None:
                        toks = re.split(r"\s+", s)
                    else:
                        toks = s.split(cand)
                    # strip tokens and drop empty tokens (handles trailing commas)
                    toks = [t.strip() for t in toks if t.strip() != ""]
                    counts.append(len(toks))
                if not counts:
                    continue
                # score: median token count (prefers more fields) minus variance (prefers consistency)
                med = float(np.median(counts))
                var = float(np.var(counts))
                score = med - 0.2 * var
                if score > best_score:
                    best_score = score
                    best = cand
            delim = best

        # Split function based on detected delimiter; remove empty tokens and trailing commas
        def split_line(line):
            s = line.strip()
            if s == "":
                return []
            if delim is None:
                toks = re.split(r"\s+", s)
            else:
                toks = [t.strip() for t in s.split(delim)]
            # remove empty tokens and tokens that are just commas
            toks = [t for t in toks if t != "" and t != ","]
            return toks

        # Find first line that contains at least three numeric tokens (x, y, error).
        def numeric_tokens_from_line(line):
            toks = split_line(line)
            nums = []
            for t in toks:
                t2 = t.strip().strip(",")
                try:
                    nums.append(float(t2))
                except Exception:
                    pass
            return nums

        start_idx = None
        for i, line in enumerate(raw_lines):
            nums = numeric_tokens_from_line(line)
            if len(nums) >= 3:
                start_idx = i
                break
        # If no 3-number line found, fall back to first 2-number line
        if start_idx is None:
            for i, line in enumerate(raw_lines):
                nums = numeric_tokens_from_line(line)
                if len(nums) >= 2:
                    start_idx = i
                    break
        if start_idx is None:
            raise ValueError("No numeric data rows found in file.")

        # Parse numeric rows starting at start_idx. For each row take first 3 numeric values (x,y,error).
        numeric_rows = []
        for line in raw_lines[start_idx:]:
            if line.strip() == "" or line.lstrip().startswith("#"):
                continue
            nums = numeric_tokens_from_line(line)
            if len(nums) >= 3:
                numeric_rows.append(nums[:3])
            elif len(nums) == 2:
                numeric_rows.append([nums[0], nums[1], np.nan])
            else:
                # skip lines without at least 2 numeric values
                continue

        if not numeric_rows:
            raise ValueError("No numeric data rows found after parsing.")

        # Build DataFrame from numeric_rows (each row has exactly 3 entries now)
        arr = np.array(numeric_rows, dtype=float)
        df = pd.DataFrame(arr)

        if df.shape[1] < 2:
            raise ValueError("File must have at least two numeric columns.")

        # Normalize header tokens for mapping if present
        def norm_label(s):
            if s is None:
                return ""
            s = str(s)
            s = s.strip().lower()
            s = re.sub(r"[/\s\-\_]+", "", s)
            s = re.sub(r"[^a-z0-9]+", "", s)
            return s

        # We constructed df so columns 0,1,2 correspond to energy, counts, error
        energy = pd.to_numeric(df.iloc[:, 0], errors="coerce").to_numpy(dtype=float)
        counts = pd.to_numeric(df.iloc[:, 1], errors="coerce").to_numpy(dtype=float)
        errors = pd.to_numeric(df.iloc[:, 2], errors="coerce").to_numpy(dtype=float)

        # If counts or energy contain NaNs, try to drop rows with NaNs
        valid_mask = np.isfinite(energy) & np.isfinite(counts)
        if errors is not None:
            valid_mask = valid_mask & np.isfinite(errors)
        energy = energy[valid_mask]
        counts = counts[valid_mask]
        if errors is not None:
            errors = errors[valid_mask]

        # If no errors column, estimate from counts
        if errors is None or len(errors) == 0:
            errors = np.sqrt(np.clip(np.abs(counts), 1e-12, np.inf))

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

    # Persist the folder and last-loaded file to configuration so future dialogs
    # open in the same directory and UI can remember what was last loaded.
    try:
        # import via package wrapper to avoid circular imports at module import time
        from dataio import get_config
        cfg = get_config()
        # store first selected file and its directory
        try:
            cfg.last_loaded_file = filepaths[0]
            cfg.default_load_folder = os.path.dirname(filepaths[0]) or cfg.default_load_folder
            cfg.save()
        except Exception:
            # be tolerant — don't block loading if config write fails
            pass
    except Exception:
        # ignore if config subsystem not available
        pass

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

