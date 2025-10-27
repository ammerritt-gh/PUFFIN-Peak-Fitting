# model/data_loader.py
import os
import numpy as np
import pandas as pd
from PySide6.QtWidgets import QFileDialog, QMessageBox
import csv
import io
import re

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
        # Read a chunk of the file to detect delimiter and find where numeric data starts
        with open(filepath, "r", encoding="utf-8", errors="replace") as fh:
            raw_lines = fh.readlines()

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

        # Find first line that looks like numeric data (most tokens parse as float)
        def is_numeric_row(tokens):
            """Return True if a line looks like numeric data."""
            if len(tokens) < 2:
                return False
            num_ok = 0
            for t in tokens:
                s = t.strip().strip(",")
                # reject if contains alphabetic characters or '='
                if re.search(r"[A-Za-z=]", s):
                    return False
                try:
                    float(s)
                    num_ok += 1
                except Exception:
                    pass
            return num_ok >= 2 and num_ok >= 0.5 * len(tokens)

        # --- Find first numeric data row ---
        start_idx = None
        for i, line in enumerate(raw_lines):
            tokens = split_line(line)
            if is_numeric_row(tokens):
                start_idx = i
                break
        if start_idx is None:
            raise ValueError("No numeric data rows found in file.")

        # --- Skip unit rows just before numeric data ---
        if start_idx > 0:
            prev_line = split_line(raw_lines[start_idx - 1])
            if all(re.match(r"^[A-Za-zµμ]+(/[A-Za-z]+)?$", t) for t in prev_line if t):
                start_idx += 1

        # --- Improved header detection ---
        header_tokens = None
        best_header_score = -1
        for j in range(max(0, start_idx - 5), start_idx):
            candidate = raw_lines[j].strip()
            if candidate == "":
                continue
            toks = split_line(candidate)
            if not toks:
                continue
            norm = [re.sub(r"[^a-z0-9/]+", "", t.strip().lower()) for t in toks]
            letter_frac = sum(1 for t in toks if re.search(r"[A-Za-z]", t)) / len(toks)
            num_frac = sum(1 for t in toks if re.search(r"[0-9]", t)) / len(toks)
            key_matches = 0
            for t in norm:
                if any(k in t for k in energy_keywords | counts_keywords | error_keywords):
                    key_matches += 1
            score = key_matches * 2 + letter_frac - num_frac
            if score > best_header_score:
                best_header_score = score
                header_tokens = toks

        # Manually parse numeric rows to avoid pandas tokenization/deprecation issues.
        numeric_rows = []
        for line in raw_lines[start_idx:]:
            s = line.strip()
            if s == "" or s.lstrip().startswith("#"):
                continue
            toks = split_line(s)
            nums = []
            for t in toks:
                t2 = t.strip().strip(",")
                try:
                    nums.append(float(t2))
                except Exception:
                    # skip non-numeric tokens (e.g. trailing commas or stray text)
                    pass
            if len(nums) >= 2:
                numeric_rows.append(nums)

        if not numeric_rows:
            raise ValueError("No numeric data rows found after parsing.")

        # Make a rectangular array (pad shorter rows with NaN)
        max_cols = max(len(r) for r in numeric_rows)
        arr = np.full((len(numeric_rows), max_cols), np.nan, dtype=float)
        for i, r in enumerate(numeric_rows):
            arr[i, : len(r)] = r
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

        counts_keywords = {"counts", "count", "cts", "intensity", "y", "countsmin", "counts/min", "countspermin", "countsmin"}
        error_keywords = {"error", "err", "sigma", "unc", "uncertainty", "std"}

        col_labels = []
        if header_tokens:
            # ensure header_tokens length covers df columns if possible
            for i in range(df.shape[1]):
                if i < len(header_tokens):
                    col_labels.append(norm_label(header_tokens[i]))
                else:
                    col_labels.append("")
        else:
            col_labels = [""] * df.shape[1]

        # Map columns to energy, counts, error
        energy_col = None
        counts_col = None
        error_col = None
        for idx, lab in enumerate(col_labels):
            if any(k in lab for k in energy_keywords) and energy_col is None:
                energy_col = idx
            elif any(k in lab for k in counts_keywords) and counts_col is None:
                counts_col = idx
            elif any(k in lab for k in error_keywords) and error_col is None:
                error_col = idx

        # Fallback positional mapping
        if energy_col is None:
            energy_col = 0
        if counts_col is None:
            counts_col = 1 if df.shape[1] > 1 else 0
        if error_col is None and df.shape[1] > 2:
            error_col = 2

        # Extract columns safely (coerce to numeric)
        def col_to_array(index):
            if index is None or index >= df.shape[1]:
                return None
            return pd.to_numeric(df.iloc[:, index], errors="coerce").to_numpy(dtype=float)

        energy = col_to_array(energy_col)
        counts = col_to_array(counts_col)
        errors = col_to_array(error_col) if error_col is not None else None

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

