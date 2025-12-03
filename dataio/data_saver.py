# io/data_saver.py
import os
import numpy as np
import pandas as pd
from PySide6.QtWidgets import QFileDialog, QMessageBox


def save_dataset(x, y, y_fit=None, parent=None):
    """Prompt user to save a dataset (optionally with fit)."""
    save_path, _ = QFileDialog.getSaveFileName(
        parent, "Save Data As", os.path.expanduser("~"), "Data Files (*.txt *.csv)"
    )
    if not save_path:
        return

    df = pd.DataFrame({"Energy": x, "Counts": y})
    if y_fit is not None:
        df["Fit"] = y_fit
    df.to_csv(save_path, index=False, sep="\t")

    if parent:
        QMessageBox.information(parent, "Saved", f"Data saved to:\n{save_path}")
    else:
        print(f"[INFO] Saved to {save_path}")

