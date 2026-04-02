import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from dataio import fit_persistence
from models import ModelState


class FitPersistenceTests(unittest.TestCase):
    def test_load_rejects_mismatched_saved_path(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            temp_root = Path(tmp_dir)
            fits_dir = temp_root / "fits"
            fits_dir.mkdir()
            data_path = temp_root / "sample.dat"
            other_path = temp_root / "other.dat"
            data_path.write_text("1 10 1\n2 20 2\n", encoding="utf-8")
            other_path.write_text("1 10 1\n2 20 2\n", encoding="utf-8")

            state = ModelState()
            state.x_data = np.array([1.0, 2.0], dtype=float)
            state.y_data = np.array([10.0, 20.0], dtype=float)
            state.errors = np.array([1.0, 2.0], dtype=float)
            state.file_info = {"path": str(data_path), "name": data_path.name}

            with patch.object(fit_persistence, "_get_fits_folder", return_value=fits_dir), patch.object(
                fit_persistence, "_ensure_fits_folder", return_value=fits_dir
            ):
                self.assertTrue(fit_persistence.save_fit_for_file(state, str(data_path)))
                fit_path = fits_dir / fit_persistence._get_fit_filename_for_file(str(data_path))
                fit_data = json.loads(fit_path.read_text(encoding="utf-8"))
                fit_data["signature"]["path"] = str(other_path)
                fit_path.write_text(json.dumps(fit_data, indent=2), encoding="utf-8")

                restored = ModelState()
                restored.set_data(np.array([1.0, 2.0], dtype=float), np.array([10.0, 20.0], dtype=float))
                success, _ = fit_persistence.load_fit_for_file(restored, str(data_path), apply_excluded=True)

            self.assertFalse(success)

    def test_apply_fit_state_skips_excluded_mask_when_lengths_differ(self):
        state = ModelState()
        state.set_data(np.array([1.0, 2.0, 3.0], dtype=float), np.array([10.0, 20.0, 30.0], dtype=float))
        state.excluded = np.array([False, False, False], dtype=bool)

        fit_data = fit_persistence._extract_fit_state(state)
        self.assertIsNotNone(fit_data)
        assert fit_data is not None
        fit_data["excluded"] = [True, False]

        with self.assertLogs(fit_persistence.logger, level="WARNING") as captured:
            success = fit_persistence._apply_fit_state(state, fit_data, apply_excluded=True)

        self.assertTrue(success)
        np.testing.assert_array_equal(state.excluded, np.array([False, False, False], dtype=bool))
        self.assertTrue(any("Skipped excluded-mask restore" in line for line in captured.output))


if __name__ == "__main__":
    unittest.main()
