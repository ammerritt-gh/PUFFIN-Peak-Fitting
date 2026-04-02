import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

from dataio.data_loader import load_data_from_file


class LoadDataFromFileTests(unittest.TestCase):
    def test_loads_sample_testdata_file(self):
        repo_root = Path(__file__).resolve().parent.parent
        sample = repo_root / "testdata" / "H101_T75p0K_B0p0T_combined.dat"

        energy, counts, errors, info = load_data_from_file(str(sample))

        self.assertEqual(len(energy), 34)
        self.assertEqual(len(counts), 34)
        self.assertEqual(len(errors), 34)
        self.assertEqual(info["name"], sample.name)
        self.assertAlmostEqual(float(energy[0]), 3.0005, places=4)
        self.assertAlmostEqual(float(counts[-1]), 526.4, places=4)

    def test_infers_missing_errors_without_dropping_rows(self):
        content = "".join(
            [
                "Energy,Counts,Error\n",
                "meV,counts,err\n",
                "1.0,10.0\n",
                "2.0,20.0,4.0\n",
                "3.0,30.0,\n",
            ]
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            sample = Path(tmp_dir) / "mixed_errors.csv"
            sample.write_text(content, encoding="utf-8")

            energy, counts, errors, _ = load_data_from_file(str(sample))

        self.assertEqual(len(energy), 3)
        np.testing.assert_allclose(energy, np.array([1.0, 2.0, 3.0]))
        np.testing.assert_allclose(counts, np.array([10.0, 20.0, 30.0]))
        np.testing.assert_allclose(
            errors,
            np.array([math.sqrt(10.0), 4.0, math.sqrt(30.0)]),
            rtol=1e-6,
        )


if __name__ == "__main__":
    unittest.main()
