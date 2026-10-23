"""Check that a BL43LXU per-analyzer file loads as (meV, y_i, ye_i) with empty bins dropped.

Run from the repo root: .venv\\Scripts\\python.exe tests\\test_bl43_loader.py
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataio.data_loader import load_data_from_file  # noqa: E402

LABEL = ("#      meV   meV_sft   tmK_lin     lrn      secs        norm_lr         a03_lr       error_lr"
         "       y_unitpk          error            ytc           ytce    npts         norm_i            y_i           ye_i")
ROWS = [
    "-16.00 -22.09 -258.9 73 50.0 0.91750E+06 0.33547E-04 0.61218E-05 0.82059E-01 0.10411E-01 0.21529E-04 0.27316E-05 5 0.25117E+07 0.25349E-04 0.32163E-05",
    "-15.80 -21.89 -255.7 86 10.0 0.10811E+07 0.32272E-04 0.55314E-05 0.72444E-01 0.21843E-01 0.18690E-04 0.56352E-05 0 0.00000E+00 0.00000E+00 0.00000E+00",
    "-15.60 -21.69 -252.5 75 20.0 0.94353E+06 0.36934E-04 0.63341E-05 0.16149E+00 0.23070E-01 0.40961E-04 0.58516E-05 2 0.10068E+07 0.22379E-04 0.67475E-05",
]


def main():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "CsV3Sb5_302K_0p5_0p5_3_sum_a03")  # real files have no extension
        with open(path, "w", encoding="utf-8", newline="\n") as fh:
            fh.write("\n".join([LABEL, *ROWS, "#Q=   18.815 /nm", "#T=  302.000 K"]) + "\n")
        x, y, e, _ = load_data_from_file(path)
    assert list(x) == [-16.00, -15.60], x       # meV, not meV_sft; the npts == 0 row dropped
    assert list(y) == [0.25349e-04, 0.22379e-04], y
    assert list(e) == [0.32163e-05, 0.67475e-05], e
    print("PASS: BL43LXU analyzer file reads meV / y_i / ye_i and drops empty bins")


if __name__ == "__main__":
    main()
