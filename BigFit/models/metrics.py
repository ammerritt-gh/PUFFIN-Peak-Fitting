import numpy as np
from typing import Optional, Tuple, Dict


def compute_reduced_chi2(x: np.ndarray,
                         y: np.ndarray,
                         y_model: np.ndarray,
                         errs: Optional[np.ndarray] = None,
                         n_params: int = 0) -> Optional[float]:
    """Compute reduced chi-squared for given arrays.

    Returns None on invalid input.
    """
    try:
        if x is None or y is None or y_model is None:
            return None
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        m = np.asarray(y_model, dtype=float)
        if errs is None or len(errs) != len(y):
            sigma = np.ones_like(y)
        else:
            sigma = np.asarray(errs, dtype=float)
        if sigma.size != y.size or m.size != y.size:
            return None
        resid = y - m
        chi2 = float(np.sum((resid / sigma) ** 2))
        dof = max(1, x.size - max(0, int(n_params)))
        return chi2 / float(dof)
    except Exception:
        return None


def compute_cash_stat(x: np.ndarray,
                      y: np.ndarray,
                      y_model: np.ndarray,
                      n_params: int = 0) -> Tuple[Optional[float], Optional[float]]:
    """Return (C_full, C_reduced) or (None, None) on failure."""
    try:
        if x is None or y is None or y_model is None:
            return None, None
        d = np.asarray(y, dtype=float)
        m = np.asarray(y_model, dtype=float)
        if d.size != m.size:
            return None, None

        # valid bins: observed >=0, model > 0
        valid = (d >= 0) & (m > 0)
        if not np.any(valid):
            return None, None

        d = np.clip(d[valid], 1e-12, None)
        m = np.clip(m[valid], 1e-12, None)

        C = 2.0 * float(np.sum(m - d + d * np.log(d / m)))
        dof = max(1, int(x.size) - max(0, int(n_params)))
        C_red = C / float(dof) if dof > 0 else None
        return C, C_red
    except Exception:
        return None, None


def compute_statistics_from_state(state, y_fit: Optional[np.ndarray], n_params: int = 0) -> Dict[str, Optional[float]]:
    """Convenience: pull arrays from a ModelState-like object and compute stats.

    Returns dict with keys: 'reduced_chi2', 'cash', 'reduced_cash'
    """
    try:
        x = getattr(state, "x_data", None)
        y = getattr(state, "y_data", None)
        errs = getattr(state, "errors", None)
        # support an exclusion mask named 'excluded' or 'excluded_mask'
        exc = getattr(state, "excluded", None)
        if exc is None:
            exc = getattr(state, "excluded_mask", None)
        if exc is not None and isinstance(exc, (list, np.ndarray)) and y is not None and len(exc) == len(y):
            inc = ~np.asarray(exc, dtype=bool)
            x = np.asarray(x)[inc]
            y = np.asarray(y)[inc]
            # handle errs which may be full-length or already masked
            if errs is not None:
                errs_arr = np.asarray(errs)
                if errs_arr.size == len(inc):
                    errs = errs_arr[inc]
                elif errs_arr.size == int(np.sum(inc)):
                    errs = errs_arr
                else:
                    errs = None
            # handle y_fit which may be full-length or already masked
            if y_fit is not None:
                yfit_arr = np.asarray(y_fit)
                if yfit_arr.size == len(inc):
                    y_fit = yfit_arr[inc]
                elif yfit_arr.size == int(np.sum(inc)):
                    y_fit = yfit_arr
                else:
                    # incompatible length — ignore y_fit so stats won't be computed
                    y_fit = None

        stats = {"reduced_chi2": None, "cash": None, "reduced_cash": None}

        if y_fit is not None and x is not None and y is not None:
            stats["reduced_chi2"] = compute_reduced_chi2(x, y, y_fit, errs, n_params)
            c, c_red = compute_cash_stat(x, y, y_fit, n_params)
            stats["cash"] = c
            stats["reduced_cash"] = c_red

        return stats
    except Exception:
        return {"reduced_chi2": None, "cash": None, "reduced_cash": None}
