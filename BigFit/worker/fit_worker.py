# worker/fit_worker.py
from PySide6.QtCore import QThread, Signal
import numpy as np
from scipy.optimize import curve_fit
import time


class FitWorker(QThread):
    progress = Signal(float)
    finished = Signal(object, object)  # emits (fit_result, y_fit)

    def __init__(self, x, y, model_func, params, err=None, bounds=None):
        """
        Generic fitting worker.

        Parameters
        ----------
        x, y : array-like
            Input data arrays.
        model_func : callable
            Function to fit, must have signature f(x, *params).
        params : dict[str, float]
            Initial parameter values.
        err : array-like or None
            Error / sigma values (optional).
        bounds : tuple of (lower, upper)
            Bounds for curve_fit. Each a list/array matching params order.
        """
        super().__init__()
        self.x = np.asarray(x)
        self.y = np.asarray(y)
        self.err = np.asarray(err) if err is not None else np.ones_like(self.y)
        self.model_func = model_func
        self.param_names = list(params.keys())
        self.p0 = [params[k] for k in self.param_names]
        self.bounds = bounds or (
            [-np.inf] * len(self.p0),
            [np.inf] * len(self.p0)
        )
        self._stop = False

    def stop(self):
        """Request stop; curve_fit itself cannot be interrupted cleanly."""
        self._stop = True

    def run(self):
        """Perform fitting in background."""
        try:
            # Simulate progress for UX (curve_fit doesn't provide iteration hooks)
            self.progress.emit(0.05)

            popt, pcov = curve_fit(
                self.model_func,
                self.x,
                self.y,
                p0=self.p0,
                sigma=self.err,
                bounds=self.bounds,
                absolute_sigma=True,
                maxfev=8000
            )

            # Construct result
            y_fit = self.model_func(self.x, *popt)
            fit_result = dict(zip(self.param_names, popt))

            self.progress.emit(1.0)
            self.finished.emit(fit_result, y_fit)

        except Exception as e:
            print(f"[FitWorker] Error: {e}")
            self.finished.emit(None, None)
