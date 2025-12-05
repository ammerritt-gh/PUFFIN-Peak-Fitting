# worker/fit_worker.py
from PySide6.QtCore import QThread, Signal
import numpy as np
from scipy.optimize import curve_fit, least_squares


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


class IterativeFitWorker(QThread):
    """Fitting worker that supports iterative (step-by-step) fitting with live updates.
    
    Uses least_squares with limited max_nfev for predictable step-by-step behavior.
    """
    
    progress = Signal(float)  # Progress 0.0 to 1.0
    step_completed = Signal(int, object, object)  # step_num, fit_result, y_fit
    finished = Signal(object, object)  # emits (fit_result, y_fit)

    # Number of function evaluations per "step" - this controls how much work
    # is done per iteration. Higher values = more progress per step.
    EVALS_PER_STEP = 30

    def __init__(self, x, y, model_func, params, err=None, bounds=None, 
                 max_steps=None, step_mode=False):
        """
        Iterative fitting worker with live updates.

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
        max_steps : int or None
            Maximum number of iterations. None for fit to completion.
        step_mode : bool
            If True, emit step_completed after each iteration for live preview.
        """
        super().__init__()
        self.x = np.asarray(x)
        self.y = np.asarray(y)
        self.err = np.asarray(err) if err is not None else np.ones_like(self.y)
        self.model_func = model_func
        self.param_names = list(params.keys())
        self.p0 = np.array([params[k] for k in self.param_names], dtype=float)
        self.max_steps = max_steps
        self.step_mode = step_mode
        self._stop = False
        
        # Store bounds in least_squares format (lower, upper)
        self.bounds = bounds or (
            [-np.inf] * len(self.p0),
            [np.inf] * len(self.p0)
        )

    def stop(self):
        """Request stop."""
        self._stop = True

    def _residuals(self, params):
        """Compute residuals for least_squares optimization."""
        try:
            y_model = self.model_func(self.x, *params)
            return (self.y - y_model) / self.err
        except Exception:
            return np.full_like(self.y, np.inf)

    def run(self):
        """Perform iterative fitting in background using least_squares with limited max_nfev."""
        try:
            current_params = self.p0.copy()
            max_iter = self.max_steps if self.max_steps else 100
            
            self.progress.emit(0.0)
            
            # Run least_squares iterations, each with limited function evaluations
            for step in range(max_iter):
                if self._stop:
                    break
                
                # Use least_squares with limited max_nfev for each step
                # This gives predictable step-by-step improvement and always returns
                # the best parameters found so far
                result = least_squares(
                    self._residuals,
                    current_params,
                    bounds=self.bounds,
                    max_nfev=self.EVALS_PER_STEP,
                    verbose=0
                )
                
                # Update current_params with the result (always makes progress)
                current_params = result.x
                
                step_num = step + 1
                progress = min(1.0, step_num / max_iter)
                self.progress.emit(progress)
                
                if self.step_mode:
                    try:
                        y_fit = self.model_func(self.x, *current_params)
                        fit_result = dict(zip(self.param_names, current_params))
                        self.step_completed.emit(step_num, fit_result, y_fit)
                    except Exception:
                        pass
                
                # Check if converged early
                if result.success and result.optimality < 1e-8:
                    break
            
            # Compute final fit
            y_fit = self.model_func(self.x, *current_params)
            fit_result = dict(zip(self.param_names, current_params))

            self.progress.emit(1.0)
            self.finished.emit(fit_result, y_fit)

        except Exception as e:
            print(f"[IterativeFitWorker] Error: {e}")
            self.finished.emit(None, None)

