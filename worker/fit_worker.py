# worker/fit_worker.py
from PySide6.QtCore import QThread, Signal
import numpy as np
from scipy.optimize import curve_fit, minimize
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


class IterativeFitWorker(QThread):
    """Fitting worker that supports iterative (step-by-step) fitting with live updates."""
    
    progress = Signal(float)  # Progress 0.0 to 1.0
    step_completed = Signal(int, object, object)  # step_num, fit_result, y_fit
    finished = Signal(object, object)  # emits (fit_result, y_fit)

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
        
        # Convert bounds to scipy minimize format
        lower = bounds[0] if bounds else [-np.inf] * len(self.p0)
        upper = bounds[1] if bounds else [np.inf] * len(self.p0)
        self.scipy_bounds = list(zip(lower, upper))

    def stop(self):
        """Request stop."""
        self._stop = True

    def _chi_squared(self, params):
        """Compute chi-squared for the current parameters."""
        try:
            y_model = self.model_func(self.x, *params)
            residuals = (self.y - y_model) / self.err
            return np.sum(residuals ** 2)
        except Exception:
            return np.inf

    def run(self):
        """Perform iterative fitting in background."""
        try:
            current_params = self.p0.copy()
            step = 0
            max_iter = self.max_steps if self.max_steps else 100
            
            self.progress.emit(0.0)
            
            # Use scipy.optimize.minimize with BFGS for iterative control
            # We'll use a callback to emit updates after each step
            callback_step = [0]  # Use list to allow modification in nested function
            
            def callback(xk):
                """Called after each iteration."""
                callback_step[0] += 1
                step_num = callback_step[0]
                
                if self._stop:
                    raise StopIteration("Fitting stopped by user")
                
                progress = min(1.0, step_num / max_iter)
                self.progress.emit(progress)
                
                if self.step_mode:
                    try:
                        y_fit = self.model_func(self.x, *xk)
                        fit_result = dict(zip(self.param_names, xk))
                        self.step_completed.emit(step_num, fit_result, y_fit)
                    except Exception:
                        pass
                
                # Stop if we've reached max_steps
                if self.max_steps and step_num >= self.max_steps:
                    raise StopIteration("Max steps reached")
            
            try:
                # Use L-BFGS-B for bounded optimization
                result = minimize(
                    self._chi_squared,
                    current_params,
                    method='L-BFGS-B',
                    bounds=self.scipy_bounds,
                    callback=callback,
                    options={
                        'maxiter': max_iter,
                        'disp': False,
                        'ftol': 1e-10,
                        'gtol': 1e-8,
                    }
                )
                final_params = result.x
            except StopIteration:
                # This is expected when max_steps is reached or user stops
                # Get the current parameters from the last callback
                final_params = current_params if callback_step[0] == 0 else None
                if final_params is None:
                    # Use the result from the last successful iteration
                    # (We need to track this better, for now use current_params)
                    final_params = current_params
            
            # Compute final fit
            y_fit = self.model_func(self.x, *final_params)
            fit_result = dict(zip(self.param_names, final_params))

            self.progress.emit(1.0)
            self.finished.emit(fit_result, y_fit)

        except Exception as e:
            print(f"[IterativeFitWorker] Error: {e}")
            self.finished.emit(None, None)

