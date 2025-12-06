# worker/fit_worker.py
from PySide6.QtCore import QThread, Signal
import numpy as np
from scipy.optimize import curve_fit, least_squares


class FitWorker(QThread):
    progress = Signal(float)
    finished = Signal(object, object)  # emits (fit_result, y_fit)
    error_occurred = Signal(str)  # emits error message for user feedback

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
            error_msg = str(e)
            print(f"[FitWorker] Error: {error_msg}")
            self.error_occurred.emit(error_msg)
            self.finished.emit(None, None)


class IterativeFitWorker(QThread):
    """Fitting worker that supports iterative (step-by-step) fitting with live updates.
    
    Uses least_squares with limited max_nfev for true step-by-step control where 
    each "step" is approximately one iteration of the trust-region algorithm.
    """
    
    progress = Signal(float)  # Progress 0.0 to 1.0
    step_completed = Signal(int, object, object, float)  # step_num, fit_result, y_fit, chi_squared
    finished = Signal(object, object)  # emits (fit_result, y_fit)
    error_occurred = Signal(str)  # emits error message for user feedback

    # Convergence threshold for early stopping
    OPTIMALITY_THRESHOLD = 1e-8
    
    # Maximum number of function evaluations per step
    # 5 gives roughly one trust-region iteration for most problems
    EVALS_PER_STEP = 5
    
    # Tolerance for chi^2 comparison when rejecting worse fits
    # Steps where chi^2 increases by more than this fraction are rejected
    CHI2_TOLERANCE = 0.001  # 0.1%

    def __init__(self, x, y, model_func, params, err=None, bounds=None, 
                 max_steps=None, step_mode=False, reject_worse_chi2=True):
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
        reject_worse_chi2 : bool
            If True, reject fit steps that increase chi-squared value.
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
        self.reject_worse_chi2 = reject_worse_chi2
        self._stop = False
        
        # Store bounds in least_squares format (lower, upper)
        self.bounds = bounds or (
            [-np.inf] * len(self.p0),
            [np.inf] * len(self.p0)
        )

    def stop(self):
        """Request stop."""
        self._stop = True

    def _check_bounds_feasibility(self, params):
        """Check if initial parameters are within bounds.
        
        Returns:
            tuple: (is_feasible, error_message)
        """
        lower, upper = self.bounds
        for p, lo, hi, name in zip(params, lower, upper, self.param_names):
            if p < lo:
                return False, f"Parameter '{name}' value {p:.4g} is below minimum bound {lo:.4g}"
            if p > hi:
                return False, f"Parameter '{name}' value {p:.4g} is above maximum bound {hi:.4g}"
        return True, ""

    def _residuals(self, params):
        """Compute residuals for least_squares optimization."""
        try:
            y_model = self.model_func(self.x, *params)
            return (self.y - y_model) / self.err
        except Exception:
            return np.full_like(self.y, np.inf)

    def _compute_chi_squared(self, params):
        """Compute chi-squared value for given parameters."""
        try:
            residuals = self._residuals(params)
            return np.sum(residuals ** 2)
        except Exception:
            return np.inf

    def run(self):
        """Perform iterative fitting in background.
        
        Each "step" runs least_squares with max_nfev=EVALS_PER_STEP, which means
        minimal progress per step for fine-grained control.
        """
        try:
            current_params = self.p0.copy()
            max_iter = self.max_steps if self.max_steps else 100
            
            # Check bounds feasibility first
            is_feasible, error_msg = self._check_bounds_feasibility(current_params)
            if not is_feasible:
                self.error_occurred.emit(f"Initial parameters infeasible: {error_msg}")
                self.finished.emit(None, None)
                return
            
            self.progress.emit(0.0)
            
            # Compute initial chi-squared
            best_chi2 = self._compute_chi_squared(current_params)
            best_params = current_params.copy()
            
            # Run least_squares iterations, each with minimal function evaluations
            for step in range(max_iter):
                if self._stop:
                    break
                
                try:
                    # Use least_squares with very limited max_nfev for true single iteration
                    result = least_squares(
                        self._residuals,
                        current_params,
                        bounds=self.bounds,
                        max_nfev=self.EVALS_PER_STEP,
                        verbose=0
                    )
                    
                    new_params = result.x
                    new_chi2 = self._compute_chi_squared(new_params)
                    
                    # Check if fit improved or if we should reject worse fits
                    if self.reject_worse_chi2 and new_chi2 > best_chi2 * (1.0 + self.CHI2_TOLERANCE):
                        # Reject this step - chi^2 got worse
                        # Keep using best params silently (this is normal behavior)
                        current_params = best_params.copy()
                    else:
                        # Accept this step
                        current_params = new_params
                        if new_chi2 < best_chi2:
                            best_chi2 = new_chi2
                            best_params = new_params.copy()
                    
                except ValueError as e:
                    # Handle infeasible bounds gracefully
                    error_msg = str(e)
                    if "infeasible" in error_msg.lower() or "x0" in error_msg.lower():
                        self.error_occurred.emit(f"Bounds error: {error_msg}. Check that parameter values are within min/max bounds.")
                    else:
                        self.error_occurred.emit(f"Fitting error: {error_msg}")
                    self.finished.emit(None, None)
                    return
                
                step_num = step + 1
                progress = min(1.0, step_num / max_iter)
                self.progress.emit(progress)
                
                if self.step_mode:
                    try:
                        y_fit = self.model_func(self.x, *current_params)
                        fit_result = dict(zip(self.param_names, current_params))
                        current_chi2 = self._compute_chi_squared(current_params)
                        self.step_completed.emit(step_num, fit_result, y_fit, current_chi2)
                    except Exception:
                        pass
                
                # Check if converged early (optimality < threshold)
                if hasattr(result, 'optimality') and result.optimality < self.OPTIMALITY_THRESHOLD:
                    break
            
            # Use best parameters found
            final_params = best_params
            
            # Compute final fit
            y_fit = self.model_func(self.x, *final_params)
            fit_result = dict(zip(self.param_names, final_params))

            self.progress.emit(1.0)
            self.finished.emit(fit_result, y_fit)

        except ValueError as e:
            error_msg = str(e)
            if "infeasible" in error_msg.lower() or "x0" in error_msg.lower():
                self.error_occurred.emit(f"Bounds error: {error_msg}. Check that parameter values are within min/max bounds.")
            else:
                self.error_occurred.emit(f"Fitting error: {error_msg}")
            self.finished.emit(None, None)
        except Exception as e:
            error_msg = str(e)
            print(f"[IterativeFitWorker] Error: {error_msg}")
            self.error_occurred.emit(f"Fitting error: {error_msg}")
            self.finished.emit(None, None)

