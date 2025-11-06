import numpy as np
from scipy.special import wofz, voigt_profile
from scipy.fft import fft, ifft
from scipy.interpolate import interp1d
from scipy.signal import fftconvolve
from scipy.interpolate import interp1d

import matplotlib.pyplot as plt

# Constants
kB = 0.086173324  # meV/K

# This Gaussian is set to use a FWHM as the width instead of sigma and integrates to area
def Gaussian(x, Area, Width, Center):
    return Area * np.sqrt(4 * np.log(2) / np.pi) / Width * np.exp(-4 * np.log(2) * (np.array(x) - Center) ** 2 / Width ** 2)

# This Lorentzian integrates to area
def Lorentzian(x, Area, Width, Center):
    return 2 * Area / np.pi * Width / (4 * (np.array(x) - Center) ** 2 + Width ** 2)

# Integrates to area
def Voigt(x, Area, gauss_fwhm, lorentz_fwhm, center):
    sigma = gauss_fwhm / (2 * np.sqrt(2 * np.log(2)))  # Convert FWHM to std
    gamma = lorentz_fwhm / 2                           # Convert FWHM to HWHM
    z = ((x - center) + 1j * gamma) / (sigma * np.sqrt(2))
    profile = np.real(wofz(z)) / (sigma * np.sqrt(2 * np.pi))
    return Area * profile

def Voigt_wBG(x, center, gauss_fwhm, lorentz_fwhm, amplitude, BG):
    return Voigt(x, amplitude, gauss_fwhm, lorentz_fwhm, center) + BG

def voigt_area_for_height(height, gauss_fwhm, lorentz_fwhm):
    """
    Given a desired peak height and FWHMs, compute the area scaling
    to produce a Voigt profile with that height.
    """
    sigma = gauss_fwhm / (2 * np.sqrt(2 * np.log(2)))  # Gaussian std dev
    gamma = lorentz_fwhm / 2                           # Lorentzian HWHM

    z = (1j * gamma) / (sigma * np.sqrt(2))
    voigt_peak = np.real(wofz(z)) / (sigma * np.sqrt(2 * np.pi))

    area = height / voigt_peak
    return area

def voigt_height_for_area(area, gauss_fwhm, lorentz_fwhm):
    sigma = gauss_fwhm / (2 * np.sqrt(2 * np.log(2)))
    gamma = lorentz_fwhm / 2
    z = (1j * gamma) / (sigma * np.sqrt(2))
    voigt_peak = np.real(wofz(z)) / (sigma * np.sqrt(2 * np.pi))
    height = area * voigt_peak
    return height

def dho_voigt_area_for_height(target_height, phonon_energy, gauss_fwhm, lorentz_fwhm, damping, T, center=0.0):
    """
    Numerically estimate the area needed for a DHO convolved with a Voigt
    to reach a given peak height.
    """
    test_area = 1.0
    x = np.linspace(center + phonon_energy - 5, center + phonon_energy + 5, 1000)
    y = convolute_voigt_dho(
        x, phonon_energy, center, gauss_fwhm, lorentz_fwhm,
        damping, test_area, 0, 0, T, peak='stokes'
    )
    max_y = np.max(y)
    area = test_area * (target_height / max_y) if max_y > 0 else 0
    return area

def dho_voigt_height_for_area(area, phonon_energy, gauss_fwhm, lorentz_fwhm, damping, T, center=0.0):
    test_area = 1.0
    x = np.linspace(center + phonon_energy - 5, center + phonon_energy + 5, 1000)
    y = convolute_voigt_dho(
        x, phonon_energy, center, gauss_fwhm, lorentz_fwhm,
        damping, test_area, 0, 0, T, peak='stokes'
    )
    max_y = np.max(y)
    height = area * (max_y / test_area) if test_area > 0 else 0
    return height

def Stokes_DHO(x, amplitude, damping, center):
    damping_sq = center**2 - damping**2
    if damping_sq <= 0:
        # Soft penalty to avoid non-smooth behavior
        penalty = 1e6 * (1 + np.abs(damping_sq))
        return np.full_like(x, penalty)
    return amplitude / ((x - np.sqrt(center**2 - damping**2))**2 + damping**2)

def antiStokes_DHO(x, amplitude, damping, center):
    damping_sq = center**2 - damping**2
    if damping_sq <= 0:
        # Soft penalty to avoid non-smooth behavior
        penalty = 1e6 * (1 + np.abs(damping_sq))
        return np.full_like(x, penalty)
    return amplitude / ((x + np.sqrt(center**2 - damping**2))**2 + damping**2)

def weighted_discrete_delta(x, center, amplitude):
    delta = np.zeros_like(x)
    idx_below = np.searchsorted(x, center) - 1
    idx_above = idx_below + 1

    if idx_below < 0 or idx_above >= len(x):
        return delta  # out of bounds

    x0, x1 = x[idx_below], x[idx_above]
    w_above = (center - x0) / (x1 - x0)
    w_below = 1.0 - w_above

    delta[idx_below] = amplitude * w_below
    delta[idx_above] = amplitude * w_above
    return delta

def narrow_gaussian_delta(x, center, amplitude, fwhm=0.05):
    return Gaussian(x, Area=amplitude, Width=fwhm, Center=center)

def convolute_voigt_dho(x_target, phonon_energy, center, gauss_fwhm, lorentz_fwhm,
                        damping, phonon_amplitude, elastic_amplitude, BG, T, peak='all'):
    """
    Computes Voigt-convolved DHO + elastic Voigt peak + BG, interpolated to x_target.
    The elastic peak is added as a Voigt directly (not convolved).
    """
    dx = 0.05
    pad_range = 20

    # Convolution grid centered at 0
    x_min, x_max = np.min(x_target), np.max(x_target)
    x_span = max(abs(x_min), abs(x_max)) + pad_range
    x_uniform = np.arange(-x_span, x_span, dx)

    # Build the signal: DHO (Stokes, anti-Stokes) only
    signal = np.zeros_like(x_uniform)
    peak = peak.lower()
    if peak == 'all':
        peak_parts = ['stokes', 'astokes', 'elastic']
    else:
        peak_parts = peak.split('+')

    if 'stokes' in peak_parts:
        signal += Stokes_DHO(x_uniform, amplitude=phonon_amplitude,
                             damping=damping, center=phonon_energy)

    if 'astokes' in peak_parts:
        astokes_amp = phonon_amplitude / np.exp(phonon_energy / (kB * T))
        signal += antiStokes_DHO(x_uniform, amplitude=astokes_amp,
                                 damping=damping, center=phonon_energy)

    # Voigt resolution function, centered at 0
    voigt = Voigt(x_uniform, Area=1.0, gauss_fwhm=gauss_fwhm, lorentz_fwhm=lorentz_fwhm, center=0)
    convolved = fftconvolve(signal, voigt, mode='same') * dx

    # Interpolate to x_target - center
    interp = interp1d(x_uniform, convolved, kind='linear', bounds_error=False, fill_value=0.0)
    y = interp(x_target - center)

    # Add elastic peak as a Voigt directly at 'center'
    if 'elastic' in peak_parts:
        y += Voigt(x_target, Area=elastic_amplitude, gauss_fwhm=gauss_fwhm, lorentz_fwhm=lorentz_fwhm, center=center)

    # Add background
    y += BG

    return y

def convolute_gaussian_dho(x_target, phonon_energy, center, gauss_fwhm, lorentz_fwhm,
                           damping, phonon_amplitude, elastic_amplitude, BG, T, peak='all'):
    """
    Computes Gaussian-convolved DHO + elastic Gaussian peak + BG, interpolated to x_target.
    The elastic peak is added as a Gaussian directly (not convolved).
    """
    dx = 0.05
    pad_range = 20

    # Convolution grid centered at 0
    x_min, x_max = np.min(x_target), np.max(x_target)
    x_span = max(abs(x_min), abs(x_max)) + pad_range
    x_uniform = np.arange(-x_span, x_span, dx)

    # Build the signal: DHO (Stokes, anti-Stokes) only
    signal = np.zeros_like(x_uniform)
    peak = peak.lower()
    if peak == 'all':
        peak_parts = ['stokes', 'astokes', 'elastic']
    else:
        peak_parts = peak.split('+')

    if 'stokes' in peak_parts:
        signal += Stokes_DHO(x_uniform, amplitude=phonon_amplitude,
                             damping=damping, center=phonon_energy)

    if 'astokes' in peak_parts:
        astokes_amp = phonon_amplitude / np.exp(phonon_energy / (kB * T))
        signal += antiStokes_DHO(x_uniform, amplitude=astokes_amp,
                                 damping=damping, center=phonon_energy)

    # Gaussian resolution function, centered at 0 (use gauss_fwhm as width)
    gaussian = Gaussian(x_uniform, Area=1.0, Width=gauss_fwhm, Center=0)
    convolved = fftconvolve(signal, gaussian, mode='same') * dx

    # Interpolate to x_target - center
    interp = interp1d(x_uniform, convolved, kind='linear', bounds_error=False, fill_value=0.0)
    y = interp(x_target - center)

    # Add elastic peak as a Gaussian directly at 'center'
    if 'elastic' in peak_parts:
        y += Gaussian(x_target, Area=elastic_amplitude, Width=gauss_fwhm, Center=center)

    # Add background
    y += BG

    return y

from typing import Any, Dict, Optional, List
import copy

# Allowed input action keys that model parameter specs may provide.
# Models should use these keys (case-sensitive): "drag", "wheel", "hotkey".
# - "drag": means left-click + drag to update a parameter (value usually comes from x or y)
# - "wheel": means mouse-wheel; action dict may include "modifiers": ["Ctrl","Shift","Alt"]
# - "hotkey": means a keyboard shortcut; action dict should specify key name and action.
ALLOWED_INPUT_ACTIONS = ["drag", "wheel", "hotkey"]

class Parameter:
    """Represents a single parameter with metadata usable by the view.

    Fields:
      name       : parameter identifier (string)
      value      : initial/default value for the parameter
      ptype      : recommended type for the UI: "float", "int", "str", "bool", "choice"
      minimum    : numeric lower bound (optional)
      maximum    : numeric upper bound (optional)
      choices    : list of allowed values for 'choice' type (optional)
      hint       : short help text/tooltip for the UI (optional)
      decimals   : for float-type controls — number of decimal places to display
                   (maps to QDoubleSpinBox.setDecimals). Not a rounding guarantee,
                   only a display/precision hint for the widget.
      step       : single-step increment for numeric controls (maps to
                   QDoubleSpinBox.setSingleStep / QSpinBox.setSingleStep).
      input_hint : Optional[Any]
         - If None: no interactive input advertised.
         - If a string: treated as a human-readable hint only.
         - If a dict: structured interactive spec. Top-level keys should be in ALLOWED_INPUT_ACTIONS.
           Examples:
             { "drag":  { "action": "set", "value_from": "x" } }
             { "wheel": { "modifiers": ["Ctrl"], "action": "scale", "factor": 1.05 } }
             { "hotkey": { "key": "F", "action": "trigger_fit" } }
         The view and ViewModel agree on the action semantics: "set" sets parameter to coordinate,
         "scale" multiplies numeric parameter by factor, "increment" adds a step, etc.
    """
    def __init__(self,
                 name: str,
                 value: Any = None,
                 ptype: Optional[str] = None,
                 minimum: Optional[float] = None,
                 maximum: Optional[float] = None,
                 choices: Optional[List[Any]] = None,
                 hint: str = "",
                 decimals: Optional[int] = None,
                 step: Optional[float] = None,
                 input_hint: Optional[Any] = None):   # <-- note: can be str or structured dict
        self.name = name
        self.value = value
        # ptype recommended to be one of: "float","int","str","bool","choice"
        self.ptype = ptype
        # numeric bounds (or None)
        self.min = minimum
        self.max = maximum
        # choices for 'choice' type
        self.choices = choices
        # short help text
        self.hint = hint
        # number of decimal places to display when using a QDoubleSpinBox
        self.decimals = decimals
        # single-step increment (float for QDoubleSpinBox, int for QSpinBox)
        self.step = step
        # short text describing interactive input (hotkeys/wheel/mouse)
        self.input_hint = input_hint

    def to_spec(self) -> Dict[str, Any]:
        """Export the parameter as a spec dict the view expects.

        The returned dict contains at minimum:
          { "value": <value> }

        Optional keys (when present) include:
          "type"     -> "float"|"int"|"str"|"bool"|"choice"
          "min"      -> numeric minimum
          "max"      -> numeric maximum
          "choices"  -> list of allowed values (for type "choice")
          "hint"     -> help text for UI/tooltip
          "decimals" -> integer number of decimal places for float widgets
          "step"     -> numeric step increment for spin widgets
          "input"    -> short text describing interactive input (hotkeys/wheel/mouse)
        """
        spec: Dict[str, Any] = {"value": self.value}
        if self.ptype:
            spec["type"] = self.ptype
        if self.min is not None:
            spec["min"] = self.min
        if self.max is not None:
            spec["max"] = self.max
        if self.choices is not None:
            spec["choices"] = list(self.choices)
        if self.hint:
            spec["hint"] = self.hint
        if self.decimals is not None:
            # decimals is the number of decimal places the UI should display.
            spec["decimals"] = int(self.decimals)
        if self.step is not None:
            # step is the widget increment (single step).
            spec["step"] = float(self.step)
        if self.input_hint is not None:
            # pass structured input metadata through under 'input' key.
            # may be a str (hint) or a dict describing events/actions.
            spec["input"] = self.input_hint
        return spec

class BaseModelSpec:
    """Base container for model parameter specs and initialisation."""
    def __init__(self):
        self.params: Dict[str, Parameter] = {}
        # Flag to indicate whether this model is a constructed element
        # (a starting point that should not be addable to other elements).
        # Default: False (basic element)
        self.is_constructed: bool = False

    def add(self, param: Parameter):
        self.params[param.name] = param

    def get_parameters(self) -> Dict[str, Any]:
        """Return a dict suitable for the view: name -> spec-dict.

        Each value in the returned dict is typically the dict produced by
        Parameter.to_spec(). Example:
          {
            "gauss_fwhm": { "value": 1.14, "type": "float", "min": 0.0, "decimals": 6, "step": 0.01 },
            "mode":       { "value": "Voigt", "type": "choice", "choices": ["Voigt","Gaussian"] }
          }
        The view will infer widget types from the "type" key (or from the value).
        """
        return {name: p.to_spec() for name, p in self.params.items()}

    def initialize(self, data_x=None, data_y=None):
        """Hook for model-specific initialization (e.g., estimate from data)."""
        # default no-op
        return

    def get_param_values(self, overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Return a plain dict of parameter name -> simple value.
        If overrides provided, they take precedence."""
        out = {}
        for name, p in self.params.items():
            try:
                base = getattr(p, "value", None)
            except Exception:
                base = None
            out[name] = base
        if overrides:
            for k, v in overrides.items():
                try:
                    out[k] = v
                except Exception:
                    pass
        return out

    def evaluate(self, x, params: Optional[Dict[str, Any]] = None):
        """Default evaluation: return zeros. Subclasses may accept optional params dict."""
        try:
            return np.zeros_like(np.asarray(x, dtype=float))
        except Exception:
            return np.array([])

# Concrete model specs
class GaussianModelSpec(BaseModelSpec):
    def __init__(self):
        super().__init__()
        self.add(Parameter("Area", value=1.0, ptype="float", minimum=0.0,
                           hint="Integrated area of the Gaussian peak", decimals=6, step=0.1,
                           input_hint={"wheel": {"modifiers": ["Ctrl"], "action": "scale", "factor": 1.1}}))
        self.add(Parameter("Width", value=1.0, ptype="float", minimum=1e-6,
                           hint="Gaussian FWHM", decimals=6, step=0.01))
        self.add(Parameter("Center", value=0.0, ptype="float",
                           hint="Peak center (x-axis)",
                           input_hint={"drag": {"action": "set", "value_from": "x"}}))

    def evaluate(self, x, params: Optional[Dict[str, Any]] = None):
        try:
            pvals = self.get_param_values(params)
            Area = float(pvals.get("Area", 1.0))
            Width = float(pvals.get("Width", 0.1))
            Center = float(pvals.get("Center", 0.0))
            return Gaussian(np.asarray(x, dtype=float), Area=Area, Width=Width, Center=Center)
        except Exception:
            return super().evaluate(x, params)


class LorentzModelSpec(BaseModelSpec):
    def __init__(self):
        super().__init__()
        self.add(Parameter("Area", value=1.0, ptype="float", minimum=0.0,
                           hint="Integrated area of the Lorentzian peak", decimals=6, step=0.1,
                           input_hint={"wheel": {"modifiers": ["Ctrl"], "action": "scale", "factor": 1.1}}))
        self.add(Parameter("Width", value=1.0, ptype="float", minimum=1e-6,
                           hint="Lorentzian FWHM", decimals=6, step=0.01))
        self.add(Parameter("Center", value=0.0, ptype="float",
                           hint="Peak center (x-axis)",
                           input_hint={"drag": {"action": "set", "value_from": "x"}}))

    def evaluate(self, x, params: Optional[Dict[str, Any]] = None):
        try:
            pvals = self.get_param_values(params)
            Area = float(pvals.get("Area", 1.0))
            Width = float(pvals.get("Width", 0.1))
            Center = float(pvals.get("Center", 0.0))
            return Lorentzian(np.asarray(x, dtype=float), Area=Area, Width=Width, Center=Center)
        except Exception:
            return super().evaluate(x, params)

class VoigtModelSpec(BaseModelSpec):
    def __init__(self):
        super().__init__()
        self.add(Parameter("Area", value=1.0, ptype="float", minimum=0.0,
                           hint="Integrated area of the Voigt peak", decimals=6, step=0.1,
                           input_hint={"wheel": {"modifiers": ["Ctrl"], "action": "scale", "factor": 1.1}}))
        self.add(Parameter("gauss_fwhm", value=1.14, ptype="float", minimum=0.0,
                           hint="Gaussian resolution FWHM", decimals=6, step=0.01,
                           input_hint={"wheel": {"modifiers": ["Ctrl"], "action": "scale", "factor": 1.05}}))
        self.add(Parameter("lorentz_fwhm", value=0.28, ptype="float", minimum=0.0,
                           hint="Lorentzian FWHM (HWHM*2)", decimals=6, step=0.01,
                           input_hint={"wheel": {"modifiers": ["Shift"], "action": "scale", "factor": 1.05}}))
        # Use "drag" (left-click + drag) instead of simple click so clicks remain available for other UI parts.
        self.add(Parameter("center", value=0.0, ptype="float",
                           hint="Peak center",
                           input_hint={"drag": {"action": "set", "value_from": "x"}}))

    def initialize(self, data_x=None, data_y=None):
        # simple example: set center to x of max if data provided
        try:
            if data_x is not None and data_y is not None:
                arrx = np.asarray(data_x)
                arry = np.asarray(data_y)
                idx = int(np.nanargmax(arry))
                self.params["center"].value = float(arrx[idx])
        except Exception:
            pass

    def evaluate(self, x, params: Optional[Dict[str, Any]] = None):
        try:
            pvals = self.get_param_values(params)
            Area = float(pvals.get("Area", 1.0))
            gauss_fwhm = float(pvals.get("gauss_fwhm", 1.14))
            lorentz_fwhm = float(pvals.get("lorentz_fwhm", 0.28))
            center = float(pvals.get("center", 0.0))
            return Voigt(np.asarray(x, dtype=float), Area=Area, gauss_fwhm=gauss_fwhm, lorentz_fwhm=lorentz_fwhm, center=center)
        except Exception:
            return super().evaluate(x, params)

class DHOModelSpec(BaseModelSpec):
    def __init__(self):
        super().__init__()
        self.add(Parameter("phonon_energy", value=5.0, ptype="float", minimum=0.0,
                           hint="Phonon energy (meV)", decimals=4, step=0.1,
                           input_hint={"wheel": {"modifiers": ["Ctrl"], "action": "scale", "factor": 1.05}}))
        self.add(Parameter("damping", value=0.5, ptype="float", minimum=1e-6,
                           hint="DHO damping parameter", decimals=4, step=0.01,
                           input_hint={"wheel": {"modifiers": ["Shift"], "action": "scale", "factor": 1.05}}))
        self.add(Parameter("phonon_amplitude", value=0.1, ptype="float", minimum=0.0,
                           hint="Phonon amplitude (area-like)", decimals=6, step=0.01,
                           input_hint={"wheel": {"modifiers": ["Ctrl"], "action": "scale", "factor": 1.05}}))
        self.add(Parameter("elastic_amplitude", value=0.0, ptype="float", minimum=0.0,
                           hint="Elastic (zero-energy) amplitude", decimals=6, step=0.01))
        self.add(Parameter("BG", value=0.0, ptype="float",
                           hint="Flat background"))
        self.add(Parameter("T", value=10.0, ptype="float", minimum=0.1,
                           hint="Temperature (K)", decimals=2, step=0.1))
        self.add(Parameter("center", value=0.0, ptype="float", hint="Spectral center",
                           input_hint={"drag": {"action": "set", "value_from": "x"}}))

    def initialize(self, data_x=None, data_y=None):
        # as example, pick center near max if available
        try:
            if data_x is not None and data_y is not None:
                arrx = np.asarray(data_x)
                arry = np.asarray(data_y)
                idx = int(np.nanargmax(arry))
                self.params["center"].value = float(arrx[idx])
        except Exception:
            pass

    def evaluate(self, x, params: Optional[Dict[str, Any]] = None):
        """Simple visible DHO-like curve (no instrument convolution here)."""
        try:
            arr = np.asarray(x, dtype=float)
            pvals = self.get_param_values(params)
            center_spec = float(pvals.get("center", 0.0))
            phonon_energy = float(pvals.get("phonon_energy", 5.0))
            damping = float(pvals.get("damping", 0.5))
            phonon_amp = float(pvals.get("phonon_amplitude", 0.1))
            elastic_amp = float(pvals.get("elastic_amplitude", 0.0))
            BG = float(pvals.get("BG", 0.0))
            T = float(pvals.get("T", 10.0))

            x_shifted = arr - center_spec
            stokes = Stokes_DHO(x_shifted, amplitude=phonon_amp, damping=damping, center=phonon_energy)
            try:
                astokes_amp = phonon_amp / np.exp(phonon_energy / (kB * T))
            except Exception:
                astokes_amp = phonon_amp
            astokes = antiStokes_DHO(x_shifted, amplitude=astokes_amp, damping=damping, center=phonon_energy)
            elastic = narrow_gaussian_delta(arr, center_spec, amplitude=elastic_amp, fwhm=0.1)
            return stokes + astokes + elastic + BG
        except Exception:
            return super().evaluate(x, params)


class LinearBackgroundModelSpec(BaseModelSpec):
    """Simple linear background: y = slope * x + constant"""
    def __init__(self):
        super().__init__()
        # slope controlled by Ctrl+mouse wheel (scale)
        self.add(Parameter("slope", value=0.0, ptype="float",
                           hint="Linear slope (m)", decimals=6, step=0.01,
                           input_hint={"wheel": {"modifiers": ["Ctrl"], "action": "scale", "factor": 1.05}}))
        # constant (intercept) controlled by mouse wheel
        self.add(Parameter("constant", value=0.0, ptype="float",
                           hint="Constant background (b)", decimals=6, step=0.1,
                           input_hint={"wheel": {"action": "increment", "step": 0.1}}))

    def evaluate(self, x, params: Optional[Dict[str, Any]] = None):
        try:
            pvals = self.get_param_values(params)
            m = float(pvals.get("slope", 0.0))
            b = float(pvals.get("constant", 0.0))
            arr = np.asarray(x, dtype=float)
            return m * arr + b
        except Exception:
            return super().evaluate(x, params)

class DHOVoigtModelSpec(BaseModelSpec):
    """Composite DHO convolved with Voigt resolution + elastic Voigt."""
    def __init__(self):
        super().__init__()
        # This composite is a constructed element (starting point) and
        # should not be addable to other models.
        self.is_constructed = True
        # include DHO params
        self.add(Parameter("phonon_energy", value=5.0, ptype="float", minimum=0.0, hint="Phonon energy (meV)",
                           input_hint={"wheel": {"modifiers": ["Ctrl"], "action": "scale", "factor": 1.05}}))
        self.add(Parameter("damping", value=0.5, ptype="float", minimum=1e-6, hint="DHO damping",
                           input_hint={"wheel": {"modifiers": ["Shift"], "action": "scale", "factor": 1.05}}))
        self.add(Parameter("phonon_amplitude", value=0.1, ptype="float", minimum=0.0, hint="Phonon amplitude",
                           input_hint={"wheel": {"modifiers": ["Ctrl"], "action": "scale", "factor": 1.05}}))
        # Voigt resolution params
        self.add(Parameter("gauss_fwhm", value=1.14, ptype="float", minimum=0.0, hint="Gaussian FWHM of resolution",
                           input_hint={"wheel": {"modifiers": ["Ctrl"], "action": "scale", "factor": 1.05}}))
        self.add(Parameter("lorentz_fwhm", value=0.28, ptype="float", minimum=0.0, hint="Lorentzian FWHM of resolution",
                           input_hint={"wheel": {"modifiers": ["Shift"], "action": "scale", "factor": 1.05}}))
        # elastic and BG
        self.add(Parameter("elastic_amplitude", value=0.0, ptype="float", minimum=0.0, hint="Elastic Voigt area",
                           input_hint={"wheel": {"modifiers": ["Ctrl"], "action": "scale", "factor": 1.1}}))
        self.add(Parameter("BG", value=0.0, ptype="float", hint="Flat background",
                           input_hint={"wheel": {"modifiers": ["Alt"], "action": "increment", "step": 0.01}}))
        self.add(Parameter("T", value=10.0, ptype="float", minimum=0.1, hint="Temperature (K)",
                           input_hint=None))
        self.add(Parameter("center", value=0.0, ptype="float", hint="Spectrum center",
                           input_hint={"drag": {"action": "set", "value_from": "x"}}))

    def initialize(self, data_x=None, data_y=None):
        # example: set center from data if available
        try:
            if data_x is not None and data_y is not None:
                arrx = np.asarray(data_x)
                arry = np.asarray(data_y)
                idx = int(np.nanargmax(arry))
                self.params["center"].value = float(arrx[idx])
        except Exception:
            pass

    def evaluate(self, x, params: Optional[Dict[str, Any]] = None):
        """Use the convolute_voigt_dho helper to produce a visible, realistic line."""
        try:
            arr = np.asarray(x, dtype=float)
            pvals = self.get_param_values(params)
            phonon_energy = float(pvals.get("phonon_energy", 5.0))
            damping = float(pvals.get("damping", 0.5))
            phonon_amplitude = float(pvals.get("phonon_amplitude", 0.1))
            gauss_fwhm = float(pvals.get("gauss_fwhm", 1.14))
            lorentz_fwhm = float(pvals.get("lorentz_fwhm", 0.28))
            elastic_amplitude = float(pvals.get("elastic_amplitude", 0.0))
            BG = float(pvals.get("BG", 0.0))
            T = float(pvals.get("T", 10.0))
            center = float(pvals.get("center", 0.0))

            return convolute_voigt_dho(arr, phonon_energy=phonon_energy, center=center,
                                       gauss_fwhm=gauss_fwhm, lorentz_fwhm=lorentz_fwhm,
                                       damping=damping, phonon_amplitude=phonon_amplitude,
                                       elastic_amplitude=elastic_amplitude, BG=BG, T=T, peak='all')
        except Exception:
            return super().evaluate(x, params)

# Factory / helper
# Programmatic model registry -------------------------------------------------
# Each entry: key (canonical), display name, aliases, factory, is_constructed
MODEL_REGISTRY = [
    {"key": "voigt", "display": "Voigt", "factory": VoigtModelSpec, "is_constructed": False},
    {"key": "dho+voigt", "display": "DHO+Voigt", "factory": DHOVoigtModelSpec, "is_constructed": True},
    {"key": "gaussian", "display": "Gaussian", "factory": GaussianModelSpec, "is_constructed": False},
    {"key": "dho", "display": "DHO", "factory": DHOModelSpec, "is_constructed": False},
    {"key": "lorentzian", "display": "Lorentzian", "factory": LorentzModelSpec, "is_constructed": False},
    {"key": "linear", "display": "Linear", "factory": LinearBackgroundModelSpec, "is_constructed": False},
]


def _find_registry_entry(name: str):
    """Return the registry entry matching name (case-insensitive) or None.

    Matching rules (STRICT):
      - exact match on canonical key (case-insensitive)
      - exact match on display name (case-insensitive)

    This intentionally avoids tolerant/alias matching so that the UI
    and other callers must use the canonical names exposed by
    get_available_model_names().
    """
    if not name:
        return None
    s = str(name).strip().lower()
    # direct key or display match only
    for e in MODEL_REGISTRY:
        if e["key"].lower() == s or (e.get("display") or "").lower() == s:
            return e
    return None


def canonical_model_key(name: str) -> str:
    """Return the canonical registry key for a model name or the cleaned input.

    This is useful for storing a normalized model identifier in state.
    """
    ent = _find_registry_entry(name)
    if ent:
        return ent["key"]
    # fallback: cleaned lowercase
    return (name or "").strip().lower()


def get_model_spec(model_name: str) -> BaseModelSpec:
    """Create and return a ModelSpec instance for the requested model name.

    The lookup accepts:
    - Canonical keys and display names from MODEL_REGISTRY
    - Custom models with "Custom: <name>" or "custom:<name>" format
    """
    # Check if this is a custom model request
    name_lower = str(model_name).lower().strip()
    
    # Handle "Custom: Name" or "custom:Name" format
    if name_lower.startswith("custom:") or name_lower.startswith("custom "):
        # Extract the actual custom model name
        if ":" in model_name:
            custom_name = model_name.split(":", 1)[1].strip()
        else:
            # "Custom Name" format
            custom_name = model_name.split(None, 1)[1].strip() if " " in model_name else ""
        
        try:
            from models.custom_model_registry import get_custom_model_registry
            registry = get_custom_model_registry()
            model_data = registry.get_model(custom_name)
            
            if model_data:
                components = model_data.get("components", [])
                return CompositeModelSpec(components)
        except Exception as e:
            print(f"Failed to load custom model '{custom_name}': {e}")
            return BaseModelSpec()
    
    # Standard built-in model lookup
    ent = _find_registry_entry(model_name)
    if ent is None:
        return BaseModelSpec()
    try:
        return ent["factory"]()
    except Exception:
        return BaseModelSpec()


def get_available_model_names(addable_only: bool = False) -> list:
    """Return a list of human-friendly model display names.

    If addable_only is True, filter out models marked as constructed (not addable).
    """
    names = []
    for e in MODEL_REGISTRY:
        if addable_only and e.get("is_constructed"):
            continue
        names.append(e.get("display", e.get("key")))
    
    # Add custom models from the registry
    try:
        from models.custom_model_registry import get_custom_model_registry
        registry = get_custom_model_registry()
        custom_names = registry.get_custom_model_names()
        for name in custom_names:
            names.append(f"Custom: {name}")
    except Exception:
        pass
    
    return names


class CompositeModelSpec(BaseModelSpec):
    """A composite model that sums the outputs of multiple component models.
    
    Each component is defined by:
      - base_spec: the canonical key of the base model (e.g., "gaussian")
      - label: a user-friendly label (e.g., "Gaussian 1")
      - params: a dict of parameter values specific to this component
    
    Parameters are organized into groups, one per component.
    """
    
    def __init__(self, components: Optional[List[Dict[str, Any]]] = None):
        """Initialize a composite model.
        
        Args:
            components: List of component dicts, each with:
                - base_spec: str (canonical model key)
                - label: str (display label)
                - params: dict (parameter values)
        """
        super().__init__()
        self.is_constructed = True  # Composites are constructed, not addable to other composites
        self.components = components or []
        
        # Build parameter map: create independent copies of parameters for each component
        self._rebuild_params()
    
    def _rebuild_params(self):
        """Rebuild the params dict from components."""
        self.params.clear()
        
        for idx, comp in enumerate(self.components):
            try:
                base_spec_key = comp.get("base_spec", "")
                label = comp.get("label", f"Component {idx+1}")
                comp_params = comp.get("params", {})
                
                # Get the base model spec
                base_spec = get_model_spec(base_spec_key)
                
                # Create prefixed parameters for this component
                # Format: "label::param_name"
                for param_name, param_obj in base_spec.params.items():
                    # Make a copy of the Parameter object
                    param_copy = copy.deepcopy(param_obj)
                    
                    # Override value if provided in component params
                    if param_name in comp_params:
                        param_copy.value = comp_params[param_name]
                    
                    # Store with prefixed name
                    prefixed_name = f"{label}::{param_name}"
                    self.params[prefixed_name] = param_copy
            
            except Exception as e:
                print(f"Failed to add component {idx} to composite: {e}")
    
    def get_parameters(self) -> Dict[str, Any]:
        """Return grouped parameter specs for UI display.
        
        Returns a dict with a special "groups" key containing grouped parameters:
        {
            "groups": [
                {
                    "label": "Gaussian 1",
                    "params": {
                        "Area": {"value": 1.0, "type": "float", ...},
                        ...
                    }
                },
                ...
            ]
        }
        
        For backwards compatibility, also includes flattened params with prefixed names.
        """
        # Build grouped structure
        groups = []
        
        for idx, comp in enumerate(self.components):
            label = comp.get("label", f"Component {idx+1}")
            base_spec_key = comp.get("base_spec", "")
            
            # Get base spec to know which params belong to this component
            try:
                base_spec = get_model_spec(base_spec_key)
                base_param_names = list(base_spec.params.keys())
            except Exception:
                base_param_names = []
            
            # Collect parameters for this group
            group_params = {}
            for param_name in base_param_names:
                prefixed_name = f"{label}::{param_name}"
                if prefixed_name in self.params:
                    param_obj = self.params[prefixed_name]
                    group_params[param_name] = param_obj.to_spec()
            
            if group_params:
                groups.append({
                    "label": label,
                    "base_spec": base_spec_key,
                    "params": group_params,
                    "component_index": idx
                })
        
        # Return both grouped and flat representations
        result = {
            "groups": groups
        }
        
        # Also add flattened params for backwards compatibility
        for name, p in self.params.items():
            result[name] = p.to_spec()
        
        return result
    
    def evaluate(self, x, params: Optional[Dict[str, Any]] = None):
        """Evaluate the composite model by summing all component outputs."""
        try:
            x_arr = np.asarray(x, dtype=float)
            result = np.zeros_like(x_arr)
            
            # Get current parameter values
            param_values = self.get_param_values(params)
            
            # Evaluate each component and sum
            for idx, comp in enumerate(self.components):
                try:
                    base_spec_key = comp.get("base_spec", "")
                    label = comp.get("label", f"Component {idx+1}")
                    
                    # Get base spec
                    base_spec = get_model_spec(base_spec_key)
                    
                    # Extract parameters for this component (remove prefix)
                    comp_params = {}
                    for param_name in base_spec.params.keys():
                        prefixed_name = f"{label}::{param_name}"
                        if prefixed_name in param_values:
                            comp_params[param_name] = param_values[prefixed_name]
                    
                    # Evaluate component
                    component_output = base_spec.evaluate(x_arr, comp_params)
                    result += component_output
                
                except Exception as e:
                    print(f"Failed to evaluate component {idx} in composite: {e}")
            
            return result
        
        except Exception:
            return super().evaluate(x, params)
