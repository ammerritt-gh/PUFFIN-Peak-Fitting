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
                           hint="Integrated area of the Gaussian peak", decimals=6, step=0.1))
        self.add(Parameter("Width", value=0.1, ptype="float", minimum=1e-6,
                           hint="Gaussian FWHM", decimals=6, step=0.01))
        self.add(Parameter("Center", value=0.0, ptype="float",
                           hint="Peak center (x-axis)"))

    def evaluate(self, x, params: Optional[Dict[str, Any]] = None):
        try:
            pvals = self.get_param_values(params)
            Area = float(pvals.get("Area", 1.0))
            Width = float(pvals.get("Width", 0.1))
            Center = float(pvals.get("Center", 0.0))
            return Gaussian(np.asarray(x, dtype=float), Area=Area, Width=Width, Center=Center)
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

class DHOVoigtModelSpec(BaseModelSpec):
    """Composite DHO convolved with Voigt resolution + elastic Voigt."""
    def __init__(self):
        super().__init__()
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
def get_model_spec(model_name: str) -> BaseModelSpec:
    name = (model_name or "").strip().lower()
    if name in ("gauss", "gaussian", "gaussmodel", "gaussianmodel"):
        return GaussianModelSpec()
    if name in ("voigt", "voigtmodel"):
        return VoigtModelSpec()
    if name in ("dho", "dhomodel", "dhoonly"):
        return DHOModelSpec()
    if name in ("dho+voigt", "dho_voigt", "dho+voigtmodel", "dho+voigtmodel"):
        return DHOVoigtModelSpec()
    # default fallback
    return BaseModelSpec()
