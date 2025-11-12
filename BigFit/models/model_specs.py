# type: ignore
"""Model specifications and helper functions for fitting.
"""
import numpy as np
from scipy.special import wofz, voigt_profile
from scipy.fft import fft, ifft
from scipy.interpolate import interp1d
from scipy.signal import fftconvolve
from dataclasses import dataclass
from itertools import cycle

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

from typing import Any, Dict, Optional, List, Iterable

# Pre-defined colors cycled for composite components.
_COMPONENT_COLORS = [
    "#1f77b4",  # blue
    "#ff7f0e",  # orange
    "#2ca02c",  # green
    "#d62728",  # red
    "#9467bd",  # purple
    "#8c564b",  # brown
    "#e377c2",  # pink
    "#7f7f7f",  # gray
    "#bcbd22",  # olive
    "#17becf",  # teal
]


def _clone_parameter(param: "Parameter", new_name: str, value: Any) -> "Parameter":
    """Return a shallow copy of `param` with a new name/value."""
    return Parameter(
        name=new_name,
        value=value,
        ptype=getattr(param, "ptype", None),
        minimum=getattr(param, "min", None),
        maximum=getattr(param, "max", None),
        choices=getattr(param, "choices", None),
        hint=getattr(param, "hint", ""),
        decimals=getattr(param, "decimals", None),
        step=getattr(param, "step", None),
        control=getattr(param, "control", None),
        fixed=getattr(param, "fixed", False),
    )


@dataclass
class CompositeComponent:
    prefix: str
    spec: "BaseModelSpec"
    color: str

    @property
    def label(self) -> str:
        return f"{self.prefix.rstrip('_')}"



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
                 control: Optional[Dict[str, Any]] = None,
                 fixed: Optional[bool] = False):
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
    # optional interactive control metadata describing how UI input maps to this param
    # example: {"action": "wheel", "modifiers": []}
        self.control = control
        # whether this parameter should be held fixed during fitting
        self.fixed = bool(fixed)

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

        The view will use these keys to pick and configure widgets.
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
        if getattr(self, "control", None) is not None:
            # pass through control metadata (UI may map input events based on this).
            # Interactive step/increment is intentionally NOT stored here; the view
            # layer derives the interactive increment from the parameter 'step'
            # so that a single-step value controls both widget increments and
            # interactive inputs.
            spec["control"] = dict(self.control)
        # expose fixed state to the UI so parameters can be marked fixed/unfixed
        try:
            spec["fixed"] = bool(getattr(self, "fixed", False))
        except Exception:
            spec["fixed"] = False
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


class CompositeModelSpec(BaseModelSpec):
    """Model spec composed of multiple atomic specs summed together."""

    is_composite = True

    def __init__(self):
        super().__init__()
        self._components: List[CompositeComponent] = []
        self._color_iter = cycle(_COMPONENT_COLORS)
        self._prefix_counter = 1
        self._param_links: Dict[str, Any] = {}

    # ----- component management -------------------------------------------------
    def _next_color(self) -> str:
        try:
            return next(self._color_iter)
        except Exception:
            return "#1f77b4"

    def _generate_prefix(self) -> str:
        attempt = self._prefix_counter
        existing = {comp.prefix for comp in self._components}
        while True:
            candidate = f"elem{attempt}_"
            if candidate not in existing:
                self._prefix_counter = attempt + 1
                return candidate
            attempt += 1

    def add_component(
        self,
        spec_name: str,
        initial_params: Optional[Dict[str, Any]] = None,
        prefix: Optional[str] = None,
        data_x=None,
        data_y=None,
    ) -> CompositeComponent:
        from models import get_model_spec

        spec_name = (spec_name or "").strip()
        if not spec_name:
            raise ValueError("spec_name required")
        spec = get_model_spec(spec_name)
        if isinstance(spec, CompositeModelSpec):
            raise ValueError("Nested composite models are not supported")

        try:
            spec.initialize(data_x, data_y)
        except Exception:
            try:
                spec.initialize()
            except Exception:
                pass

        prefix = prefix or self._generate_prefix()
        color = self._next_color()
        component = CompositeComponent(prefix=prefix, spec=spec, color=color)
        self._components.append(component)

        if initial_params:
            for name, value in initial_params.items():
                if name in spec.params:
                    spec.params[name].value = value

        self._rebuild_flat_params()
        return component

    def remove_component_at(self, index: int) -> Optional[CompositeComponent]:
        if index < 0 or index >= len(self._components):
            return None
        removed = self._components.pop(index)
        self._rebuild_flat_params()
        return removed

    def reorder_component(self, old_index: int, new_index: int) -> bool:
        if old_index < 0 or new_index < 0:
            return False
        if old_index >= len(self._components) or new_index >= len(self._components):
            return False
        if old_index == new_index:
            return True
        comp = self._components.pop(old_index)
        self._components.insert(new_index, comp)
        self._rebuild_flat_params()
        return True

    def reorder_by_prefix(self, prefix_order: Iterable[str]) -> bool:
        prefix_order = [p for p in prefix_order]
        if not prefix_order:
            return False
        current = {comp.prefix: comp for comp in self._components}
        new_order: List[CompositeComponent] = []
        for prefix in prefix_order:
            comp = current.get(prefix)
            if comp is not None:
                new_order.append(comp)
        if len(new_order) != len(self._components):
            return False
        if all(a.prefix == b.prefix for a, b in zip(new_order, self._components)):
            return True
        self._components = new_order
        self._rebuild_flat_params()
        return True

    def clear_components(self):
        self._components.clear()
        self._rebuild_flat_params()

    # ----- aggregation helpers --------------------------------------------------
    def _rebuild_flat_params(self):
        self.params = {}
        self._param_links = {}
        for component in self._components:
            for name, param in component.spec.params.items():
                flat_name = f"{component.prefix}{name}"
                value = getattr(param, "value", None)
                cloned = _clone_parameter(param, flat_name, value)
                try:
                    setattr(cloned, "component_prefix", component.prefix)
                    setattr(cloned, "component_label", component.label)
                    setattr(cloned, "component_color", component.color)
                except Exception:
                    pass
                self.params[flat_name] = cloned
                self._param_links[flat_name] = (component, name)

    def get_parameters(self) -> Dict[str, Any]:
        self._rebuild_flat_params()
        specs = super().get_parameters()
        for name, spec in specs.items():
            link = self._param_links.get(name)
            if link:
                component, _ = link
                spec.setdefault("component", component.prefix)
                spec.setdefault("component_label", component.label)
                spec.setdefault("color", component.color)
        return specs

    def get_param_values(self, overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        self._rebuild_flat_params()
        return super().get_param_values(overrides=overrides)

    def list_components(self) -> List[CompositeComponent]:
        return list(self._components)

    def set_param_value(self, flat_name: str, value: Any):
        link = self._param_links.get(flat_name)
        if link:
            component, name = link
            try:
                component.spec.params[name].value = value
            except Exception:
                pass
        if flat_name in self.params:
            try:
                self.params[flat_name].value = value
            except Exception:
                pass

    def get_link(self, flat_name: str):
        return self._param_links.get(flat_name)

    # ----- BaseModelSpec overrides ---------------------------------------------
    def initialize(self, data_x=None, data_y=None):
        for component in self._components:
            try:
                component.spec.initialize(data_x, data_y)
            except Exception:
                continue
        self._rebuild_flat_params()

    def evaluate_components(self, x, params: Optional[Dict[str, Any]] = None):
        x_arr = np.asarray(x, dtype=float)
        if x_arr.size == 0:
            return []

        param_values = self.get_param_values(overrides=params or {})
        outputs = []

        for component in self._components:
            sub_params: Dict[str, Any] = {}
            for name in component.spec.params.keys():
                key = f"{component.prefix}{name}"
                if key in param_values:
                    sub_params[name] = param_values[key]
                else:
                    sub_params[name] = component.spec.params[name].value
            try:
                contribution = component.spec.evaluate(x_arr, sub_params)
            except Exception:
                contribution = np.zeros_like(x_arr, dtype=float)
            try:
                outputs.append((component, np.asarray(contribution, dtype=float)))
            except Exception:
                outputs.append((component, np.zeros_like(x_arr, dtype=float)))

        return outputs

    def evaluate(self, x, params: Optional[Dict[str, Any]] = None):
        component_outputs = self.evaluate_components(x, params=params)
        if not component_outputs:
            return np.zeros_like(np.asarray(x, dtype=float), dtype=float)

        total = np.zeros_like(np.asarray(x, dtype=float), dtype=float)
        for _, values in component_outputs:
            try:
                total += values
            except Exception:
                pass

        return total

# Concrete model specs
class GaussianModelSpec(BaseModelSpec):
    def __init__(self):
        super().__init__()
        # Interactive controls added so the InputHandler / View can map
        # wheel and mouse actions to these parameters.
        self.add(Parameter("Area", value=1.0, ptype="float", minimum=0.0,
                           hint="Integrated area of the Gaussian peak", decimals=6, step=1.0,
                           control={"action": "wheel", "modifiers": []}))
        # Ctrl + wheel adjusts the width (FWHM)
        self.add(Parameter("Width", value=2, ptype="float", minimum=1e-6,
                           hint="Gaussian FWHM", decimals=6, step=0.1,
                           control={"action": "wheel", "modifiers": ["Control"]}))
        # Click-drag (mouse_move / peak drag) should update the center.
        # Keep the parameter named "Center" (capital C) to match evaluate()'s
        # argument names, but provide a control hint for the UI.
        self.add(Parameter("Center", value=0.0, ptype="float",
                           hint="Peak center (x-axis)",
                           control={"action": "mouse_move", "modifiers": []}))

    def evaluate(self, x, params: Optional[Dict[str, Any]] = None):
        try:
            pvals = self.get_param_values(params)
            Area = float(pvals.get("Area", 1.0))
            Width = float(pvals.get("Width", 2.0))
            Center = float(pvals.get("Center", 0.0))
            return Gaussian(np.asarray(x, dtype=float), Area=Area, Width=Width, Center=Center)
        except Exception:
            return super().evaluate(x, params)

class VoigtModelSpec(BaseModelSpec):
    def __init__(self):
        super().__init__()
        # Interactive control bindings are included in the Parameter so the view/input
        # layer can map events to parameter updates dynamically (no UI hard-coding).
        self.add(Parameter("Area", value=1.0, ptype="float", minimum=0.0,
                           hint="Integrated area of the Voigt peak", decimals=6, step=1.0,
                           control={"action": "wheel", "modifiers": []}))
        # Ctrl + wheel adjusts the gaussian contribution
        self.add(Parameter("Gauss FWHM", value=1.14, ptype="float", minimum=0.0,
                           hint="Gaussian resolution FWHM", decimals=6, step=0.1,
                           control={"action": "wheel", "modifiers": ["Control"]}))
        # Shift + wheel adjusts the lorentzian contribution
        self.add(Parameter("Lorentz FWHM", value=0.28, ptype="float", minimum=0.0,
                           hint="Lorentzian FWHM", decimals=6, step=0.1,
                           control={"action": "wheel", "modifiers": ["Shift"]}))
        # Mouse movement (no modifiers) controls center by default
        self.add(Parameter("Center", value=0.0, ptype="float",
                           hint="Peak center",
                           control={"action": "mouse_move", "modifiers": []}))

    def initialize(self, data_x=None, data_y=None):
        # simple example: set center to x of max if data provided
        try:
            if data_x is not None and data_y is not None:
                arrx = np.asarray(data_x)
                arry = np.asarray(data_y)
                idx = int(np.nanargmax(arry))
                self.params["Center"].value = float(arrx[idx])
        except Exception:
            pass

    def evaluate(self, x, params: Optional[Dict[str, Any]] = None):
        try:
            pvals = self.get_param_values(params)
            Area = float(pvals.get("Area", 1.0))
            gauss_fwhm = float(pvals.get("Gauss FWHM", 1.14))
            lorentz_fwhm = float(pvals.get("Lorentz FWHM", 0.28))
            center = float(pvals.get("Center", 0.0))
            return Voigt(np.asarray(x, dtype=float), Area=Area, gauss_fwhm=gauss_fwhm, lorentz_fwhm=lorentz_fwhm, center=center)
        except Exception:
            return super().evaluate(x, params)




class LinearBackgroundModelSpec(BaseModelSpec):
    """Simple linear background y = m*x + b

    Parameters exposed to the view are:
      - "Slope" (m)
      - "Intercept" (b)
    """
    def __init__(self):
        super().__init__()
        # slope (m)
        self.add(Parameter("Slope", value=0.0, ptype="float", hint="Linear slope (m)", decimals=6, step=0.01,
                           control={"action": "wheel", "modifiers": []}))
        # intercept (b)
        self.add(Parameter("Intercept", value=0.0, ptype="float", hint="Vertical offset / intercept (b)", decimals=6, step=0.1,
                           control={"action": "wheel", "modifiers": ["Control"]}))

    def evaluate(self, x, params: Optional[Dict[str, Any]] = None):
        try:
            pvals = self.get_param_values(params)
            m = float(pvals.get("Slope", 0.0))
            b = float(pvals.get("Intercept", 0.0))
            xarr = np.asarray(x, dtype=float)
            return m * xarr + b
        except Exception:
            return super().evaluate(x, params)

# Factory / helper
def get_model_spec(model_name: str) -> BaseModelSpec:
    name = (model_name or "").strip().lower()
    if name in ("composite", "custom", "custom model", "custommodel"):
        return CompositeModelSpec()
    if name in ("gauss", "gaussian", "gaussmodel", "gaussianmodel"):
        return GaussianModelSpec()
    if name in ("voigt", "voigtmodel"):
        return VoigtModelSpec()
    if name in ("linear", "linear background", "linearbackground", "linearbackgroundmodel", "linearbg", "background", "linear model"):
        return LinearBackgroundModelSpec()
    # default fallback
    return BaseModelSpec()


def get_atomic_component_names() -> List[str]:
    """Return display names for atomic components usable in composite models."""
    return [
        "Gaussian",
        "Voigt",
        "Linear Background",
    ]
