# -*- coding: utf-8 -*-
import sys
import os
import re
import json
import numpy as np
import pandas as pd
from functools import partial

# --- PyQtGraph Imports ---
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets, QtGui
from pyqtgraph import PlotWidget, PlotDataItem, ErrorBarItem, ScatterPlotItem, InfiniteLine, LinearRegionItem, BarGraphItem
from pyqtgraph.exporters import ImageExporter

# Bind Qt classes through pyqtgraph's selected backend to avoid mismatches
QApplication = QtWidgets.QApplication
QMainWindow = QtWidgets.QMainWindow
QWidget = QtWidgets.QWidget
QVBoxLayout = QtWidgets.QVBoxLayout
QHBoxLayout = QtWidgets.QHBoxLayout
QPushButton = QtWidgets.QPushButton
QFileDialog = QtWidgets.QFileDialog
QTextEdit = QtWidgets.QTextEdit
QDoubleSpinBox = QtWidgets.QDoubleSpinBox
QLabel = QtWidgets.QLabel
QFormLayout = QtWidgets.QFormLayout
QLineEdit = QtWidgets.QLineEdit
QAbstractSpinBox = QtWidgets.QAbstractSpinBox
QListWidget = QtWidgets.QListWidget
QListWidgetItem = QtWidgets.QListWidgetItem
QAbstractItemView = QtWidgets.QAbstractItemView
QMessageBox = QtWidgets.QMessageBox
QCheckBox = QtWidgets.QCheckBox
QSplitter = QtWidgets.QSplitter

QObject = QtCore.QObject
QThread = QtCore.QThread
QTimer = QtCore.QTimer
Qt = QtCore.Qt
QKeyEvent = QtGui.QKeyEvent

# Cross-backend Signal alias (PySide: Signal, PyQt: pyqtSignal)
try:
    Signal = QtCore.Signal  # PySide6
except AttributeError:
    try:
        Signal = QtCore.pyqtSignal  # PyQt6/PyQt5
    except AttributeError:
        Signal = None

# Ensure a single QApplication exists for the module to avoid warnings and init order issues
_app = QApplication.instance() or pg.mkQApp("Interactive Peak Fitter (PyQtGraph)")

# ----- Qt enum compatibility shims (PyQt6/PySide6 moved many enums under sub-enums) -----
try:
    AlignLeft = Qt.AlignmentFlag.AlignLeft
    AlignVCenter = Qt.AlignmentFlag.AlignVCenter
except Exception:
    AlignLeft = getattr(Qt, 'AlignLeft', 0)
    AlignVCenter = getattr(Qt, 'AlignVCenter', 0)

try:
    TextSelectableByMouse = Qt.TextInteractionFlag.TextSelectableByMouse
except Exception:
    TextSelectableByMouse = getattr(Qt, 'TextSelectableByMouse', 0)

try:
    DashLine = Qt.PenStyle.DashLine
except Exception:
    DashLine = getattr(Qt, 'DashLine', 1)

# Provide a dotted line style with fallback
try:
    DotLine = Qt.PenStyle.DotLine
except Exception:
    DotLine = getattr(Qt, 'DotLine', DashLine)

try:
    RightButton = Qt.MouseButton.RightButton
    LeftButton = Qt.MouseButton.LeftButton
except Exception:
    RightButton = getattr(Qt, 'RightButton', 2)
    LeftButton = getattr(Qt, 'LeftButton', 1)

try:
    ControlModifier = Qt.KeyboardModifier.ControlModifier
    ShiftModifier = Qt.KeyboardModifier.ShiftModifier
    AltModifier = Qt.KeyboardModifier.AltModifier
except Exception:
    ControlModifier = getattr(Qt, 'ControlModifier', 0x04000000)
    ShiftModifier = getattr(Qt, 'ShiftModifier', 0x02000000)
    AltModifier = getattr(Qt, 'AltModifier', 0x08000000)

try:
    Horizontal = Qt.Orientation.Horizontal
except Exception:
    Horizontal = getattr(Qt, 'Horizontal', 1)

try:
    StrongFocus = Qt.FocusPolicy.StrongFocus
except Exception:
    StrongFocus = getattr(Qt, 'StrongFocus', 0x08)

def _qt_key(name, fallback=None):
    try:
        return getattr(Qt.Key, name)
    except Exception:
        return getattr(Qt, name, fallback)

Key_D = _qt_key('Key_D')
Key_E = _qt_key('Key_E')
Key_Tab = _qt_key('Key_Tab')
Key_Right = _qt_key('Key_Right')
Key_Left = _qt_key('Key_Left')
Key_R = _qt_key('Key_R')
Key_1 = _qt_key('Key_1', ord('1'))
Key_2 = _qt_key('Key_2', ord('2'))
Key_3 = _qt_key('Key_3', ord('3'))
Key_4 = _qt_key('Key_4', ord('4'))
Key_5 = _qt_key('Key_5', ord('5'))
Key_6 = _qt_key('Key_6', ord('6'))
Key_7 = _qt_key('Key_7', ord('7'))
Key_8 = _qt_key('Key_8', ord('8'))
Key_9 = _qt_key('Key_9', ord('9'))
Key_Space = _qt_key('Key_Space', ord(' '))

# Set PyQtGraph to use antialiasing for better quality
pg.setConfigOptions(antialias=True)

# --- Placeholder for your scientific fitting functions ---
# To make this runnable, we use a stub. Replace this with your actual file.
from fitting_functions import convolute_voigt_dho, voigt_area_for_height, dho_voigt_area_for_height, Voigt

# --- Configuration Constants (tune defaults and UI behavior here) ---
# General UI defaults
DEFAULT_WINDOW_WIDTH = 1400
DEFAULT_WINDOW_HEIGHT = 850
MIN_LEFT_PANEL_WIDTH = 380
MAX_LEFT_PANEL_WIDTH = 600
DEFAULT_LEFT_PANEL_WIDTH = 420
DEFAULT_RIGHT_PLOT_SIZE = 900
DEFAULT_HELP_PANEL_WIDTH = 280

# File dialog and IO
DATA_FILE_FILTER = "Data Files (*.dat)"
RESULTS_SUFFIX = "_fit.txt"
FIGURE_SUFFIX = "_fit.png"
FIGURE_EXPORT_DPI = 200

# Data reading
CSV_COL_ENERGY = 'Energy'
CSV_COL_COUNTS = 'Counts/min'
CSV_COL_ERROR = 'Error'

# Preview grid
PREVIEW_SAMPLES_MIN = 1000
PREVIEW_SAMPLES_FACTOR = 2

# Model defaults and bounds
DEFAULT_GAUSS_FWHM = 1.14
DEFAULT_LORENTZ_FWHM = 0.28
DEFAULT_BG = 50.0
DEFAULT_ELASTIC_HEIGHT = 1000.0
RES_FWHM_MIN = 0.01
RES_FWHM_MAX = 10.0
COMPOSITION_DHO = 'stokes+astokes'

# DHO damping
DHO_DAMPING_DEFAULT = 0.1
DHO_DAMPING_MIN = 0.005
DHO_DAMPING_MAX = 2.0
DHO_DAMPING_STEP = 0.01

# Bounds strategy around initial guesses
RES_BOUND_SCALE_LO = 0.5
RES_BOUND_SCALE_HI = 2.0
RES_BOUND_ZERO_FALLBACK = 1.0
ENERGY_MIN = 0.01
ENERGY_SCALE_LO = 0.5
ENERGY_SCALE_HI = 1.5
ENERGY_ABS_PAD = 2.0

# Optimizer
FIT_MAX_NFEV = 400

# Interaction scales
RES_SCROLL_SCALE = 1.1
DAMP_SCROLL_SCALE = 1.1
HEIGHT_SCROLL_SCALE = 1.05
ZOOM_WHEEL_FACTOR = 1.2

# Pixel thresholds (squared distance for click vs drag boxes)
BOX_CLICK_PIXEL_DIST2 = 25.0
NEAREST_TARGET_PIXEL_THRESHOLD = 50
EXCLUDE_CLICK_PIXEL_THRESHOLD = 10

# Marker/line visuals
PHONON_COLOR_CYCLE = ['b', 'g', 'c', 'm', 'y']
LINEWIDTH_SELECTED = 2.5
LINEWIDTH_NORMAL = 1.5
MARKER_SIZE_SELECTED = 12
MARKER_SIZE_NORMAL = 10

# Axes margins for reset zoom
X_MARGIN_FRAC = 0.06
Y_MARGIN_LOW = 0.08
Y_MARGIN_HIGH = 0.12

# Numeric stability
EPS_REL_FIX = 1e-9
EPS_ABS_TINY = 1e-12
FD_REL_STEP = 1e-6

# Helper functions
def clamp(val, lo, hi):
    try:
        return min(hi, max(lo, float(val)))
    except Exception:
        return lo

def clamp_damping(val):
    return clamp(val, DHO_DAMPING_MIN, DHO_DAMPING_MAX)

# --- Small helper wrappers to reduce repetition ---
def elastic_area_from_height(height: float, gauss_fwhm: float, lorentz_fwhm: float) -> float:
    """Convert elastic peak height to Voigt area using current resolution widths."""
    return float(voigt_area_for_height(float(height), float(gauss_fwhm), float(lorentz_fwhm)))

def elastic_voigt_y(x, area: float, gauss_fwhm: float, lorentz_fwhm: float, center: float = 0.0):
    """Elastic Voigt profile for given area and resolution widths."""
    return Voigt(np.asarray(x, dtype=float), Area=float(area), gauss_fwhm=float(gauss_fwhm), lorentz_fwhm=float(lorentz_fwhm), center=float(center))

def elastic_baseline_y(x, height: float, gauss_fwhm: float, lorentz_fwhm: float, bg: float, center: float = 0.0):
    """Elastic baseline y(x) = Voigt(height -> area) + background."""
    area = elastic_area_from_height(height, gauss_fwhm, lorentz_fwhm)
    return elastic_voigt_y(x, area, gauss_fwhm, lorentz_fwhm, center) + float(bg)

def elastic_baseline_at_energy(energy_x: float, height: float, gauss_fwhm: float, lorentz_fwhm: float, bg: float) -> float:
    """Scalar elastic baseline value at a given energy coordinate."""
    y = elastic_baseline_y(np.array([float(energy_x)]), height, gauss_fwhm, lorentz_fwhm, bg, center=0.0)
    return float(y[0])

def dho_area_for_height_ui(height: float, phonon_energy: float, gauss_fwhm: float, lorentz_fwhm: float, damping: float, T: float, x_grid):
    """Wrap area_for_dho_voigt_height with composition/center defaults and provided preview grid."""
    return float(area_for_dho_voigt_height(float(height), float(phonon_energy), float(gauss_fwhm), float(lorentz_fwhm), float(damping), float(T), center=0.0, composition=COMPOSITION_DHO, x_grid=x_grid))

def dho_y(x, phonon_energy: float, gauss_fwhm: float, lorentz_fwhm: float, damping: float, area: float, T: float):
    """DHO+Voigt inelastic contribution for given parameters (Stokes+AntiStokes)."""
    return convolute_voigt_dho(np.asarray(x, dtype=float), float(phonon_energy), 0.0, float(gauss_fwhm), float(lorentz_fwhm), float(damping), float(area), 0.0, 0.0, T=float(T), peak=COMPOSITION_DHO)

def resolve_default_input_dir():
    """
    Determine a good default directory for the file dialog:
    - FMO_ANALYSIS_INPUT_DIR env var, if set
    - ~/Documents/Github/2025 ILL IN8 FMO Data Processing/Data Parsing/Analysis_Input
    - Path relative to this file: ../Data Parsing/Analysis_Input
    - Fallback to user's home directory
    """
    env = os.environ.get("FMO_ANALYSIS_INPUT_DIR")
    if env and os.path.isdir(env):
        return env

    base_dir = os.path.join(os.path.expanduser("~"), "Documents", "Github", "2025 ILL IN8 FMO Data Processing")
    candidate = os.path.join(base_dir, "Data Parsing", "Analysis_Input")
    if os.path.isdir(candidate):
        return candidate

    here = os.path.abspath(os.path.dirname(__file__))
    candidate2 = os.path.abspath(os.path.join(here, "..", "Data Parsing", "Analysis_Input"))
    if os.path.isdir(candidate2):
        return candidate2

    return os.path.expanduser("~")

# --- Re-used helper functions from your original script ---
def _convert_coord_token(token):
    if token is None: return 0.0
    t = token.strip().replace('p', '.').replace('m', '-')
    if all(c.isdigit() or c == '-' for c in t) and len(t.lstrip('-')) > 2 and '.' not in t:
        neg, core = t.startswith('-'), t[1:] if t.startswith('-') else t
        t = ('-' if neg else '') + core[:-2] + '.' + core[-2:]
    try: return float(t)
    except: return 0.0

def parse_filename_new_patterns(filepath):
    name = os.path.basename(filepath)
    base = os.path.splitext(name)[0]
    m1 = re.match(r'^scan_(\d+)_H([-\dpm]+)_([-\dpm]+)_([-\dpm]+)_T([-\dpm]+)K(?:_B([-\dpm]+)T)?', base)
    if m1:
        return {'scan': m1.group(1), 'H': _convert_coord_token(m1.group(2)), 'K': _convert_coord_token(m1.group(3)),
                'L': _convert_coord_token(m1.group(4)), 'T': _convert_coord_token(m1.group(5)), 'B': _convert_coord_token(m1.group(6) or '0')}
    m2 = re.match(r'^scan_(\d+)_H([-\dpm\d]+)_T([-\dpm]+)K(?:_B([-\dpm]+)T)?', base)
    if m2:
        token = (m2.group(2) or '').strip()
        if re.fullmatch(r'm?\d{3}', token):
            neg = token.startswith('m')
            digits = token[1:] if neg else token
            H = -float(int(digits[0])) if neg else float(int(digits[0]))
            K = float(int(digits[1]))
            L = float(int(digits[2]))
        else:
            H = _convert_coord_token(token)
            K = 0.0
            L = 0.0
        return {'scan': m2.group(1), 'H': H, 'K': K, 'L': L,
                'T': _convert_coord_token(m2.group(3)), 'B': _convert_coord_token(m2.group(4) or '0')}
    m3a = re.match(r'^H([-\dpm]+)_([-\dpm]+)_([-\dpm]+)_T([-\dpm]+)K(?:_B([-\dpm]+)T)?', base)
    if m3a:
        return {
            'H': _convert_coord_token(m3a.group(1)),
            'K': _convert_coord_token(m3a.group(2)),
            'L': _convert_coord_token(m3a.group(3)),
            'T': _convert_coord_token(m3a.group(4)),
            'B': _convert_coord_token(m3a.group(5) or '0'),
        }
    m3 = re.match(r'^H([-\dpm]+)(?:_K([-\dpm]+))?(?:_L([-\dpm]+))?_T([-\dpm]+)K(?:_B([-\dpm]+)T)?', base)
    if m3:
        htok = m3.group(1) or ''
        if (m3.group(2) is None) and (m3.group(3) is None) and re.fullmatch(r'\d{3}', htok):
            H = float(int(htok[0]))
            K = float(int(htok[1]))
            L = float(int(htok[2]))
        else:
            H = _convert_coord_token(htok)
            K = _convert_coord_token(m3.group(2)) if m3.group(2) is not None else 0.0
            L = _convert_coord_token(m3.group(3)) if m3.group(3) is not None else 0.0
        T = _convert_coord_token(m3.group(4))
        B = _convert_coord_token(m3.group(5) or '0')
        return {'H': H, 'K': K, 'L': L, 'T': T, 'B': B}
    return {}

def area_for_dho_voigt_height(target_height, phonon_energy, gauss_fwhm, lorentz_fwhm, damping, T, center=0.0, composition='stokes+astokes', x_grid=None):
    """
    Compute area parameter that yields the desired height at x = center + phonon_energy
    for the DHO+Voigt model.
    """
    try:
        test_area = 1.0
        if x_grid is None:
            x = np.linspace(center + phonon_energy - 5, center + phonon_energy + 5, 1200)
        else:
            x = np.asarray(x_grid, dtype=float)
        y = convolute_voigt_dho(
            x, phonon_energy, center, gauss_fwhm, lorentz_fwhm,
            damping, test_area, 0.0, 0.0, T, peak=composition
        )
        target_x = float(center + phonon_energy)
        if y is None or len(y) == 0:
            return 0.0
        idx = int(np.argmin(np.abs(x - target_x)))
        y_at = float(y[idx]) if np.isfinite(y[idx]) else 0.0
        return (test_area * (target_height / y_at)) if y_at > 0 else 0.0
    except Exception:
        return 0.0

def height_for_dho_voigt_area(area, phonon_energy, gauss_fwhm, lorentz_fwhm, damping, T, center=0.0, composition='stokes+astokes', x_grid=None):
    """
    Given an area, return the height at x = center + phonon_energy for the DHO+Voigt model.
    """
    try:
        if x_grid is None:
            x = np.linspace(center + phonon_energy - 5, center + phonon_energy + 5, 1200)
        else:
            x = np.asarray(x_grid, dtype=float)
        y = convolute_voigt_dho(
            x, phonon_energy, center, gauss_fwhm, lorentz_fwhm,
            damping, float(area), 0.0, 0.0, T, peak=composition
        )
        if y is None or len(y) == 0:
            return 0.0
        target_x = float(center + phonon_energy)
        idx = int(np.argmin(np.abs(x - target_x)))
        return float(y[idx]) if np.isfinite(y[idx]) else 0.0
    except Exception:
        return 0.0

# --- Worker for running the fit in a background thread ---
class FitWorker(QObject):
    finished = Signal()
    result = Signal(object)
    error = Signal(str)

    def __init__(self, energy, counts, errors, file_info, initial_peaks):
        super().__init__()
        self.energy = energy
        self.counts = counts
        self.errors = errors
        self.file_info = file_info
        self.initial_peaks = initial_peaks

    def run(self):
        try:
            from scipy.optimize import least_squares

            Tval = float(self.file_info.get('T', 1.5))

            el = self.file_info['elastic_params']
            g0 = float(el['gauss_fwhm'])
            l0 = float(el['lorentz_fwhm'])
            bg0 = float(el['bg'])
            elastic_h0 = float(el['height'])
            elastic_a0 = float(voigt_area_for_height(elastic_h0, g0, l0))

            pks = []
            for p in self.initial_peaks:
                e = abs(float(p['center']))
                h = max(0.0, float(p['height']))
                dmp = clamp_damping(float(p.get('damping', DHO_DAMPING_DEFAULT)))
                a = dho_area_for_height_ui(h, e, g0, l0, dmp, Tval, self.energy)
                pks.append({'E': e, 'D': dmp, 'A': a,
                            'fix_E': bool(p.get('fix_E', False)),
                            'fix_D': bool(p.get('fix_D', False)),
                            'fix_H': bool(p.get('fix_H', False))})

            n = len(pks)

            p0 = [g0, l0, bg0, elastic_a0]
            for pk in pks:
                p0 += [pk['E'], pk['D'], pk['A']]

            g_lo = max(RES_FWHM_MIN, RES_BOUND_SCALE_LO * g0)
            g_hi = min(RES_FWHM_MAX, RES_BOUND_SCALE_HI * g0 if g0 > 0 else RES_BOUND_ZERO_FALLBACK)
            l_lo = max(RES_FWHM_MIN, RES_BOUND_SCALE_LO * l0)
            l_hi = min(RES_FWHM_MAX, RES_BOUND_SCALE_HI * l0 if l0 > 0 else RES_BOUND_ZERO_FALLBACK)
            lo = [g_lo, l_lo, -np.inf, 0.0]
            hi = [g_hi, l_hi,  np.inf, np.inf]
            fix = self.file_info.get('fix_flags', {}) if isinstance(self.file_info, dict) else {}
            def _eps(v):
                try:
                    return max(EPS_ABS_TINY, EPS_REL_FIX * (abs(float(v)) + 1.0))
                except Exception:
                    return EPS_ABS_TINY
            if fix.get('gauss_fwhm', False):
                e = _eps(g0)
                lo[0] = max(1e-9, g0 - e)
                hi[0] = max(lo[0] + e, g0 + e)
            if fix.get('lorentz_fwhm', False):
                e = _eps(l0)
                lo[1] = max(1e-9, l0 - e)
                hi[1] = max(lo[1] + e, l0 + e)
            if fix.get('bg', False):
                e = _eps(bg0)
                lo[2] = bg0 - e
                hi[2] = bg0 + e if (bg0 + e) > lo[2] else (lo[2] + e)
            if fix.get('elastic_height', False):
                e = _eps(elastic_a0)
                lo[3] = max(0.0, elastic_a0 - e)
                hi[3] = elastic_a0 + e
                if not (hi[3] > lo[3]):
                    hi[3] = lo[3] + max(e, 1e-12)
            try:
                x_absmax = float(np.nanmax(np.abs(self.energy))) if self.energy is not None and len(self.energy) else np.inf
            except Exception:
                x_absmax = np.inf
            for pk in pks:
                e = pk['E']; dmp = pk['D']; a = pk['A']
                e_lo = max(ENERGY_MIN, ENERGY_SCALE_LO * e)
                e_hi = min(x_absmax, max(ENERGY_SCALE_HI * e, e + ENERGY_ABS_PAD))
                if not (e_hi > e_lo):
                    e_lo, e_hi = ENERGY_MIN, max(0.5, x_absmax)
                lo_e, hi_e = e_lo, e_hi
                lo_d, hi_d = DHO_DAMPING_MIN, DHO_DAMPING_MAX
                lo_a, hi_a = 0.0, np.inf
                if pk.get('fix_E', False):
                    ee = max(EPS_ABS_TINY, FD_REL_STEP * (abs(e) + 1.0))
                    lo_e = max(e_lo, e - ee)
                    hi_e = min(e_hi, e + ee)
                    if not (hi_e > lo_e):
                        hi_e = lo_e + ee
                if pk.get('fix_D', False):
                    ed = max(EPS_ABS_TINY, FD_REL_STEP * (abs(dmp) + 1.0))
                    lo_d = max(DHO_DAMPING_MIN, min(DHO_DAMPING_MAX, dmp - ed))
                    hi_d = min(DHO_DAMPING_MAX, max(lo_d + ed, dmp + ed))
                if pk.get('fix_H', False):
                    ea = max(EPS_ABS_TINY, FD_REL_STEP * (abs(a) + 1.0))
                    lo_a = max(0.0, a - ea)
                    hi_a = max(lo_a + ea, a + ea)
                lo += [lo_e, lo_d, lo_a]
                hi += [hi_e, hi_d, hi_a]

            xdata = self.energy
            ydata = self.counts
            w = 1.0 / np.clip(self.errors, 1e-12, np.inf)

            def model(params):
                g = float(params[0]); l = float(params[1]); bg = float(params[2]); ea = float(params[3])
                y = elastic_voigt_y(xdata, ea, g, l, center=0.0)
                idx = 4
                for i in range(n):
                    e = float(params[idx]); dmp = float(params[idx+1]); a = float(params[idx+2])
                    y += dho_y(xdata, e, g, l, dmp, a, Tval)
                    idx += 3
                y += bg
                return y

            def residuals(params):
                y = model(params)
                return (y - ydata) * w

            res = least_squares(
                residuals,
                p0,
                bounds=(lo, hi),
                loss='linear',
                max_nfev=FIT_MAX_NFEV,
                verbose=0,
            )

            popt = res.x if res.success else p0

            perr_full = None
            try:
                names = ['gauss_fwhm', 'lorentz_fwhm', 'BG', 'elastic_amplitude']
                for i in range(n):
                    names += [f'phonon_energy_{i}', f'phonon_damping_{i}', f'phonon_amplitude_{i}']
                is_fixed = []
                fix = self.file_info.get('fix_flags', {}) if isinstance(self.file_info, dict) else {}
                is_fixed.append(bool(fix.get('gauss_fwhm', False)))
                is_fixed.append(bool(fix.get('lorentz_fwhm', False)))
                is_fixed.append(bool(fix.get('bg', False)))
                is_fixed.append(bool(fix.get('elastic_height', False)))
                for i in range(n):
                    is_fixed.append(bool(pks[i].get('fix_E', False)))
                    is_fixed.append(bool(pks[i].get('fix_D', False)))
                    is_fixed.append(bool(pks[i].get('fix_H', False)))
                is_fixed = np.array(is_fixed, dtype=bool)
                lo_arr = np.asarray(lo, dtype=float)
                hi_arr = np.asarray(hi, dtype=float)
                tight = (hi_arr - lo_arr) <= (1e-12 * (np.abs(np.asarray(popt)) + 1.0))
                is_fixed = np.array(is_fixed | tight, dtype=bool)

                J = np.asarray(res.jac, dtype=float)
                m = J.shape[0]
                free_idx = np.where(~is_fixed)[0]
                dof = max(1, m - int(free_idx.size))
                s_sq = (2.0 * float(res.cost)) / dof
                perr_full = np.full(len(popt), np.nan, dtype=float)
                if free_idx.size > 0:
                    Jfree = J[:, free_idx]
                    JTJ = Jfree.T @ Jfree
                    try:
                        cov_free = s_sq * np.linalg.inv(JTJ)
                    except np.linalg.LinAlgError:
                        cov_free = s_sq * np.linalg.pinv(JTJ, rcond=1e-12)
                    diag = np.diag(cov_free)
                    diag = np.where(diag >= 0, diag, np.nan)
                    perr = np.sqrt(diag)
                    perr_full[free_idx] = perr
                try:
                    active = np.asarray(res.active_mask, dtype=int)
                    perr_full[np.where(active != 0)[0]] = np.nan
                except Exception:
                    pass
            except Exception:
                perr_full = np.full(len(popt), np.nan, dtype=float)

            g, l, bg, ea = float(popt[0]), float(popt[1]), float(popt[2]), float(popt[3])
            param_dict = {
                'gauss_fwhm': {'fit': g, 'err': float(perr_full[0]) if np.isfinite(perr_full[0]) else np.nan},
                'lorentz_fwhm': {'fit': l, 'err': float(perr_full[1]) if np.isfinite(perr_full[1]) else np.nan},
                'elastic_amplitude': {'fit': ea, 'err': float(perr_full[3]) if np.isfinite(perr_full[3]) else np.nan},
                'BG': {'fit': bg, 'err': float(perr_full[2]) if np.isfinite(perr_full[2]) else np.nan},
            }
            idx = 4
            for i in range(n):
                e = float(popt[idx]); dmp = float(popt[idx+1]); a = float(popt[idx+2])
                err_e = float(perr_full[idx]) if np.isfinite(perr_full[idx]) else np.nan
                err_d = float(perr_full[idx+1]) if np.isfinite(perr_full[idx+1]) else np.nan
                err_a = float(perr_full[idx+2]) if np.isfinite(perr_full[idx+2]) else np.nan
                param_dict[f'phonon_energy_{i}'] = {'fit': e, 'err': err_e}
                param_dict[f'phonon_amplitude_{i}'] = {'fit': a, 'err': err_a}
                param_dict[f'phonon_damping_{i}'] = {'fit': dmp, 'err': err_d}
                idx += 3

            self.result.emit(param_dict)
        except Exception as e:
            self.error.emit(f"An error occurred during fitting: {e}")
        finally:
            self.finished.emit()

# --- Main Application Window ---
class PeakViewBox(pg.ViewBox):
    """Custom ViewBox to manage mouse drag behavior for phonon peaks.
    - When a phonon is selected, left-drag moves the peak (no panning).
    - When no peak is selected, default panning/zoom behavior is used.
    - Dragging ends cleanly on mouse release.
    - Exclude mode toggles nearest point on drag start.
    """
    def __init__(self, host, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.host = host
        self._dragging_peak = False
        self._drag_obj = None
        # Exclude-mode drag box state
        self._exclude_active = False
        self._exclude_start = None  # (x0, y0) in view/data coords
        self._exclude_rect = None   # QGraphicsRectItem overlay in data coords

    def mouseClickEvent(self, ev):
        try:
            if ev.button() == LeftButton:
                pos_v = self.mapSceneToView(ev.scenePos())
                x, y = float(pos_v.x()), float(pos_v.y())
                # Exclude mode: toggle nearest point on click
                if getattr(self.host, 'exclude_mode', False):
                    self.host._toggle_nearest_point_exclusion_xy(x, y)
                    ev.accept()
                    return
                # Selection by click
                kind, obj = self.host._nearest_target_xy(x, y)
                if kind is not None:
                    self.host.set_selected(kind, obj)
                    ev.accept()
                    return
                # Clicked empty space: clear selection to allow panning next drag
                self.host.set_selected(None, None)
        except Exception:
            pass
        # Fallback to default behavior
        super().mouseClickEvent(ev)

    def mouseDragEvent(self, ev, axis=None):
        try:
            # While in exclude mode, consume drags to avoid panning/zoom and support box selection
            if getattr(self.host, 'exclude_mode', False):
                # Left-drag draws/updates the selection box, other drags are ignored to keep view fixed
                if ev.button() == LeftButton:
                    pos_v = self.mapSceneToView(ev.scenePos())
                    x = float(pos_v.x()); y = float(pos_v.y())
                    if ev.isStart():
                        # Begin rubber-band rectangle in data coordinates
                        self._exclude_active = True
                        self._exclude_start = (x, y)
                        # Create rect overlay in data coords
                        try:
                            RectItemCls = getattr(QtWidgets, 'QGraphicsRectItem', None) or getattr(QtGui, 'QGraphicsRectItem', None)
                            if RectItemCls is not None:
                                self._exclude_rect = RectItemCls()
                                pen = pg.mkPen((255, 140, 0), width=2, style=DashLine)
                                brush = pg.mkBrush(255, 165, 0, 50)
                                try:
                                    self._exclude_rect.setPen(pen)
                                    self._exclude_rect.setBrush(brush)
                                    self._exclude_rect.setZValue(1e6)
                                except Exception:
                                    pass
                                # Parent into data transform space
                                try:
                                    if hasattr(self, 'childGroup') and self.childGroup is not None:
                                        self._exclude_rect.setParentItem(self.childGroup)
                                    else:
                                        # Fallback addItem
                                        self.addItem(self._exclude_rect)
                                except Exception:
                                    pass
                        except Exception:
                            self._exclude_rect = None
                        ev.accept()
                        return
                    # Update rectangle while dragging
                    if self._exclude_active and self._exclude_start is not None:
                        try:
                            x0, y0 = self._exclude_start
                            x1, y1 = x, y
                            rx0, rx1 = (x0, x1) if x0 <= x1 else (x1, x0)
                            ry0, ry1 = (y0, y1) if y0 <= y1 else (y1, y0)
                            if self._exclude_rect is not None:
                                r = QtCore.QRectF(rx0, ry0, rx1 - rx0, ry1 - ry0)
                                self._exclude_rect.setRect(r)
                        except Exception:
                            pass
                        ev.accept()
                        if ev.isFinish():
                            # Toggle inclusion for points within the box
                            try:
                                rx0, rx1 = (self._exclude_start[0], x) if self._exclude_start[0] <= x else (x, self._exclude_start[0])
                                ry0, ry1 = (self._exclude_start[1], y) if self._exclude_start[1] <= y else (y, self._exclude_start[1])
                                en = getattr(self.host, 'energy', None)
                                ct = getattr(self.host, 'counts', None)
                                if en is not None and ct is not None:
                                    en = np.asarray(en, dtype=float)
                                    ct = np.asarray(ct, dtype=float)
                                    sel = (en >= rx0) & (en <= rx1) & (ct >= ry0) & (ct <= ry1)
                                    if sel.any():
                                        if not isinstance(self.host.excluded_mask, np.ndarray) or len(self.host.excluded_mask) != len(en):
                                            self.host.excluded_mask = np.zeros(len(en), dtype=bool)
                                        self.host.excluded_mask[sel] = ~self.host.excluded_mask[sel]
                                        self.host._update_data_plot(do_range=False)
                                        self.host.update_previews()
                            except Exception:
                                pass
                            # Cleanup overlay
                            try:
                                if self._exclude_rect is not None:
                                    try:
                                        # Remove from viewbox/scene
                                        if hasattr(self, 'removeItem'):
                                            self.removeItem(self._exclude_rect)
                                        else:
                                            pr = self._exclude_rect.parentItem()
                                            if pr is not None:
                                                pr.removeFromGroup(self._exclude_rect)
                                    except Exception:
                                        pass
                                    self._exclude_rect = None
                            except Exception:
                                pass
                            self._exclude_active = False
                            self._exclude_start = None
                        return
                else:
                    # In exclude mode, consume other drag gestures to keep the view fixed
                    ev.accept()
                    return
            if ev.button() == LeftButton:
                if ev.isStart():
                    pos_v = self.mapSceneToView(ev.scenePos())
                    x, y = float(pos_v.x()), float(pos_v.y())
                    # Exclude mode: toggle nearest on drag start, no panning
                    if getattr(self.host, 'exclude_mode', False):
                        # Already handled above; but if we get here, consume to avoid panning
                        ev.accept()
                        return
                    # Only start dragging if a phonon is already selected; otherwise pan
                    if self.host.selected_kind == 'phonon' and self.host.selected_obj is not None:
                        self._dragging_peak = True
                        self._drag_obj = self.host.selected_obj
                        ev.accept()
                        return
                # While dragging
                if self._dragging_peak and self._drag_obj is not None:
                    pos_v = self.mapSceneToView(ev.scenePos())
                    x = float(pos_v.x())
                    # Update center of active peak
                    try:
                        self._drag_obj['center'] = x
                        w = self._drag_obj.get('widgets', {}).get('center_spin')
                        if w is not None:
                            was = w.blockSignals(True)
                            w.setValue(x)
                            w.blockSignals(was)
                        # Invalidate cached area
                        if 'area' in self._drag_obj:
                            self._drag_obj.pop('area', None)
                    except Exception:
                        pass
                    self.host.update_previews()
                    ev.accept()
                    if ev.isFinish():
                        self._dragging_peak = False
                        self._drag_obj = None
                    return
        except Exception:
            pass
        # Default behavior (panning/zoom) when not dragging a peak
        super().mouseDragEvent(ev, axis=axis)

    def wheelEvent(self, ev, axis=None):
        """Block wheel zoom when a selection is active; otherwise, default behavior."""
        try:
            # Keep view fixed while excluding points
            if getattr(self.host, 'exclude_mode', False):
                ev.accept()
                return
            if getattr(self.host, 'selected_kind', None) is not None:
                ev.accept()
                return
        except Exception:
            pass
        super().wheelEvent(ev, axis=axis)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Interactive Peak Fitter (PyQtGraph)")

        # --- Data Storage ---
        self.energy = None
        self.counts = None
        self.errors = None
        self.file_info = {}
        self.draggable_artists = []
        self.active_artist = None

        # PyQtGraph plot items
        self.data_included_item = None
        self.data_excluded_item = None
        self.preview_line = None
        self.elastic_line = None
        self.background_line = None
        self.phonon_lines = []
        self.phonon_markers = []
        
        # Residuals state
        self.resid_bars = None
        self.resid_zero_line = None

        # Selection state
        self.selected_kind = None
        self.selected_obj = None

        # --- Main Layout ---
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)

        # --- Controls Panel (Left Side) ---
        controls_widget = QWidget()
        controls_layout = QVBoxLayout(controls_widget)
        try:
            controls_widget.setMinimumWidth(MIN_LEFT_PANEL_WIDTH)
            controls_widget.setMaximumWidth(MAX_LEFT_PANEL_WIDTH)
        except Exception:
            pass

        # File buttons
        self.load_button = QPushButton("Load Files")
        self.load_button.clicked.connect(self.load_data)
        
        # File list for multi-selection navigation
        self.file_list_paths = []
        try:
            controls_layout.addWidget(QLabel("Files:"))
        except Exception:
            pass
        self.file_list_widget = QListWidget()
        try:
            self.file_list_widget.setSelectionMode(QAbstractItemView.SingleSelection)
            self.file_list_widget.setSelectionBehavior(QAbstractItemView.SelectItems)
            self.file_list_widget.setFocusPolicy(Qt.StrongFocus)
        except Exception:
            pass
        try:
            self.file_list_widget.currentRowChanged.connect(self.on_file_selected)
            self.file_list_widget.itemClicked.connect(self.on_file_item_clicked)
            self.file_list_widget.currentItemChanged.connect(self.on_current_item_changed)
            self.file_list_widget.itemSelectionChanged.connect(self.on_file_selection_changed)
            self.file_list_widget.itemActivated.connect(self.on_file_item_clicked)
            self.file_list_widget.selectionModel().currentChanged.connect(self.on_model_current_changed)
        except Exception:
            pass
        controls_layout.addWidget(self.file_list_widget)
        
        # File list management buttons
        filelist_btn_row = QWidget()
        filelist_btn_layout = QHBoxLayout(filelist_btn_row)
        filelist_btn_layout.setContentsMargins(0, 0, 0, 0)
        filelist_btn_layout.setSpacing(6)
        self.remove_file_btn = QPushButton("Remove Selected")
        self.clear_filelist_btn = QPushButton("Clear List")
        self.add_to_list_btn = QPushButton("Add to List")
        self.remove_file_btn.clicked.connect(self.remove_selected_file_from_list)
        self.clear_filelist_btn.clicked.connect(self.clear_file_list)
        self.add_to_list_btn.clicked.connect(self.add_files_to_list)
        filelist_btn_layout.addWidget(self.remove_file_btn)
        filelist_btn_layout.addWidget(self.add_to_list_btn)
        filelist_btn_layout.addWidget(self.clear_filelist_btn)
        controls_layout.addWidget(filelist_btn_row)
        
        # Peak buttons
        self.add_peak_button = QPushButton("Add Phonon Peak")
        self.add_peak_button.clicked.connect(self.add_peak)
        self.fit_button = QPushButton("Fit Data")
        self.fit_button.clicked.connect(self.run_fit)
        self.clear_button = QPushButton("Clear Peaks")
        self.clear_button.clicked.connect(self.clear_peaks)
        self.fit_button.setEnabled(False)
        self.add_peak_button.setEnabled(False)
        self.clear_button.setEnabled(False)

        # Parameter inputs
        form_layout = QFormLayout()
        self.gauss_fwhm_spinbox = QDoubleSpinBox(value=DEFAULT_GAUSS_FWHM, minimum=RES_FWHM_MIN, maximum=RES_FWHM_MAX, singleStep=0.05, decimals=3)
        self.lorentz_fwhm_spinbox = QDoubleSpinBox(value=DEFAULT_LORENTZ_FWHM, minimum=RES_FWHM_MIN, maximum=RES_FWHM_MAX, singleStep=0.05, decimals=3)
        self.bg_spinbox = QDoubleSpinBox(value=DEFAULT_BG, minimum=-1e6, maximum=1e6, singleStep=10)
        self.elastic_height_spinbox = QDoubleSpinBox(value=DEFAULT_ELASTIC_HEIGHT, minimum=0, maximum=1e9, singleStep=100)
        
        # Fix toggles
        self.fix_gauss_cb = QCheckBox("Fix")
        self.fix_lorentz_cb = QCheckBox("Fix")
        self.fix_bg_cb = QCheckBox("Fix")
        self.fix_el_height_cb = QCheckBox("Fix")
        
        # Row widgets
        row_gauss = QWidget(); row_gauss_l = QHBoxLayout(row_gauss); row_gauss_l.setContentsMargins(0,0,0,0); row_gauss_l.addWidget(self.gauss_fwhm_spinbox); row_gauss_l.addWidget(self.fix_gauss_cb)
        row_lorentz = QWidget(); row_lorentz_l = QHBoxLayout(row_lorentz); row_lorentz_l.setContentsMargins(0,0,0,0); row_lorentz_l.addWidget(self.lorentz_fwhm_spinbox); row_lorentz_l.addWidget(self.fix_lorentz_cb)
        row_bg = QWidget(); row_bg_l = QHBoxLayout(row_bg); row_bg_l.setContentsMargins(0,0,0,0); row_bg_l.addWidget(self.bg_spinbox); row_bg_l.addWidget(self.fix_bg_cb)
        row_el = QWidget(); row_el_l = QHBoxLayout(row_el); row_el_l.setContentsMargins(0,0,0,0); row_el_l.addWidget(self.elastic_height_spinbox); row_el_l.addWidget(self.fix_el_height_cb)
        form_layout.addRow("Gauss FWHM:", row_gauss)
        form_layout.addRow("Lorentz FWHM:", row_lorentz)
        form_layout.addRow("Background:", row_bg)
        form_layout.addRow("Elastic Height:", row_el)
        
        # Store defaults for reset
        self._default_params = {
            'gauss': DEFAULT_GAUSS_FWHM,
            'lorentz': DEFAULT_LORENTZ_FWHM,
            'bg': DEFAULT_BG,
            'elastic_height': DEFAULT_ELASTIC_HEIGHT,
        }

        # Wire spinboxes to live preview updates
        self.gauss_fwhm_spinbox.valueChanged.connect(self.update_previews)
        self.lorentz_fwhm_spinbox.valueChanged.connect(self.update_previews)
        self.bg_spinbox.valueChanged.connect(self.update_previews)
        self.elastic_height_spinbox.valueChanged.connect(self.update_previews)

        # Chi-squared display
        self.chi2_label = QLabel("Chi^2: -")
        controls_layout.addWidget(self.chi2_label)

        # Selection status
        self.selection_label = QLabel("Selected: None")
        controls_layout.addWidget(self.load_button)
        controls_layout.addWidget(self.add_peak_button)
        controls_layout.addWidget(self.clear_button)
        
        # Reset preview button
        self.reset_button = QPushButton("Reset")
        self.reset_button.setEnabled(True)
        self.reset_button.clicked.connect(self.reset_preview_defaults)
        controls_layout.addWidget(self.reset_button)
        
        # Exclude points toggle button
        self.exclude_button = QPushButton("Exclude Points")
        self.exclude_button.setCheckable(True)
        self.exclude_button.toggled.connect(self.on_toggle_exclude_mode)
        controls_layout.addWidget(self.exclude_button)
        controls_layout.addLayout(form_layout)
        controls_layout.addWidget(self.selection_label)
        
        # Save Fit button
        self.save_button = QPushButton("Save Fit")
        self.save_button.setEnabled(False)
        self.save_button.clicked.connect(self.save_fit)
        controls_layout.addWidget(self.fit_button)
        controls_layout.addWidget(self.save_button)
        controls_layout.addWidget(QLabel("Fit Results:"))
        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        controls_layout.addWidget(self.results_text)

        # Peaks parameters panel
        controls_layout.addWidget(QLabel("Peaks:"))
        self.peaks_panel = QWidget()
        self.peaks_layout = QVBoxLayout(self.peaks_panel)
        self.peaks_layout.setContentsMargins(0, 0, 0, 0)
        self.peaks_layout.setSpacing(6)
        controls_layout.addWidget(self.peaks_panel)
        
        # --- PyQtGraph Plot (Right Side) ---
        self.plot_container = QWidget()
        plot_vbox = QVBoxLayout(self.plot_container)
        try:
            plot_vbox.setContentsMargins(0, 0, 0, 0)
            plot_vbox.setSpacing(4)
        except Exception:
            pass
        self.header_label = QLabel("")
        try:
            self.header_label.setTextInteractionFlags(TextSelectableByMouse)
            self.header_label.setAlignment(AlignLeft | AlignVCenter)
            self.header_label.setMinimumHeight(18)
        except Exception:
            pass
        plot_vbox.addWidget(self.header_label)
        
        # Create main plot widget with custom ViewBox to control drag/pan behavior
        self.vb = PeakViewBox(host=self)
        self.plot_widget = PlotWidget(viewBox=self.vb)
        self.plot_widget.setBackground('w')
        self.plot_widget.setLabel('left', 'Counts')
        self.plot_widget.setLabel('bottom', 'Energy (meV)')
        self.plot_widget.showGrid(x=True, y=True, alpha=0.3)
        
        # CRITICAL: Disable aspect ratio locking so data fills the plot area
        vb = self.plot_widget.getViewBox()
        if vb is not None:
            vb.setAspectLocked(False)
        
        # Add legend inside the PlotItem (overlay), not as a separate layout column
        try:
            pi = self.plot_widget.getPlotItem()
            if getattr(pi, 'legend', None) is None:
                # Anchor legend inside the ViewBox to avoid altering layout columns
                pi.addLegend(offset=(10, 10))
        except Exception:
            pass
        
        # Enable mouse interaction
        self.plot_widget.setMouseEnabled(x=True, y=True)
        
        plot_vbox.addWidget(self.plot_widget, 1)
        
        # Create residuals plot widget
        self.resid_widget = PlotWidget()
        self.resid_widget.setBackground('w')
        self.resid_widget.setLabel('left', 'Residuals (sigma)')
        self.resid_widget.showGrid(x=False, y=True, alpha=0.3)
        self.resid_widget.setMaximumHeight(150)
        
        plot_vbox.addWidget(self.resid_widget)
        
        # Connect plot events
        self.connect_plot_events()

        # Global shortcuts (work regardless of widget focus)
        try:
            self._sc_exclude_D = QtWidgets.QShortcut(QtGui.QKeySequence('D'), self)
            self._sc_exclude_D.setContext(QtCore.Qt.ApplicationShortcut)
            self._sc_exclude_D.activated.connect(self._toggle_exclude_shortcut)
            # Ensure lower-case also works on some platforms
            self._sc_exclude_d = QtWidgets.QShortcut(QtGui.QKeySequence('d'), self)
            self._sc_exclude_d.setContext(QtCore.Qt.ApplicationShortcut)
            self._sc_exclude_d.activated.connect(self._toggle_exclude_shortcut)
        except Exception:
            pass

        # Pre-create help panel container
        self.help_widget = QWidget()

        # Build splitter layout
        right_splitter = QSplitter(Horizontal)
        right_splitter.addWidget(self.plot_container)
        right_splitter.addWidget(self.help_widget)
        try:
            right_splitter.setStretchFactor(0, 1)
            right_splitter.setStretchFactor(1, 0)
        except Exception:
            pass
        # Keep a reference for later sizing when toggling help
        self.right_splitter = right_splitter
            
        main_splitter = QSplitter(Horizontal)
        main_splitter.addWidget(controls_widget)
        main_splitter.addWidget(right_splitter)
        try:
            main_splitter.setStretchFactor(0, 0)
            main_splitter.setStretchFactor(1, 1)
            main_splitter.setSizes([420, 980])
            try:
                help_w = int(getattr(self, '_help_panel_width', DEFAULT_HELP_PANEL_WIDTH))
            except Exception:
                help_w = DEFAULT_HELP_PANEL_WIDTH
            right_splitter.setSizes([DEFAULT_RIGHT_PLOT_SIZE, help_w])
        except Exception:
            pass
        # Keep reference to main splitter too (optional)
        self.main_splitter = main_splitter
        main_layout.addWidget(main_splitter)
        
        # --- Help/Instructions Sidebar (Right Side) ---
        help_layout = QVBoxLayout(self.help_widget)
        help_layout.setContentsMargins(8, 8, 8, 8)
        help_layout.setSpacing(6)
        help_layout.addWidget(QLabel("Help & Shortcuts"))
        self.help_text = QTextEdit()
        self.help_text.setReadOnly(True)
        self.help_text.setPlainText(
            "Workflow:\n"
            "- Load Files: choose one or more .dat files. The list on the left lets you navigate. Use Add to List / Remove Selected / Clear List as needed. Selecting a file loads it.\n"
            "- Plot: included data (black) with error bars; excluded points show as gray X. Residuals (model - data)/sigma are shown below and share the X-axis.\n"
            "- Parameters (left): Elastic Height, Background, and resolution FWHMs (Gaussian/Lorentzian).\n"
            "- Peaks: Click Add Phonon Peak to create a phonon. Each peak has Center (meV), Height, Damping, and Fix toggles.\n"
            "\n"
            "Mouse/Keyboard:\n"
            "- Select: click near a phonon marker (X) or the elastic baseline; Space clears selection.\n"
            "- Move peak: with a phonon selected, left-click then drag horizontally to change Center.\n"
            "- Mouse wheel on selected phonon: Height (no modifiers); Shift + Wheel: Damping.\n"
            "- Global resolution via wheel: Ctrl + Wheel = Gaussian FWHM; Ctrl + Shift + Wheel = Lorentzian FWHM.\n"
            "- Elastic-specific (when Elastic is selected, no Ctrl): Shift + Wheel = Gaussian FWHM; Alt + Wheel = Lorentzian FWHM.\n"
            "- Zoom/Pan: mouse wheel to zoom; left-drag to pan; right-drag for box-zoom (pyqtgraph defaults). Press R to reset view.\n"
            "  Note: while a selection is active, the wheel controls parameters; press Space to clear selection to zoom.\n"
            "- Selection keys: E selects elastic; numbers 1-9 select phonons; Tab/Right/Left cycle selection.\n"
            "\n"
            "Excluding Points:\n"
            "- Toggle Exclude Points (button or D), then click near points to include/exclude. Residuals and Chi^2 update live.\n"
            "\n"
            "Fitting:\n"
            "- Fit Data performs a non-linear least-squares fit: Elastic Voigt + BG + sum of DHO phonons (convolved with resolution).\n"
            "- Fix checkboxes keep parameters near their current values during the fit.\n"
            "- Results show best-fit values with uncertainties; applied only if reduced Chi^2 improves.\n"
            "\n"
            "Saving/Loading:\n"
            "- Save Fit writes parameters (and uncertainties when available) plus ExcludedIndices to the Results directory and exports a PNG to the Figures directory.\n"
            "- When reloading a file, any previous fit/exclusion is auto-applied if a results file exists.\n"
            "- Configure directories in the Paths section; Input defaults can be set via FMO_ANALYSIS_INPUT_DIR.\n"
            "\n"
            "Hints:\n"
            "- Legend order: Data → Excluded → Background → Elastic → Phonon i → Total.\n"
            "- Reset restores default preview parameters and clears peaks (does not unload data).\n"
            "- Show/Hide Help toggles this panel.\n"
        )
        help_layout.addWidget(self.help_text)
        
        # Paths configuration UI
        try:
            self._init_paths_ui(help_layout)
        except Exception:
            pass

        # Help toggle button in left panel
        self.help_toggle_button = QPushButton("Hide Help")
        self.help_toggle_button.clicked.connect(self.toggle_help_panel)
        controls_layout.addWidget(self.help_toggle_button)
        
        # Hide help panel by default
        try:
            self._help_panel_width = max(DEFAULT_HELP_PANEL_WIDTH + 40, int(self.help_widget.sizeHint().width()))
            try:
                self.help_widget.setMinimumWidth(self._help_panel_width)
                self.help_widget.setMaximumWidth(self._help_panel_width)
            except Exception:
                pass
            self.help_widget.setVisible(False)
            self.help_toggle_button.setText("Show Help")
            # When hidden by default, give all width to the plot area
            try:
                sizes = self.right_splitter.sizes() if hasattr(self, 'right_splitter') else None
                if sizes and len(sizes) >= 2:
                    self.right_splitter.setSizes([sum(sizes), 0])
            except Exception:
                pass
        except Exception:
            pass
            
        self.resize(DEFAULT_WINDOW_WIDTH, DEFAULT_WINDOW_HEIGHT)
        
        # Initialize runtime state
        self.x_preview = None
        self._mods = {'ctrl': False, 'shift': False, 'alt': False}
        self.last_fit_params = None
        
        # Exclusion state
        self.excluded_mask = None
        self.exclude_mode = False
        
        # Box selection state for exclusions
        self._box_select_active = False
        self._box_start = None
        self._box_rect = None
        
        # Zoom box state (Alt+drag)
        self._zoom_box_active = False
        self._zoom_start = None
        self._zoom_rect = None
        
        # Pan state
        self._pan_active = False
        self._pan_start = None
        self._pan_xlim0 = None
        self._pan_ylim0 = None
        self._pan_start_px = None
        self._pan_dx_per_px = None
        self._pan_dy_per_px = None
        
        # Debounce for file-list selection changes
        self._pending_row = None
        self._load_row_timer = QTimer(self)
        try:
            self._load_row_timer.setSingleShot(True)
            self._load_row_timer.timeout.connect(self._do_load_pending_row)
        except Exception:
            pass

        # Guard/debounce for range-changed updates to avoid feedback loops
        self._in_preview_update = False
        self._range_update_timer = QTimer(self)
        try:
            self._range_update_timer.setSingleShot(True)
            self._range_update_timer.setInterval(50)
            self._range_update_timer.timeout.connect(lambda: self.update_previews())
        except Exception:
            pass

        # Try to restore last saved file list
        try:
            self._load_saved_file_list()
        except Exception:
            pass
            
        # Initialize header line
        try:
            self._update_header_info()
        except Exception:
            pass

        # Install event filter for mouse wheel and key handling on the viewbox
        try:
            self.plot_widget.viewport().installEventFilter(self)
        except Exception:
            pass
        # Also install a global event filter so 'd' works regardless of focus
        try:
            app_inst = QApplication.instance()
            if app_inst is not None:
                app_inst.installEventFilter(self)
        except Exception:
            pass
        # Flag to ensure we auto-range once after the first preview is computed for a new file
        self._need_initial_autorange = False

    # --- ViewBox helpers (ensure all drawable items live in data coordinates) ---
    def _get_main_viewbox(self):
        try:
            return self.plot_widget.getViewBox()
        except Exception:
            return None

    def _vb_add(self, item):
        """Add a graphics item to the ViewBox (data coordinates)."""
        if item is None:
            return
        # Prefer ViewBox so item lands under childGroup and inherits data transform
        try:
            vb = self.plot_widget.getViewBox()
            if vb is not None:
                vb.addItem(item)
                try:
                    # Explicitly ensure parenting to childGroup for data transforms
                    if hasattr(vb, 'childGroup') and item.parentItem() is not vb.childGroup:
                        try:
                            # Some versions require grouping instead of setParentItem
                            vb.childGroup.addToGroup(item)
                        except Exception:
                            item.setParentItem(vb.childGroup)
                    # Ensure parenting succeeds; no debug output
                except Exception:
                    pass
                return
        except Exception:
            pass
        # Fallback to PlotItem
        try:
            pi = self.plot_widget.getPlotItem()
            if pi is not None:
                pi.addItem(item)
        except Exception:
            pass

    def _vb_remove(self, item):
        """Remove a graphics item from the ViewBox/PlotItem."""
        if item is None:
            return
        # Prefer removing via ViewBox
        try:
            vb = self.plot_widget.getViewBox()
            if vb is not None:
                vb.removeItem(item)
                return
        except Exception:
            pass
        # Fallback to PlotItem
        try:
            pi = self.plot_widget.getPlotItem()
            if pi is not None:
                pi.removeItem(item)
        except Exception:
            pass

    # --- Data helpers ---
    def _get_included_data(self):
        if self.energy is None or self.counts is None:
            return None, None, None
        try:
            base_errs = self.errors if self.errors is not None else np.sqrt(np.clip(np.abs(self.counts), 1e-12, np.inf))
            if isinstance(self.excluded_mask, np.ndarray) and len(self.excluded_mask) == len(self.energy):
                inc = ~self.excluded_mask
                x = np.asarray(self.energy, dtype=float)[inc]
                y = np.asarray(self.counts, dtype=float)[inc]
                e = np.asarray(base_errs, dtype=float)[inc]
            else:
                x = np.asarray(self.energy, dtype=float)
                y = np.asarray(self.counts, dtype=float)
                e = np.asarray(base_errs, dtype=float)
            e[~np.isfinite(e)] = np.nan
            if not np.all(np.isfinite(e)):
                finite = np.isfinite(e)
                e[~finite] = np.nanmedian(e[finite]) if np.any(finite) else 1.0
            e = np.clip(e, 1e-9, np.inf)
            return x, y, e
        except Exception:
            return np.asarray(self.energy, dtype=float), np.asarray(self.counts, dtype=float), (self.errors if self.errors is not None else np.sqrt(np.clip(np.abs(self.counts), 1e-12, np.inf)))

    # [Additional methods would follow - I'll continue in the next part due to length]
    # The rest of the methods need to be converted to use PyQtGraph APIs...

    def _default_base_dir(self):
        try:
            return os.path.join(os.path.expanduser('~'), 'Documents', 'Github', '2025 ILL IN8 FMO Data Processing')
        except Exception:
            return os.path.expanduser('~')

    def _get_config_dir_default(self):
        try:
            here = os.path.abspath(os.path.dirname(__file__))
        except Exception:
            here = os.getcwd()
        return os.path.join(here, 'config')

    def _load_user_dirs(self):
        path = os.path.join(self._get_config_dir_default(), 'user_paths.json')
        try:
            if os.path.isfile(path):
                with open(path, 'r', encoding='utf-8') as f:
                    d = json.load(f)
                if isinstance(d, dict):
                    return d
        except Exception:
            pass
        return {}

    def _save_user_dirs(self):
        try:
            cfg_dir = self._get_config_dir()
            os.makedirs(cfg_dir, exist_ok=True)
            path = os.path.join(cfg_dir, 'user_paths.json')
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(self.user_dirs, f, indent=2)
        except Exception:
            pass

    def _init_paths_ui(self, layout: QVBoxLayout):
        self.user_dirs = self._load_user_dirs()
        base = self._default_base_dir()
        defaults = {
            'input_dir': resolve_default_input_dir(),
            'results_dir': os.path.join(base, 'FMO Results'),
            'figures_dir': os.path.join(base, 'FMO Figures'),
            'config_dir': self._get_config_dir_default(),
        }
        for k, v in defaults.items():
            if not self.user_dirs.get(k):
                self.user_dirs[k] = v

        layout.addWidget(QLabel('Paths'))
        self.path_edits = {}

        def add_path_row(title, key):
            row = QWidget(); hl = QHBoxLayout(row); hl.setContentsMargins(0,0,0,0); hl.setSpacing(6)
            lab = QLabel(title)
            edit = QLineEdit()
            try:
                edit.setPlaceholderText(self.user_dirs.get(key, ''))
                edit.setText(self.user_dirs.get(key, ''))
            except Exception:
                pass
            btn = QPushButton('Browse')

            def on_browse():
                start = edit.text().strip() or self.user_dirs.get(key) or os.path.expanduser('~')
                d = QFileDialog.getExistingDirectory(self, f"Select {title}", start)
                if d:
                    edit.setText(d)
                    self._on_path_changed(key, d)

            def on_edit_finished():
                self._on_path_changed(key, edit.text().strip())

            btn.clicked.connect(on_browse)
            edit.editingFinished.connect(on_edit_finished)
            hl.addWidget(lab)
            hl.addWidget(edit, 1)
            hl.addWidget(btn)
            layout.addWidget(row)
            self.path_edits[key] = edit

        add_path_row('Input Dir', 'input_dir')
        add_path_row('Results Dir', 'results_dir')
        add_path_row('Figures Dir', 'figures_dir')
        add_path_row('Config Dir', 'config_dir')

        self._save_user_dirs()

    def _on_path_changed(self, key, value):
        if not isinstance(key, str):
            return
        val = (value or '').strip()
        if not val:
            return
        try:
            if key in ('results_dir', 'figures_dir', 'config_dir'):
                os.makedirs(val, exist_ok=True)
        except Exception:
            pass
        self.user_dirs[key] = val
        self._save_user_dirs()

    def _get_config_dir(self):
        try:
            if hasattr(self, 'user_dirs') and isinstance(self.user_dirs, dict):
                d = self.user_dirs.get('config_dir', None)
                if d:
                    try:
                        os.makedirs(d, exist_ok=True)
                    except Exception:
                        pass
                    return d
        except Exception:
            pass
        cfg_dir = self._get_config_dir_default()
        try:
            os.makedirs(cfg_dir, exist_ok=True)
        except Exception:
            pass
        return cfg_dir

    # Placeholder for additional conversion methods - this is a substantial undertaking
    # that requires converting all matplotlib-specific code to PyQtGraph equivalents
    
    def connect_plot_events(self):
        """Connect PyQtGraph plot events."""
        # Mouse interactions are handled by the custom ViewBox.
        # Keyboard events on the main window
        try:
            self.plot_widget.keyPressEvent = self.on_key_press
            self.plot_widget.keyReleaseEvent = self.on_key_release
        except Exception:
            pass
        # Drag state (managed by ViewBox)
        self._dragging = False
        self._drag_artist = None
        
    def on_mouse_click(self, event):
        """Handle mouse click events in PyQtGraph."""
        try:
            mouse_evt = event
            pos = mouse_evt.scenePos()
            vb = self.plot_widget.getViewBox()
            if vb is None:
                return
            if not vb.sceneBoundingRect().contains(pos):
                return
            mousePoint = vb.mapSceneToView(pos)
            x = float(mousePoint.x())
            y = float(mousePoint.y())
            # Exclude mode handled by ViewBox; keep selection update on simple clicks
            if self.exclude_mode:
                self._toggle_nearest_point_exclusion_xy(x, y)
                return
            kind, obj = self._nearest_target_xy(x, y)
            if kind is not None:
                self.set_selected(kind, obj)
            else:
                self.set_selected(None, None)
        except Exception:
            pass
        
    def on_mouse_move(self, event):
        """Handle mouse move events in PyQtGraph."""
        # Dragging is handled by the custom ViewBox; no-op here to avoid conflicts
        return
        # Dragging is handled by the custom ViewBox; no-op here to avoid conflicts
        return
        
    def on_range_changed(self, viewbox, range_info):
        """Handle zoom/pan changes."""
        try:
            # Do not trigger preview recompute from range changes during exclusion edits
            if getattr(self, 'exclude_mode', False):
                return
            # Avoid re-entrant storms while we are actively updating previews
            if getattr(self, '_in_preview_update', False):
                return
            # Debounce preview updates during continuous zoom/pan
            if hasattr(self, '_range_update_timer') and self._range_update_timer is not None:
                try:
                    self._range_update_timer.start()
                except Exception:
                    self.update_previews()
            else:
                self.update_previews()
        except Exception:
            pass

    # --- Qt event filter for wheel ---
    def eventFilter(self, obj, ev):
        try:
            from pyqtgraph.Qt import QtGui as _QtGui
            QWheelEvent = getattr(_QtGui, 'QWheelEvent', None)
            # Global key handling: ensure 'd' toggles exclude regardless of focus
            if ev is not None and getattr(ev, 'type', None) is not None:
                # Also catch ShortcutOverride to ensure shortcuts win over focused editors
                try:
                    et = ev.type()
                    QT = getattr(QtCore, 'QEvent')
                    sc_types = [getattr(QT, 'ShortcutOverride', None)]
                    QT_Type = getattr(QT, 'Type', None)
                    if QT_Type is not None:
                        sc_types.append(getattr(QT_Type, 'ShortcutOverride', None))
                    is_shortcut_override = any(t is not None and et == t for t in sc_types)
                except Exception:
                    is_shortcut_override = False
                if is_shortcut_override:
                    try:
                        key = ev.key() if hasattr(ev, 'key') else None
                    except Exception:
                        key = None
                    try:
                        mods = ev.modifiers()
                    except Exception:
                        mods = 0
                    key_d = (getattr(Qt, 'Key_D', None), getattr(getattr(Qt, 'Key', object), 'Key_D', None), ord('D'), ord('d'))
                    if key in key_d and not (mods & (ControlModifier | ShiftModifier | AltModifier)):
                        try:
                            self.exclude_button.setChecked(not self.exclude_button.isChecked())
                        except Exception:
                            self.on_toggle_exclude_mode(not getattr(self, 'exclude_mode', False))
                        try:
                            ev.accept()
                        except Exception:
                            pass
                        return True
                try:
                    et = ev.type()
                    # Support both Qt5/Qt6 enums
                    QT = getattr(QtCore, 'QEvent')
                    types = [getattr(QT, 'KeyPress', None)]
                    QT_Type = getattr(QT, 'Type', None)
                    if QT_Type is not None:
                        types.append(getattr(QT_Type, 'KeyPress', None))
                    is_keypress = any(t is not None and et == t for t in types)
                    if not is_keypress:
                        # Last resort: compare ints if available
                        try:
                            is_keypress = int(et) == int(getattr(QT_Type, 'KeyPress'))
                        except Exception:
                            is_keypress = False
                except Exception:
                    is_keypress = False
                if is_keypress:
                    try:
                        key = ev.key() if hasattr(ev, 'key') else None
                    except Exception:
                        key = None
                        try:
                            mods = ev.modifiers()
                        except Exception:
                            mods = 0
                key_d = (getattr(Qt, 'Key_D', None), getattr(getattr(Qt, 'Key', object), 'Key_D', None), ord('D'), ord('d'))
                if key in key_d and not (mods & (ControlModifier | ShiftModifier | AltModifier)):
                    try:
                        self.exclude_button.setChecked(not self.exclude_button.isChecked())
                    except Exception:
                        self.on_toggle_exclude_mode(not getattr(self, 'exclude_mode', False))
                        try:
                            ev.accept()
                        except Exception:
                            pass
                    return True
            if QWheelEvent is not None and isinstance(ev, QWheelEvent):
                delta = ev.angleDelta().y()
                step = 1 if delta > 0 else -1
                mods = ev.modifiers()
                is_ctrl = bool(mods & ControlModifier)
                is_shift = bool(mods & ShiftModifier)
                is_alt = bool(mods & AltModifier)
                # Lock out zoom when something is selected (elastic or phonon)
                try:
                    if getattr(self, 'selected_kind', None) is not None and not is_ctrl:
                        # We will handle the wheel for parameter adjustments; prevent zoom.
                        pass
                except Exception:
                    pass
                # Global resolution controls (Ctrl)
                if is_ctrl and is_shift:
                    # Ctrl+Shift = global Lorentzian FWHM
                    val = self.lorentz_fwhm_spinbox.value()
                    new_val = max(self.lorentz_fwhm_spinbox.minimum(), min(self.lorentz_fwhm_spinbox.maximum(), val * (RES_SCROLL_SCALE ** step)))
                    self.lorentz_fwhm_spinbox.setValue(new_val)
                    self.update_previews()
                    return True
                if is_ctrl and not is_shift and not is_alt:
                    # Ctrl = global Gaussian FWHM
                    val = self.gauss_fwhm_spinbox.value()
                    new_val = max(self.gauss_fwhm_spinbox.minimum(), min(self.gauss_fwhm_spinbox.maximum(), val * (RES_SCROLL_SCALE ** step)))
                    self.gauss_fwhm_spinbox.setValue(new_val)
                    self.update_previews()
                    return True
                # Elastic-specific resolution controls when Elastic selected (no Ctrl)
                if self.selected_kind == 'elastic':
                    if is_alt:
                        # Alt or Alt+Shift = Lorentzian FWHM for elastic
                        val = self.lorentz_fwhm_spinbox.value()
                        new_val = max(self.lorentz_fwhm_spinbox.minimum(), min(self.lorentz_fwhm_spinbox.maximum(), val * (RES_SCROLL_SCALE ** step)))
                        self.lorentz_fwhm_spinbox.setValue(new_val)
                        self.update_previews()
                        return True
                    if is_shift and not is_alt:
                        # Shift = Gaussian FWHM for elastic
                        val = self.gauss_fwhm_spinbox.value()
                        new_val = max(self.gauss_fwhm_spinbox.minimum(), min(self.gauss_fwhm_spinbox.maximum(), val * (RES_SCROLL_SCALE ** step)))
                        self.gauss_fwhm_spinbox.setValue(new_val)
                        self.update_previews()
                        return True
                # Per-selection controls
                if self.selected_kind == 'phonon' and self.selected_obj is not None:
                    if is_shift:
                        dmp = clamp_damping(float(self.selected_obj.get('damping', DHO_DAMPING_DEFAULT)) * (DAMP_SCROLL_SCALE ** step))
                        self.selected_obj['damping'] = dmp
                        try:
                            if 'widgets' in self.selected_obj and 'damping_spin' in self.selected_obj['widgets']:
                                w = self.selected_obj['widgets']['damping_spin']
                                was = w.blockSignals(True)
                                w.setValue(dmp)
                                w.blockSignals(was)
                        except Exception:
                            pass
                    else:
                        new_h = self.selected_obj['height'] * (HEIGHT_SCROLL_SCALE ** step)
                        if self.selected_obj['height'] == 0 and step > 0:
                            yr = self.plot_widget.getViewBox().viewRange()[1]
                            new_h = max(1.0, 0.02 * (yr[1] - yr[0]))
                        self.selected_obj['height'] = max(0.0, new_h)
                        try:
                            if 'widgets' in self.selected_obj and 'height_spin' in self.selected_obj['widgets']:
                                w = self.selected_obj['widgets']['height_spin']
                                was = w.blockSignals(True)
                                w.setValue(self.selected_obj['height'])
                                w.blockSignals(was)
                        except Exception:
                            pass
                    # Invalidate cached area
                    try:
                        if 'area' in self.selected_obj:
                            del self.selected_obj['area']
                    except Exception:
                        pass
                    self.update_previews()
                    return True
                elif self.selected_kind == 'elastic' and not (is_ctrl or is_alt or is_shift):
                    val = self.elastic_height_spinbox.value()
                    new_val = val * (HEIGHT_SCROLL_SCALE ** step)
                    if val == 0 and step > 0:
                        yr = self.plot_widget.getViewBox().viewRange()[1]
                        new_val = max(1.0, 0.02 * (yr[1] - yr[0]))
                    self.elastic_height_spinbox.setValue(max(self.elastic_height_spinbox.minimum(), min(self.elastic_height_spinbox.maximum(), new_val)))
                    self.update_previews()
                    return True
                # If a selection is active but no handler above consumed it, block zoom
                if getattr(self, 'selected_kind', None) is not None:
                    return True
        except Exception:
            pass
        return super().eventFilter(obj, ev)

    # --- Keyboard handling ---
    def on_key_press(self, event):
        try:
            key = event.key()
            if key == Key_Space or key == getattr(Qt, 'Key_Space', None) or key == getattr(getattr(Qt, 'Key', object), 'Key_Space', None):
                # Spacebar deselects
                self.set_selected(None, None)
                self._dragging = False
                self._drag_artist = None
                return
            if key == Qt.Key_D:
                try:
                    self.exclude_button.setChecked(not self.exclude_button.isChecked())
                except Exception:
                    self.exclude_mode = not self.exclude_mode
                    self.on_toggle_exclude_mode(self.exclude_mode)
                return
            if key in (Qt.Key_E, getattr(Qt, 'Key_e', None)):
                self.set_selected('elastic', None)
                return
            if key in (Qt.Key_Tab, Qt.Key_Right):
                self._cycle_selection(direction=1)
                return
            if key == Qt.Key_Left:
                self._cycle_selection(direction=-1)
                return
            if key == Qt.Key_R:
                self._reset_zoom()
                return
            # Number keys 1-9 select phonons
            if Qt.Key_1 <= key <= Qt.Key_9:
                idx = key - Qt.Key_1
                if 0 <= idx < len(self.draggable_artists):
                    self.set_selected('phonon', self.draggable_artists[idx])
                return
        except Exception:
            pass
        try:
            return super(PlotWidget, self.plot_widget).keyPressEvent(event)
        except Exception:
            pass

    def on_key_release(self, event):
        try:
            return super(PlotWidget, self.plot_widget).keyReleaseEvent(event)
        except Exception:
            pass

    def _cycle_selection(self, direction=1):
        options = []
        if True:
            options.append(('elastic', None))
        for info in self.draggable_artists:
            options.append(('phonon', info))
        if not options:
            self.set_selected(None, None)
            return
        try:
            idx = options.index((self.selected_kind, self.selected_obj))
            idx = (idx + (1 if direction >= 0 else -1)) % len(options)
        except ValueError:
            idx = 0
        self.set_selected(*options[idx])

    # --- File loading and list handling ---
    def load_data(self):
        try:
            default_dir = self.user_dirs.get('input_dir', resolve_default_input_dir()) if hasattr(self, 'user_dirs') else resolve_default_input_dir()
        except Exception:
            default_dir = resolve_default_input_dir()
        filepaths, _ = QFileDialog.getOpenFileNames(self, "Open Data Files", default_dir, DATA_FILE_FILTER)
        if not filepaths:
            return False
        self.file_list_paths = list(filepaths)
        self._populate_file_list_widget()
        try:
            self._save_file_list()
        except Exception:
            pass
        try:
            self.file_list_widget.setCurrentRow(0)
        except Exception:
            pass

    def _populate_file_list_widget(self):
        try:
            self.file_list_widget.blockSignals(True)
            self.file_list_widget.clear()
            for p in self.file_list_paths:
                self.file_list_widget.addItem(os.path.basename(p))
        finally:
            try:
                self.file_list_widget.blockSignals(False)
            except Exception:
                pass

    def _save_file_list(self):
        try:
            cfg_dir = self._get_config_dir()
            path = os.path.join(cfg_dir, 'last_file_list.json')
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(self.file_list_paths, f, indent=2)
        except Exception:
            pass

    def _load_saved_file_list(self):
        try:
            cfg_dir = self._get_config_dir()
            path = os.path.join(cfg_dir, 'last_file_list.json')
            if not os.path.isfile(path):
                return
            with open(path, 'r', encoding='utf-8') as f:
                paths = json.load(f)
            paths = [p for p in paths if isinstance(p, str) and os.path.isfile(p)]
            if not paths:
                return
            self.file_list_paths = paths
            self._populate_file_list_widget()
            try:
                self.file_list_widget.setCurrentRow(0)
            except Exception:
                pass
        except Exception:
            pass

    def on_file_selected(self, row):
        self._schedule_file_load(row)

    def on_file_item_clicked(self, item):
        try:
            row = self.file_list_widget.row(item)
        except Exception:
            row = -1
        if row is None or row < 0:
            return False
        self._schedule_file_load(row)

    def on_current_item_changed(self, current, previous):
        try:
            row = self.file_list_widget.row(current) if current is not None else -1
        except Exception:
            row = -1
        if row is None or row < 0:
            return False
        self._schedule_file_load(row)

    def on_file_selection_changed(self):
        try:
            row = self.file_list_widget.currentRow()
        except Exception:
            row = -1
        if row is None or row < 0:
            return False
        self._schedule_file_load(row)

    def on_model_current_changed(self, current, previous):
        try:
            row = current.row() if current is not None else -1
        except Exception:
            row = -1
        if row is None or row < 0:
            return False
        self._schedule_file_load(row)

    def _schedule_file_load(self, row):
        try:
            if not isinstance(row, int) or row < 0 or row >= len(self.file_list_paths):
                return
            self._pending_row = int(row)
            self._load_row_timer.start(30)
        except Exception:
            self._do_load_pending_row(force_row=row)

    def _do_load_pending_row(self, force_row=None):
        try:
            row = int(force_row) if force_row is not None else int(self._pending_row if self._pending_row is not None else -1)
        except Exception:
            row = -1
        self._pending_row = None
        if not isinstance(row, int) or row < 0 or row >= len(self.file_list_paths):
            return False
        filepath = self.file_list_paths[row]
        self._load_and_plot_file(filepath)

    def remove_selected_file_from_list(self):
        try:
            row = self.file_list_widget.currentRow()
        except Exception:
            row = -1
        if row is None or row < 0 or row >= len(self.file_list_paths):
            return False
        removed = None
        try:
            removed = self.file_list_paths.pop(row)
        except Exception:
            pass
        try:
            self.file_list_widget.blockSignals(True)
            item = self.file_list_widget.takeItem(row)
            del item
        except Exception:
            pass
        finally:
            try:
                self.file_list_widget.blockSignals(False)
            except Exception:
                pass
        try:
            current_loaded = self.file_info.get('path') if isinstance(self.file_info, dict) else None
        except Exception:
            current_loaded = None
        if removed and current_loaded and os.path.abspath(removed) == os.path.abspath(current_loaded):
            if 0 <= row < len(self.file_list_paths):
                try:
                    self.file_list_widget.setCurrentRow(row)
                    self.on_file_selected(row)
                except Exception:
                    pass
            elif len(self.file_list_paths) > 0:
                try:
                    self.file_list_widget.setCurrentRow(len(self.file_list_paths) - 1)
                    self.on_file_selected(len(self.file_list_paths) - 1)
                except Exception:
                    pass
        try:
            self._save_file_list()
        except Exception:
            pass

    def clear_file_list(self):
        self.file_list_paths = []
        try:
            self.file_list_widget.blockSignals(True)
            self.file_list_widget.clear()
        finally:
            try:
                self.file_list_widget.blockSignals(False)
            except Exception:
                pass
        try:
            self._save_file_list()
        except Exception:
            pass

    def add_files_to_list(self):
        default_dir = resolve_default_input_dir()
        filepaths, _ = QFileDialog.getOpenFileNames(self, "Add Files to List", default_dir, DATA_FILE_FILTER)
        if not filepaths:
            return False
        existing_set = set(map(os.path.abspath, self.file_list_paths))
        new_abs = [os.path.abspath(p) for p in filepaths]
        uniques = []
        duplicates = []
        for p in new_abs:
            if p in existing_set:
                duplicates.append(p)
            else:
                uniques.append(p)
        if not uniques and duplicates:
            try:
                QMessageBox.warning(self, "Already in List", "All selected files are already present in the list.")
            except Exception:
                pass
            return False
        self.file_list_paths.extend(uniques)
        try:
            self.file_list_widget.blockSignals(True)
            for p in uniques:
                self.file_list_widget.addItem(os.path.basename(p))
        finally:
            try:
                self.file_list_widget.blockSignals(False)
            except Exception:
                pass
        try:
            self._save_file_list()
        except Exception:
            pass
        if duplicates:
            try:
                QMessageBox.information(self, "Duplicates Skipped", f"{len(duplicates)} file(s) were already in the list and were skipped.")
            except Exception:
                pass
        try:
            if uniques:
                idx0 = len(self.file_list_paths) - len(uniques)
                self.file_list_widget.setCurrentRow(idx0)
        except Exception:
            pass
        return True

    # --- Load file and plot ---
    def _load_and_plot_file(self, filepath):
        if not filepath:
            return False
        self.file_info = parse_filename_new_patterns(filepath)
        self.file_info['path'] = filepath
        self.file_info['dir'] = os.path.dirname(filepath)
        self.file_info['name'] = os.path.basename(filepath)
        try:
            df = pd.read_csv(
                filepath,
                comment='#',
                header=None,
                names=[CSV_COL_ENERGY, CSV_COL_COUNTS, CSV_COL_ERROR],
                engine='python',
                skip_blank_lines=True
            )
            df = df.dropna(how='all')
            df[CSV_COL_ENERGY] = pd.to_numeric(df[CSV_COL_ENERGY], errors='coerce')
            df[CSV_COL_COUNTS] = pd.to_numeric(df[CSV_COL_COUNTS], errors='coerce')
            if CSV_COL_ERROR in df.columns:
                df[CSV_COL_ERROR] = pd.to_numeric(df[CSV_COL_ERROR], errors='coerce')
            df = df.dropna(subset=[CSV_COL_ENERGY, CSV_COL_COUNTS])
            if df.empty:
                raise ValueError("No numeric data found in file. Verify the file format.")
            self.energy = df[CSV_COL_ENERGY].to_numpy(dtype=float)
            self.counts = df[CSV_COL_COUNTS].to_numpy(dtype=float)
            if CSV_COL_ERROR in df.columns and df[CSV_COL_ERROR].notna().any():
                self.errors = df[CSV_COL_ERROR].fillna(method='ffill').fillna(method='bfill').to_numpy(dtype=float)
            else:
                self.errors = np.sqrt(np.abs(self.counts))
            try:
                self.bg_spinbox.setValue(float(np.nanmin(self.counts)))
            except Exception:
                pass
        except Exception as e:
            self.results_text.setText(f"Error loading file: {e}")
            return False

        self.add_peak_button.setEnabled(True)
        self.fit_button.setEnabled(True)
        self.clear_button.setEnabled(True)
        self.save_button.setEnabled(True)
        self.last_fit_params = None
        self.excluded_mask = None
        
        # CRITICAL: Always call plot_initial_data to properly clear old items
        self.plot_initial_data()
        
        # Then try to load previous fit (which will update the previews)
        fit_loaded = False
        try:
            fit_loaded = bool(self._load_previous_fit_if_exists())
        except Exception:
            fit_loaded = False
        try:
            if not fit_loaded:
                self.update_previews()
        except Exception:
            pass
        try:
            self._update_header_info()
        except Exception:
            pass

    def plot_initial_data(self):
        # First, properly clear all peak indicators and their UI
        try:
            # Clear the peaks UI panel first
            self._clear_layout(self.peaks_layout)
        except Exception:
            pass
        
        # Remove all phonon markers and lines from the plot
        for info in list(self.draggable_artists):
            try:
                if 'marker' in info and info['marker'] is not None:
                    self.plot_widget.getPlotItem().removeItem(info['marker'])
            except Exception:
                pass
            try:
                if 'preview_line' in info and info['preview_line'] is not None:
                    self.plot_widget.getPlotItem().removeItem(info['preview_line'])
            except Exception:
                pass
        
        # Clear the list of draggable artists
        self.draggable_artists = []
        self.selected_kind = None
        self.selected_obj = None
        
        # Clear plots using the simplified clear method
        try:
            self._reset_main_plots()
        except Exception as e:
            pass
        
        # Ensure legend and grid are present
        try:
            pi = self.plot_widget.getPlotItem()
            # If a legend exists, clear it to avoid duplicate entries when replotting
            if getattr(pi, 'legend', None) is not None:
                try:
                    pi.legend.clear()
                except Exception:
                    pass
            else:
                pi.addLegend(offset=(10, 10))
            self.plot_widget.showGrid(x=True, y=True, alpha=0.3)
        except Exception:
            pass
        
        # Reset cached item references
        self.data_included_item = None
        self.data_excluded_item = None
        self._err_inc = None
        self._err_exc = None
        
        # Initialize excluded mask
        try:
            if self.energy is not None:
                if self.excluded_mask is None or len(self.excluded_mask) != len(self.energy):
                    self.excluded_mask = np.zeros(len(self.energy), dtype=bool)
        except Exception:
            self.excluded_mask = None
        
        # Setup preview grid
        try:
            n = max(PREVIEW_SAMPLES_MIN, int(len(self.energy) * PREVIEW_SAMPLES_FACTOR))
        except Exception:
            n = PREVIEW_SAMPLES_MIN
        self.x_preview = np.linspace(float(np.min(self.energy)), float(np.max(self.energy)), n)
        
        # CRITICAL: Remove old elastic/preview lines before creating new ones
        pi = self.plot_widget.getPlotItem()
        try:
            if hasattr(self, 'elastic_line') and self.elastic_line is not None:
                pi.removeItem(self.elastic_line)
                self.elastic_line = None
        except Exception:
            pass
        try:
            if hasattr(self, 'background_line') and self.background_line is not None:
                pi.removeItem(self.background_line)
                self.background_line = None
        except Exception:
            pass
        try:
            if hasattr(self, 'preview_line') and self.preview_line is not None:
                pi.removeItem(self.preview_line)
                self.preview_line = None
        except Exception:
            pass
        
        # Create elastic and total preview lines directly on PlotItem
        # Background (darker, bolder dotted gray)
        self.background_line = pi.plot([], [], pen=pg.mkPen((80, 80, 80), style=DotLine, width=2), name='Background')
        self.elastic_line = pi.plot([], [], pen=pg.mkPen('m', style=DashLine, width=2), name='Elastic')
        self.preview_line = pi.plot([], [], pen=pg.mkPen('r', width=2), name='Total Fit')
        try:
            vb = self.plot_widget.getViewBox()
            if hasattr(vb, 'childGroup'):
                self.background_line.setParentItem(vb.childGroup)
                self.elastic_line.setParentItem(vb.childGroup)
                self.preview_line.setParentItem(vb.childGroup)
        except Exception:
            pass
        
        # Now plot data - this will set the range
        # Start initial data plot
        self._update_data_plot(do_range=True)
        # Compute previews once so elastic/background/phonon lines have data for range computation
        try:
            self.update_previews()
        except Exception:
            pass
        # Done plotting initial data
        # Ensure legend order is consistent after initial plot
        try:
            self._refresh_legend()
        except Exception:
            pass
        
        # Residuals viewbox: set fixed Y range initially
        try:
            rvb = self.resid_widget.getViewBox()
            if rvb is not None:
                rvb.enableAutoRange(enable=False)
                rvb.setYRange(-5, 5, padding=0)
        except Exception:
            pass
            
        # Initialize residuals
        try:
            self._init_residual_plot()
        except Exception:
            pass
            
        # Link residuals X to main plot
        try:
            self.resid_widget.setXLink(self.plot_widget)
        except Exception:
            pass
            
        # Default to no selection on entry
        self.set_selected(None, None)
        # After initial draw, ensure everything is visible
        try:
            # Set a flag to also ensure after next preview update (for reliability)
            self._need_initial_autorange = True
            self._ensure_all_visible()
        except Exception:
            pass

    def _reset_main_plots(self):
        """Clear the main plot and residual plot to avoid lingering items."""
        # CRITICAL: Completely clear PlotItem first, THEN remove orphaned ViewBox items
        try:
            if self.plot_widget is not None:
                pi = self.plot_widget.getPlotItem()
                vb = self.plot_widget.getViewBox()
                
                # Step 1: Clear PlotItem (removes PlotDataItems from PlotItem's list)
                pi.clear()
                
                # Step 2: Explicitly remove items that were added directly to the ViewBox (Scatter/ErrorBars)
                try:
                    if vb is not None and hasattr(vb, 'addedItems'):
                        for it in list(vb.addedItems):
                            try:
                                vb.removeItem(it)
                            except Exception:
                                pass
                except Exception:
                    pass
                
                # Step 3: Restore UI elements
                # Re-add legend after clearing
                if getattr(pi, 'legend', None) is None:
                    pi.addLegend(offset=(10, 10))
                # Restore grid
                self.plot_widget.showGrid(x=True, y=True, alpha=0.3)
                # CRITICAL: Ensure aspect ratio lock is disabled
                if vb is not None:
                    vb.setAspectLocked(False)
        except Exception as e:
            print(f"Error clearing main plot: {e}")
            
        try:
            if self.resid_widget is not None:
                rpi = self.resid_widget.getPlotItem()
                rpi.clear()
                # Restore grid
                self.resid_widget.showGrid(x=False, y=True, alpha=0.3)
                # Disable aspect lock for residuals too
                rvb = self.resid_widget.getViewBox()
                if rvb is not None:
                    rvb.setAspectLocked(False)
        except Exception as e:
            print(f"Error clearing residual plot: {e}")
            
        # Reset cached references (but NOT elastic_line/preview_line - those are handled in plot_initial_data)
        self.data_included_item = None
        self.data_excluded_item = None
        # DO NOT reset elastic_line/preview_line here - they're removed in plot_initial_data()
        self.resid_bars = None
        self.resid_zero_line = None

    def _ensure_all_visible(self):
        """Ensure that all relevant plot items are inside the current view. Called after initial draw."""
        vb = self.plot_widget.getViewBox()
        if vb is None:
            return
        try:
            # Build extents from data and model lines
            xs = []
            ys = []
            def add_xy(item):
                if item is None:
                    return
                try:
                    X, Y = item.getData()
                    if X is None or Y is None or len(X) == 0:
                        return
                    xs.append(np.nanmin(X)); xs.append(np.nanmax(X))
                    ys.append(np.nanmin(Y)); ys.append(np.nanmax(Y))
                except Exception:
                    pass
            # Data
            try:
                if self.data_included_item is not None:
                    Xi = np.asarray(self.data_included_item.getData()[0], dtype=float)
                    Yi = np.asarray(self.data_included_item.getData()[1], dtype=float)
                    if Xi.size and Yi.size:
                        xs.extend([np.nanmin(Xi), np.nanmax(Xi)])
                        ys.extend([np.nanmin(Yi), np.nanmax(Yi)])
                        if hasattr(self, '_err_inc') and self._err_inc is not None:
                            try:
                                top = np.asarray(self._err_inc.opts.get('top'), dtype=float)
                                bot = np.asarray(self._err_inc.opts.get('bottom'), dtype=float)
                                ys.extend([np.nanmin(Yi - bot), np.nanmax(Yi + top)])
                            except Exception:
                                pass
                if self.data_excluded_item is not None:
                    Xe = np.asarray(self.data_excluded_item.getData()[0], dtype=float)
                    Ye = np.asarray(self.data_excluded_item.getData()[1], dtype=float)
                    if Xe.size and Ye.size:
                        xs.extend([np.nanmin(Xe), np.nanmax(Xe)])
                        ys.extend([np.nanmin(Ye), np.nanmax(Ye)])
                        if hasattr(self, '_err_exc') and self._err_exc is not None:
                            try:
                                top = np.asarray(self._err_exc.opts.get('top'), dtype=float)
                                bot = np.asarray(self._err_exc.opts.get('bottom'), dtype=float)
                                ys.extend([np.nanmin(Ye - bot), np.nanmax(Ye + top)])
                            except Exception:
                                pass
            except Exception:
                pass
            # Background, elastic, total
            add_xy(self.background_line)
            add_xy(self.elastic_line)
            add_xy(self.preview_line)
            # Phonon previews
            for info in self.draggable_artists:
                add_xy(info.get('preview_line'))
                try:
                    mk = info.get('marker')
                    if mk is not None:
                        X, Y = mk.getData()
                        if X is not None and Y is not None and len(X) > 0:
                            xs.extend([np.nanmin(X), np.nanmax(X)])
                            ys.extend([np.nanmin(Y), np.nanmax(Y)])
                except Exception:
                    pass
            if not xs or not ys:
                return
            xmin, xmax = float(np.nanmin(xs)), float(np.nanmax(xs))
            ymin, ymax = float(np.nanmin(ys)), float(np.nanmax(ys))
            if not np.isfinite(xmin) or not np.isfinite(xmax) or not np.isfinite(ymin) or not np.isfinite(ymax):
                return
            xr = xmax - xmin if (xmax - xmin) > 0 else 1.0
            yr = ymax - ymin if (ymax - ymin) > 0 else 1.0
            vb.enableAutoRange(enable=False)
            vb.setXRange(xmin - 0.10 * xr, xmax + 0.10 * xr, padding=0)
            vb.setYRange(ymin - 0.14 * yr, ymax + 0.20 * yr, padding=0)
        except Exception:
            pass

    def _update_data_plot(self, do_range=True):
        if self.energy is None or self.counts is None:
            return
        
        x = np.asarray(self.energy, dtype=float)
        y = np.asarray(self.counts, dtype=float)
        yerr = self.errors if self.errors is not None else np.sqrt(np.abs(y))
        mask_inc = np.ones_like(y, dtype=bool)
        if isinstance(self.excluded_mask, np.ndarray) and len(self.excluded_mask) == len(y):
            mask_inc = ~self.excluded_mask
        
        # Remove old items if they exist
        try:
            if self.data_included_item is not None:
                self._vb_remove(self.data_included_item)
                self.data_included_item = None
        except Exception as e:
            pass
            
        try:
            if self.data_excluded_item is not None:
                self._vb_remove(self.data_excluded_item)
                self.data_excluded_item = None
        except Exception as e:
            pass
            
        try:
            if hasattr(self, '_err_inc') and self._err_inc is not None:
                self._vb_remove(self._err_inc)
                self._err_inc = None
        except Exception as e:
            pass
            
        try:
            if hasattr(self, '_err_exc') and self._err_exc is not None:
                self._vb_remove(self._err_exc)
                self._err_exc = None
        except Exception as e:
            pass
        
        # Plot included points with error bars
        inc = mask_inc
        x_inc = x[inc]
        y_inc = y[inc]
        dy_inc = np.asarray(yerr)[inc]
        
        # Included points debug removed
        
        if len(x_inc) > 0:
            pi = self.plot_widget.getPlotItem()
            vb = self.plot_widget.getViewBox()
            if self.data_included_item is None:
                # Create once and add to legend
                self.data_included_item = ScatterPlotItem(
                    x=x_inc,
                    y=y_inc,
                    pen=pg.mkPen('k'),
                    brush=pg.mkBrush('k'),
                    size=7,
                    symbol='o'
                )
                self._vb_add(self.data_included_item)
                try:
                    if getattr(pi, 'legend', None) is not None:
                        pi.legend.addItem(self.data_included_item, 'Data')
                except Exception:
                    pass
            else:
                try:
                    self.data_included_item.setData(x=x_inc, y=y_inc)
                except Exception:
                    pass

            if getattr(self, '_err_inc', None) is None:
                self._err_inc = ErrorBarItem(
                    x=x_inc,
                    y=y_inc,
                    top=dy_inc,
                    bottom=dy_inc,
                    pen=pg.mkPen('k', width=1)
                )
                try:
                    if vb is not None:
                        vb.addItem(self._err_inc)
                    else:
                        pi.addItem(self._err_inc)
                except Exception:
                    try:
                        pi.addItem(self._err_inc)
                    except Exception:
                        pass
            else:
                try:
                    self._err_inc.setData(x=x_inc, y=y_inc, top=dy_inc, bottom=dy_inc)
                except Exception:
                    pass
        
        # Plot excluded points with error bars if any
        exc = (~mask_inc)
        if np.any(exc):
            x_exc = x[exc]
            y_exc = y[exc]
            dy_exc = np.asarray(yerr)[exc]
            
            if len(x_exc) > 0:
                pi = self.plot_widget.getPlotItem()
                vb = self.plot_widget.getViewBox()
                if self.data_excluded_item is None:
                    self.data_excluded_item = ScatterPlotItem(
                        x=x_exc,
                        y=y_exc,
                        pen=pg.mkPen(150),
                        brush=None,
                        size=8,
                        symbol='x'
                    )
                    self._vb_add(self.data_excluded_item)
                    try:
                        if getattr(pi, 'legend', None) is not None:
                            pi.legend.addItem(self.data_excluded_item, 'Excluded')
                    except Exception:
                        pass
                else:
                    try:
                        self.data_excluded_item.setData(x=x_exc, y=y_exc)
                    except Exception:
                        pass

                if getattr(self, '_err_exc', None) is None:
                    self._err_exc = ErrorBarItem(
                        x=x_exc,
                        y=y_exc,
                        top=dy_exc,
                        bottom=dy_exc,
                        pen=pg.mkPen(color=(180, 180, 180), width=1)
                    )
                    try:
                        if vb is not None:
                            vb.addItem(self._err_exc)
                        else:
                            pi.addItem(self._err_exc)
                    except Exception:
                        try:
                            pi.addItem(self._err_exc)
                        except Exception:
                            pass
                else:
                    try:
                        self._err_exc.setData(x=x_exc, y=y_exc, top=dy_exc, bottom=dy_exc)
                    except Exception:
                        pass
        # After updating data items, refresh legend order
        try:
            self._refresh_legend()
        except Exception:
            pass
        
        # Update range after plotting data if requested
        if do_range and len(x_inc) > 0:
            try:
                vb = self.plot_widget.getViewBox()
                if vb is not None:
                    # Calculate range from included data
                    xmin, xmax = float(np.nanmin(x_inc)), float(np.nanmax(x_inc))
                    # Include error bars when computing Y range to avoid clipping
                    try:
                        ymin = float(np.nanmin(y_inc - dy_inc))
                        ymax = float(np.nanmax(y_inc + dy_inc))
                    except Exception:
                        ymin, ymax = float(np.nanmin(y_inc)), float(np.nanmax(y_inc))
                    xr = xmax - xmin if (xmax - xmin) > 0 else 1.0
                    yr = ymax - ymin if (ymax - ymin) > 0 else 1.0
                    
                    # Range set
                    
                    # Completely disable autoRange before setting fixed ranges
                    vb.enableAutoRange(enable=False)
                    vb.setLimits(xMin=None, xMax=None, yMin=None, yMax=None)
                    
                    # Set range with manual padding
                    vb.setXRange(xmin - 0.10*xr, xmax + 0.10*xr, padding=0)
                    vb.setYRange(ymin - 0.12*yr, ymax + 0.18*yr, padding=0)
                    
                    # Verify the range was set
                    actual_range = vb.viewRange()
                    
                    # Check what items are actually in the ViewBox and PlotItem
                    # Debug state removed
            except Exception as e:
                pass
        
        # Draw previews if already initialized
        try:
            if self.x_preview is not None:
                self.update_previews()
        except Exception:
            pass

    def _add_or_update_errorbars(self, kind, x, y, dy, color='k'):
        key = f'_err_{kind}'
        item = getattr(self, key, None)
        try:
            if item is None:
                erritem = ErrorBarItem(x=np.asarray(x), y=np.asarray(y), top=np.asarray(dy), bottom=np.asarray(dy), pen=pg.mkPen(color))
                self._vb_add(erritem)
                setattr(self, key, erritem)
            else:
                item.setData(x=np.asarray(x), y=np.asarray(y), top=np.asarray(dy), bottom=np.asarray(dy))
        except Exception:
            pass

    # --- Header/status ---
    def _update_header_info(self):
        try:
            H = self.file_info.get('H', None) if isinstance(self.file_info, dict) else None
            K = self.file_info.get('K', None) if isinstance(self.file_info, dict) else None
            L = self.file_info.get('L', None) if isinstance(self.file_info, dict) else None
            B = self.file_info.get('B', None) if isinstance(self.file_info, dict) else None
            T = self.file_info.get('T', 1.5) if isinstance(self.file_info, dict) else 1.5
            def fmt(x, digits):
                try:
                    if x is None or not np.isfinite(float(x)):
                        return '-'
                    return f"{float(x):.{digits}g}"
                except Exception:
                    return '-'
            qtxt = f"Q = ({fmt(H,3)}, {fmt(K,3)}, {fmt(L,3)})"
            btxt = f"B = {fmt(B,3)} T" if B is not None else "B = -"
            ttxt = f"T = {fmt(T,3)} K"
            self.header_label.setText(f"{qtxt}    {btxt}    {ttxt}")
            try:
                fname = self.file_info.get('name', '') if isinstance(self.file_info, dict) else ''
                title_parts = []
                if fname:
                    title_parts.append(fname)
                title_parts.append(qtxt)
                title_parts.append(btxt)
                title_parts.append(ttxt)
                self.setWindowTitle(" | ".join(title_parts))
            except Exception:
                pass
        except Exception:
            try:
                self.header_label.setText("")
            except Exception:
                pass

    # --- Peaks UI and interactions ---
    def _clear_layout(self, layout):
        while layout.count():
            item = layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()

    def rebuild_peaks_panel(self):
        self._clear_layout(self.peaks_layout)
        try:
            header = QWidget()
            hl = QHBoxLayout(header)
            hl.setContentsMargins(0, 0, 0, 0)
            hl.setSpacing(6)
            for L in [QLabel("Phonon"), QLabel("C (meV)"), QLabel("Fix"), QLabel("H"), QLabel("Fix"), QLabel("Dmp"), QLabel("Fix"), QLabel("")]:
                try:
                    L.setStyleSheet("font-weight: bold;")
                except Exception:
                    pass
                hl.addWidget(L)
            self.peaks_layout.addWidget(header)
            note = QLabel("Note: checkboxes fix that parameter during the fit.")
            try:
                note.setStyleSheet("color: #555;")
            except Exception:
                pass
            self.peaks_layout.addWidget(note)
        except Exception:
            pass
        for i, info in enumerate(self.draggable_artists):
            row = QWidget()
            row_layout = QHBoxLayout(row)
            row_layout.setContentsMargins(0, 0, 0, 0)
            row_layout.setSpacing(6)
            lbl = QLabel(f"Phonon {i+1}")
            row_layout.addWidget(lbl)
            try:
                if info.get('preview_line') is not None:
                    info['preview_line'].setName(f"Phonon {i+1}")
            except Exception:
                pass
            cmin = float(np.min(self.energy)) if self.energy is not None else -100.0
            cmax = float(np.max(self.energy)) if self.energy is not None else 100.0
            center_spin = QDoubleSpinBox()
            center_spin.setDecimals(4)
            center_spin.setRange(cmin - 10.0, cmax + 10.0)
            center_spin.setSingleStep(0.05)
            center_spin.setValue(float(info['center']))
            center_spin.valueChanged.connect(partial(self.on_peak_center_spin_changed, info))
            fix_e_cb = QCheckBox()
            try:
                fix_e_cb.setToolTip("Fix center during fitting")
            except Exception:
                pass
            try:
                fix_e_cb.setChecked(bool(info.get('fix_E', False)))
            except Exception:
                pass
            fix_e_cb.toggled.connect(partial(self.on_peak_fix_flag_changed, info, 'fix_E'))
            row_layout.addWidget(center_spin)
            row_layout.addWidget(fix_e_cb)

            height_spin = QDoubleSpinBox()
            height_spin.setDecimals(2)
            height_spin.setRange(0.0, 1e9)
            height_spin.setSingleStep(10.0)
            height_spin.setValue(float(info['height']))
            height_spin.valueChanged.connect(partial(self.on_peak_height_spin_changed, info))
            fix_h_cb = QCheckBox()
            try:
                fix_h_cb.setToolTip("Fix height (area) during fitting")
            except Exception:
                pass
            try:
                fix_h_cb.setChecked(bool(info.get('fix_H', False)))
            except Exception:
                pass
            fix_h_cb.toggled.connect(partial(self.on_peak_fix_flag_changed, info, 'fix_H'))
            row_layout.addWidget(height_spin)
            row_layout.addWidget(fix_h_cb)

            damping_spin = QDoubleSpinBox()
            damping_spin.setDecimals(4)
            damping_spin.setRange(DHO_DAMPING_MIN, DHO_DAMPING_MAX)
            damping_spin.setSingleStep(DHO_DAMPING_STEP)
            d_init = clamp_damping(float(info.get('damping', DHO_DAMPING_DEFAULT)))
            info['damping'] = d_init
            damping_spin.setValue(d_init)
            damping_spin.valueChanged.connect(partial(self.on_peak_damping_spin_changed, info))
            fix_d_cb = QCheckBox()
            try:
                fix_d_cb.setToolTip("Fix damping during fitting")
            except Exception:
                pass
            try:
                fix_d_cb.setChecked(bool(info.get('fix_D', False)))
            except Exception:
                pass
            fix_d_cb.toggled.connect(partial(self.on_peak_fix_flag_changed, info, 'fix_D'))
            row_layout.addWidget(damping_spin)
            row_layout.addWidget(fix_d_cb)

            remove_btn = QPushButton('x')
            try:
                remove_btn.setFixedWidth(22)
                remove_btn.setToolTip('Remove this phonon')
            except Exception:
                pass
            remove_btn.clicked.connect(partial(self.remove_phonon, info))
            row_layout.addWidget(remove_btn)

            self.peaks_layout.addWidget(row)
            info.setdefault('widgets', {})
            info['widgets'].update({
                'row': row,
                'label': lbl,
                'center_spin': center_spin,
                'height_spin': height_spin,
                'damping_spin': damping_spin,
                'fix_e_cb': fix_e_cb,
                'fix_h_cb': fix_h_cb,
                'fix_d_cb': fix_d_cb,
                'remove_btn': remove_btn,
            })
        self._update_peak_row_styles()

    def _update_peak_row_styles(self):
        for idx, info in enumerate(self.draggable_artists):
            lbl = info.get('widgets', {}).get('label')
            if lbl:
                if self.selected_kind == 'phonon' and self.selected_obj is info:
                    lbl.setStyleSheet("font-weight: bold; color: #0077cc;")
                else:
                    lbl.setStyleSheet("")
            # Emphasize selected phonon line and marker
            try:
                line = info.get('preview_line')
                marker = info.get('marker')
                if self.selected_kind == 'phonon' and self.selected_obj is info:
                    # Bold line and larger marker
                    if line is not None:
                        pen = line.opts.get('pen') if hasattr(line, 'opts') else None
                        color = pen.color() if pen is not None and hasattr(pen, 'color') else pg.mkPen(PHONON_COLOR_CYCLE[idx % len(PHONON_COLOR_CYCLE)]).color()
                        line.setPen(pg.mkPen(color, style=DashLine, width=3))
                    if marker is not None:
                        try:
                            marker.setSymbolSize(MARKER_SIZE_SELECTED)
                        except Exception:
                            # Fallback: reset data with size via setData opts
                            xdata, ydata = marker.getData()
                            marker.setData(x=xdata, y=ydata, symbolSize=MARKER_SIZE_SELECTED)
                else:
                    if line is not None:
                        pen = line.opts.get('pen') if hasattr(line, 'opts') else None
                        color = pen.color() if pen is not None and hasattr(pen, 'color') else pg.mkPen(PHONON_COLOR_CYCLE[idx % len(PHONON_COLOR_CYCLE)]).color()
                        line.setPen(pg.mkPen(color, style=DashLine, width=2))
                    if marker is not None:
                        try:
                            marker.setSymbolSize(MARKER_SIZE_NORMAL)
                        except Exception:
                            xdata, ydata = marker.getData()
                            marker.setData(x=xdata, y=ydata, symbolSize=MARKER_SIZE_NORMAL)
            except Exception:
                pass

    def on_peak_fix_flag_changed(self, info, key, checked):
        try:
            info[key] = bool(checked)
        except Exception:
            pass

    def on_peak_center_spin_changed(self, info, val):
        info['center'] = float(val)
        if 'area' in info:
            try:
                del info['area']
            except Exception:
                pass
        self.update_previews()

    def on_peak_height_spin_changed(self, info, val):
        info['height'] = max(0.0, float(val))
        if 'area' in info:
            try:
                del info['area']
            except Exception:
                pass
        self.update_previews()

    def on_peak_damping_spin_changed(self, info, val):
        try:
            v = float(val)
        except Exception:
            v = DHO_DAMPING_DEFAULT
        v = clamp_damping(v)
        info['damping'] = v
        if 'area' in info:
            try:
                del info['area']
            except Exception:
                pass
        self.update_previews()

    def add_peak(self, checked=False, select=True):
        if self.energy is None:
            return
        # Place new peak at current x-center of view if available; otherwise data mean
        try:
            vb = self.plot_widget.getViewBox()
            if vb is not None:
                (xmin, xmax), (ymin, ymax) = vb.viewRange()
            else:
                raise RuntimeError()
            center = float(0.5 * (xmin + xmax))
            # Set height so marker lands around 30% below the top of view from the elastic baseline at center
            try:
                elastic_height = float(self.elastic_height_spinbox.value())
                gauss_fwhm = float(self.gauss_fwhm_spinbox.value())
                lorentz_fwhm = float(self.lorentz_fwhm_spinbox.value())
                bg = float(self.bg_spinbox.value())
                base_y = elastic_baseline_at_energy(center, elastic_height, gauss_fwhm, lorentz_fwhm, bg)
                target_y = ymin + 0.7 * (ymax - ymin)  # 70% up from bottom
                height = max(0.0, float(target_y - base_y))
            except Exception:
                height = float(np.max(self.counts) / 3.0)
        except Exception:
            center = float(np.mean(self.energy))
            height = float(np.max(self.counts) / 2.0)
        colors = PHONON_COLOR_CYCLE
        color = colors[len(self.draggable_artists) % len(colors)]
        # Line and marker on the axes using PlotItem.plot
        pi = self.plot_widget.getPlotItem()
        # Give each phonon preview a unique legend name
        name = f"Phonon {len(self.draggable_artists)+1}"
        line = pi.plot([], [], pen=pg.mkPen(color, style=DashLine, width=2), name=name)
        marker = pi.plot([center], [height], pen=None, symbol='x', symbolSize=MARKER_SIZE_NORMAL, symbolPen=pg.mkPen(color, width=2))
        try:
            line.setZValue(10)
            marker.setZValue(11)
        except Exception:
            pass
        try:
            if hasattr(pi, 'vb') and hasattr(pi.vb, 'childGroup'):
                line.setParentItem(pi.vb.childGroup)
                marker.setParentItem(pi.vb.childGroup)
        except Exception:
            pass
        artist_info = {
            'marker': marker,
            'preview_line': line,
            'center': center,
            'height': height,
            'fwhm': 0.5,
            'damping': DHO_DAMPING_DEFAULT
        }
        # Clicks are handled via scene events and nearest-target logic
        self.draggable_artists.append(artist_info)
        if select:
            self.set_selected('phonon', artist_info)
        # Ensure initial visual placement relative to current baseline
        try:
            elastic_height = float(self.elastic_height_spinbox.value())
            gauss_fwhm = float(self.gauss_fwhm_spinbox.value())
            lorentz_fwhm = float(self.lorentz_fwhm_spinbox.value())
            bg = float(self.bg_spinbox.value())
            base_y = elastic_baseline_at_energy(center, elastic_height, gauss_fwhm, lorentz_fwhm, bg)
            marker.setData([center], [base_y + height])
        except Exception:
            pass
        self.update_previews()
        try:
            self._refresh_legend()
        except Exception:
            pass
        self.rebuild_peaks_panel()

    def remove_phonon(self, info):
        try:
            if 'marker' in info and info['marker'] is not None:
                self._vb_remove(info['marker'])
        except Exception:
            pass
        try:
            if 'preview_line' in info and info['preview_line'] is not None:
                self._vb_remove(info['preview_line'])
        except Exception:
            pass
        try:
            self.draggable_artists.remove(info)
        except ValueError:
            pass
        if self.selected_obj is info:
            self.set_selected(None, None)
        self.rebuild_peaks_panel()
        self.update_previews()
        try:
            self._refresh_legend()
        except Exception:
            pass

    def clear_peaks(self, reset_view=True):
        for info in list(self.draggable_artists):
            try:
                if info.get('marker') is not None:
                    self._vb_remove(info['marker'])
            except Exception:
                pass
            try:
                if info.get('preview_line') is not None:
                    self._vb_remove(info['preview_line'])
            except Exception:
                pass
        self.draggable_artists = []
        self.selected_kind = None
        self.selected_obj = None
        self.set_selected(None, None)
        self.rebuild_peaks_panel()
        try:
            self._refresh_legend()
        except Exception:
            pass

    def set_selected(self, kind, obj):
        self.selected_kind = kind
        self.selected_obj = obj
        if kind == 'elastic':
            self.selection_label.setText("Selected: Elastic")
        elif kind == 'phonon':
            try:
                idx = self.draggable_artists.index(obj) + 1 if obj in self.draggable_artists else '?'
            except Exception:
                idx = '?'
            self.selection_label.setText(f"Selected: Phonon {idx}")
        else:
            self.selection_label.setText("Selected: None")
        # Emphasize elastic line when selected
        try:
            if kind == 'elastic' and self.elastic_line is not None:
                self.elastic_line.setPen(pg.mkPen('m', style=DashLine, width=3))
                if self.background_line is not None:
                    self.background_line.setPen(pg.mkPen((60, 60, 60), style=DotLine, width=3))
            elif self.elastic_line is not None:
                self.elastic_line.setPen(pg.mkPen('m', style=DashLine, width=2))
                if self.background_line is not None:
                    self.background_line.setPen(pg.mkPen((80, 80, 80), style=DotLine, width=2))
        except Exception:
            pass
        self._update_peak_row_styles()
        self.update_previews()

    # --- Previews and chi2/residuals ---
    def update_previews(self):
        if self.energy is None or self.x_preview is None:
            return
        # Reentrancy guard to avoid range-changed feedback loops
        if getattr(self, '_in_preview_update', False):
            return
        self._in_preview_update = True
        try:
            elastic_height = float(self.elastic_height_spinbox.value())
            gauss_fwhm = float(self.gauss_fwhm_spinbox.value())
            lorentz_fwhm = float(self.lorentz_fwhm_spinbox.value())
            bg = float(self.bg_spinbox.value())
            T = float(self.file_info.get('T', 1.5))
            # Update background line (constant BG as dotted line)
            if self.background_line is not None:
                try:
                    self.background_line.setData(self.x_preview, np.full_like(self.x_preview, bg, dtype=float))
                except Exception:
                    pass
            elastic_area = elastic_area_from_height(elastic_height, gauss_fwhm, lorentz_fwhm)
            elastic_baseline = elastic_voigt_y(self.x_preview, elastic_area, gauss_fwhm, lorentz_fwhm) + bg
            if self.elastic_line is not None:
                self.elastic_line.setData(self.x_preview, elastic_baseline)
                
            total_y = elastic_baseline.copy()
            for info in self.draggable_artists:
                e_ph = abs(float(info['center']))
                h_ph = max(0.0, float(info['height']))
                dmp = float(info.get('damping', 0.1))
                if 'area' in info and info['area'] is not None:
                    try:
                        a_ph = float(info['area'])
                    except Exception:
                        a_ph = dho_area_for_height_ui(h_ph, e_ph, gauss_fwhm, lorentz_fwhm, dmp, T, self.x_preview)
                else:
                    a_ph = dho_area_for_height_ui(h_ph, e_ph, gauss_fwhm, lorentz_fwhm, dmp, T, self.x_preview)
                y_ph = dho_y(self.x_preview, e_ph, gauss_fwhm, lorentz_fwhm, dmp, a_ph, T)
                if info.get('preview_line') is not None:
                    info['preview_line'].setData(self.x_preview, elastic_baseline + y_ph)
                    
                try:
                    baseline_center = elastic_baseline_at_energy(info['center'], elastic_height, gauss_fwhm, lorentz_fwhm, bg)
                    if info.get('marker') is not None:
                        info['marker'].setData([info['center']], [baseline_center + h_ph])
                except Exception:
                    pass
                total_y += y_ph
            if self.preview_line is not None:
                self.preview_line.setData(self.x_preview, total_y)
                
            self._update_chi2_label()
            try:
                self._update_residuals_plot()
            except Exception:
                pass
            # Run a one-time autorange after the first preview post-load
            try:
                # Skip autorange while excluding points to keep view steady
                if getattr(self, '_need_initial_autorange', False) and not getattr(self, 'exclude_mode', False):
                    self._ensure_all_visible()
                    self._need_initial_autorange = False
            except Exception:
                pass
        finally:
            self._in_preview_update = False

    def _compute_model_on_x(self, x):
        try:
            g = float(self.gauss_fwhm_spinbox.value())
            l = float(self.lorentz_fwhm_spinbox.value())
            bg = float(self.bg_spinbox.value())
            Tval = float(self.file_info.get('T', 1.5))
            elastic_h = float(self.elastic_height_spinbox.value())
            elastic_a = elastic_area_from_height(elastic_h, g, l)
            y = elastic_voigt_y(x, elastic_a, g, l, center=0.0)
            for info in self.draggable_artists:
                e = abs(float(info.get('center', 0.0)))
                h = max(0.0, float(info.get('height', 0.0)))
                dmp = float(info.get('damping', DHO_DAMPING_DEFAULT))
                if 'area' in info and info['area'] is not None:
                    try:
                        a = float(info['area'])
                    except Exception:
                        a = dho_area_for_height_ui(h, e, g, l, dmp, Tval, self.x_preview if getattr(self, 'x_preview', None) is not None else x)
                else:
                    a = dho_area_for_height_ui(h, e, g, l, dmp, Tval, self.x_preview if getattr(self, 'x_preview', None) is not None else x)
                y += dho_y(x, e, g, l, dmp, a, Tval)
            y += bg
            return y
        except Exception:
            return np.zeros_like(x, dtype=float)

    def _init_residual_plot(self):
        try:
            # Clear only the residual PlotItem to avoid affecting layout
            self.resid_widget.getPlotItem().clear()
            self.resid_widget.setLabel('bottom', 'Energy (meV)')
            self._resid_bars_item = None
        except Exception:
            pass
        self._update_residuals_plot(force_rebuild=True)
        try:
            vb = self.resid_widget.getViewBox()
            if vb is not None:
                vb.setDefaultPadding(0.05)
        except Exception:
            pass

    def _update_residuals_plot(self, force_rebuild=False):
        if self.energy is None or self.counts is None:
            return
        x, y, errs = self._get_included_data()
        if x is None or y is None or errs is None or len(x) == 0:
            return
        y_model = self._compute_model_on_x(x)
        r = (y_model - y) / np.clip(errs, 1e-12, np.inf)
        try:
            if len(x) > 1:
                dx = np.diff(np.asarray(x, dtype=float))
                dx = dx[np.isfinite(dx) & (dx > 0)]
                bar_w = float(np.median(dx)) * 0.9 if dx.size else 0.1
            else:
                xr = float(np.nanmax(x) - np.nanmin(x)) if np.isfinite(np.nanmax(x) - np.nanmin(x)) else 1.0
                bar_w = xr * 0.05
        except Exception:
            bar_w = 0.1
        need_rebuild = force_rebuild or (getattr(self, '_resid_bars_item', None) is None)
        if need_rebuild:
            try:
                if getattr(self, '_resid_bars_item', None) is not None:
                    # Remove from the residual PlotItem to avoid layout misplacement
                    self.resid_widget.getPlotItem().removeItem(self._resid_bars_item)
            except Exception:
                pass
            try:
                bars = BarGraphItem(x=x, height=r, width=bar_w, brush=pg.mkBrush(255, 165, 0, 200))
                self._resid_bars_item = bars
                # Add bars to the residual plot's PlotItem/ViewBox
                rpi = self.resid_widget.getPlotItem()
                rpi.addItem(bars)
                # Do not manually reparent to the ViewBox; PlotItem manages transforms
            except Exception:
                self._resid_bars_item = None
        else:
            try:
                self._resid_bars_item.setOpts(x=x, height=r, width=bar_w)
            except Exception:
                pass
        try:
            rmax = float(np.nanmax(np.abs(r)))
            if not np.isfinite(rmax) or rmax <= 0:
                rmax = 1.0
            vb = self.resid_widget.getViewBox()
            if vb is not None:
                vb.enableAutoRange(enable=False)
                vb.setYRange(-1.1 * rmax, 1.1 * rmax, padding=0)
        except Exception:
            pass

    def _refresh_legend(self):
        """Rebuild legend entries in order: Data -> Excluded -> Elastic -> Phonon 1..N -> Total."""
        try:
            pi = self.plot_widget.getPlotItem()
            if pi is None:
                return
            lg = getattr(pi, 'legend', None)
            if lg is None:
                try:
                    pi.addLegend(offset=(10, 10))
                    lg = pi.legend
                except Exception:
                    return
            # Clear current legend entries to avoid duplicates
            try:
                lg.clear()
            except Exception:
                pass
            # 1) Data
            if getattr(self, 'data_included_item', None) is not None:
                try:
                    lg.addItem(self.data_included_item, 'Data')
                except Exception:
                    pass
            # 2) Excluded Data
            if getattr(self, 'data_excluded_item', None) is not None:
                try:
                    lg.addItem(self.data_excluded_item, 'Excluded')
                except Exception:
                    pass
            # 3) Elastic baseline
            if getattr(self, 'background_line', None) is not None:
                try:
                    lg.addItem(self.background_line, 'Background')
                except Exception:
                    pass
            if getattr(self, 'elastic_line', None) is not None:
                try:
                    lg.addItem(self.elastic_line, 'Elastic')
                except Exception:
                    pass
            # 4) Phonon previews in numerical order
            for i, info in enumerate(self.draggable_artists):
                line = info.get('preview_line')
                if line is not None:
                    try:
                        lg.addItem(line, f'Phonon {i+1}')
                    except Exception:
                        pass
            # 5) Total fit/guess last
            if getattr(self, 'preview_line', None) is not None:
                try:
                    name = None
                    try:
                        name = self.preview_line.name()
                    except Exception:
                        name = None
                    lg.addItem(self.preview_line, name or 'Total')
                except Exception:
                    pass
        except Exception:
            pass

    # --- Fitting ---
    def run_fit(self):
        if self.energy is None:
            self.results_text.setText("Please load data before fitting.")
            return
        self.fit_button.setEnabled(False)
        self.results_text.setText("Fitting in progress...")
        initial_peaks = []
        for a in self.draggable_artists:
            dmp0 = clamp_damping(float(a.get('damping', DHO_DAMPING_DEFAULT)))
            initial_peaks.append({
                'center': a['center'],
                'height': a['height'],
                'damping': dmp0,
                'fix_E': bool(a.get('fix_E', False)),
                'fix_H': bool(a.get('fix_H', False)),
                'fix_D': bool(a.get('fix_D', False)),
            })
        self.file_info['T'] = self.file_info.get('T', 1.5)
        self.file_info['elastic_params'] = {
            'gauss_fwhm': self.gauss_fwhm_spinbox.value(),
            'lorentz_fwhm': self.lorentz_fwhm_spinbox.value(),
            'bg': self.bg_spinbox.value(),
            'height': self.elastic_height_spinbox.value()
        }
        self.file_info['fix_flags'] = {
            'gauss_fwhm': bool(getattr(self, 'fix_gauss_cb', None) and self.fix_gauss_cb.isChecked()),
            'lorentz_fwhm': bool(getattr(self, 'fix_lorentz_cb', None) and self.fix_lorentz_cb.isChecked()),
            'bg': bool(getattr(self, 'fix_bg_cb', None) and self.fix_bg_cb.isChecked()),
            'elastic_height': bool(getattr(self, 'fix_el_height_cb', None) and self.fix_el_height_cb.isChecked()),
        }
        errors = self.errors if self.errors is not None else np.sqrt(np.abs(self.counts))
        try:
            if isinstance(self.excluded_mask, np.ndarray) and len(self.excluded_mask) == len(self.energy):
                inc = ~self.excluded_mask
                energy_fit = self.energy[inc]
                counts_fit = self.counts[inc]
                errors_fit = errors[inc]
            else:
                energy_fit = self.energy
                counts_fit = self.counts
                errors_fit = errors
        except Exception:
            energy_fit = self.energy
            counts_fit = self.counts
            errors_fit = errors
        self.thread = QThread()
        self.worker = FitWorker(energy_fit, counts_fit, errors_fit, self.file_info, initial_peaks)
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)
        self.worker.result.connect(self.show_fit_results)
        self.worker.error.connect(self.fit_error)
        self.thread.finished.connect(lambda: self.fit_button.setEnabled(True))
        self.thread.start()

    def show_fit_results(self, param_dict, force_apply=False):
        chi2_before = self._compute_reduced_chi2_from_gui()
        chi2_after = self._compute_reduced_chi2_for_params(param_dict)
        text = ""
        for key, vals in param_dict.items():
            if 'fit' in vals:
                v = vals['fit']
                err = vals.get('err', None)
                if err is not None and np.isfinite(err):
                    text += f"{key}: {v:.4f} +/- {err:.4f}\n"
                else:
                    text += f"{key}: {v:.4f}\n"
        if chi2_after is not None and np.isfinite(chi2_after):
            text += f"chi2_red(new): {chi2_after:.4f}\n"
        if chi2_before is not None and np.isfinite(chi2_before):
            text += f"chi2_red(prev): {chi2_before:.4f}\n"
        self.results_text.setText(text)
        improve = force_apply or (chi2_before is None) or (chi2_after is None) or (chi2_after <= chi2_before * 0.999)
        if not improve:
            try:
                self.results_text.append("Fit not applied because chi^2 increased.")
            except Exception:
                pass
            return
        self.last_fit_params = param_dict
        try:
            fit_vals = {k: v['fit'] for k, v in param_dict.items() if 'fit' in v}
            if 'gauss_fwhm' in fit_vals:
                was = self.gauss_fwhm_spinbox.blockSignals(True)
                self.gauss_fwhm_spinbox.setValue(float(fit_vals['gauss_fwhm']))
                self.gauss_fwhm_spinbox.blockSignals(was)
            if 'lorentz_fwhm' in fit_vals:
                was = self.lorentz_fwhm_spinbox.blockSignals(True)
                self.lorentz_fwhm_spinbox.setValue(float(fit_vals['lorentz_fwhm']))
                self.lorentz_fwhm_spinbox.blockSignals(was)
            if 'BG' in fit_vals:
                was = self.bg_spinbox.blockSignals(True)
                self.bg_spinbox.setValue(float(fit_vals['BG']))
                self.bg_spinbox.blockSignals(was)
            if 'elastic_amplitude' in fit_vals:
                g = float(fit_vals.get('gauss_fwhm', self.gauss_fwhm_spinbox.value()))
                l = float(fit_vals.get('lorentz_fwhm', self.lorentz_fwhm_spinbox.value()))
                h0 = float(Voigt(np.array([0.0]), Area=float(fit_vals['elastic_amplitude']), gauss_fwhm=g, lorentz_fwhm=l, center=0.0)[0])
                was = self.elastic_height_spinbox.blockSignals(True)
                self.elastic_height_spinbox.setValue(h0)
                self.elastic_height_spinbox.blockSignals(was)
                if isinstance(self.last_fit_params, dict):
                    self.last_fit_params['elastic_amplitude'] = {
                        'fit': float(fit_vals['elastic_amplitude']),
                        'err': float(param_dict.get('elastic_amplitude', {}).get('err', np.nan))
                    }
            fitted_indices = []
            i = 0
            while True:
                if f'phonon_energy_{i}' in fit_vals and f'phonon_amplitude_{i}' in fit_vals:
                    fitted_indices.append(i)
                    i += 1
                else:
                    break
            n_fit = len(fitted_indices)
            if n_fit != len(self.draggable_artists):
                while len(self.draggable_artists) > n_fit:
                    info = self.draggable_artists.pop()
                    try:
                        if info.get('marker') is not None:
                            self.plot_widget.removeItem(info['marker'])
                    except Exception:
                        pass
                    try:
                        if info.get('preview_line') is not None:
                            self.plot_widget.removeItem(info['preview_line'])
                    except Exception:
                        pass
                while len(self.draggable_artists) < n_fit:
                    # When applying fitted peaks (e.g., on load), do not change selection
                    self.add_peak(select=not force_apply)
            Tval = float(self.file_info.get('T', 1.5))
            g = float(self.gauss_fwhm_spinbox.value())
            l = float(self.lorentz_fwhm_spinbox.value())
            for i in range(n_fit):
                info = self.draggable_artists[i]
                e = float(fit_vals[f'phonon_energy_{i}'])
                a = float(fit_vals[f'phonon_amplitude_{i}'])
                dmp = float(fit_vals.get(f'phonon_damping_{i}', info.get('damping', DHO_DAMPING_DEFAULT)))
                h = height_for_dho_voigt_area(a, abs(e), g, l, dmp, Tval, center=0.0, composition=COMPOSITION_DHO, x_grid=self.x_preview)
                info['center'] = abs(e)
                info['height'] = max(0.0, float(h))
                info['damping'] = float(dmp)
                info['area'] = float(a)
                for key, val in [('center_spin', info['center']), ('height_spin', info['height']), ('damping_spin', info['damping'])]:
                    w = info.get('widgets', {}).get(key)
                    if w is not None:
                        was = w.blockSignals(True)
                        w.setValue(val)
                        w.blockSignals(was)
            self.rebuild_peaks_panel()
            self.update_previews()
        except Exception:
            pass
        try:
            if self.preview_line is not None:
                self.preview_line.setName('Total Fit')
        except Exception:
            pass
        self.update_previews()
        try:
            self._update_chi2_label()
        except Exception:
            pass
        # If applying from saved fit, leave selection as None by default
        if force_apply:
            try:
                self.set_selected(None, None)
            except Exception:
                pass

    def fit_error(self, error_message):
        self.results_text.setText(error_message)

    def _compute_reduced_chi2_for_params(self, param_dict):
        try:
            if self.energy is None or self.counts is None:
                return None
            x, ydata, errs = self._get_included_data()
            vals = {k: v['fit'] for k, v in param_dict.items() if isinstance(v, dict) and 'fit' in v}
            g = float(vals.get('gauss_fwhm', self.gauss_fwhm_spinbox.value()))
            l = float(vals.get('lorentz_fwhm', self.lorentz_fwhm_spinbox.value()))
            bg = float(vals.get('BG', self.bg_spinbox.value()))
            Tval = float(self.file_info.get('T', 1.5))
            ea = float(vals.get('elastic_amplitude', elastic_area_from_height(self.elastic_height_spinbox.value(), g, l)))
            y = elastic_voigt_y(x, ea, g, l, center=0.0)
            i = 0
            n = 0
            while True:
                ek = f'phonon_energy_{i}'
                ak = f'phonon_amplitude_{i}'
                dk = f'phonon_damping_{i}'
                if ek in vals and ak in vals:
                    e = abs(float(vals[ek]))
                    a = float(vals[ak])
                    dmp = float(vals.get(dk, 0.1))
                    y += dho_y(x, e, g, l, dmp, a, Tval)
                    n += 1
                    i += 1
                else:
                    break
            y += bg
            r = (y - ydata) / errs
            chi2 = float(np.sum(r * r))
            p = 4 + 3 * n
            dof = max(1, len(x) - p)
            return chi2 / dof
        except Exception:
            return None

    def _update_chi2_label(self):
        try:
            chi2r = self._compute_reduced_chi2_from_gui()
            if chi2r is None or not np.isfinite(chi2r):
                self.chi2_label.setText("Chi^2_red: -")
            else:
                self.chi2_label.setText(f"Chi^2_red: {chi2r:.3f}")
        except Exception:
            pass

    def _compute_reduced_chi2_from_gui(self):
        if self.energy is None or self.counts is None:
            return None
        try:
            x, ydata, errs = self._get_included_data()
            g = float(self.gauss_fwhm_spinbox.value())
            l = float(self.lorentz_fwhm_spinbox.value())
            bg = float(self.bg_spinbox.value())
            Tval = float(self.file_info.get('T', 1.5))
            elastic_h = float(self.elastic_height_spinbox.value())
            elastic_a = elastic_area_from_height(elastic_h, g, l)
            y = elastic_voigt_y(x, elastic_a, g, l, center=0.0)
            n = 0
            for info in self.draggable_artists:
                e = abs(float(info.get('center', 0.0)))
                h = max(0.0, float(info.get('height', 0.0)))
                dmp = float(info.get('damping', DHO_DAMPING_DEFAULT))
                if 'area' in info and info['area'] is not None:
                    try:
                        a = float(info['area'])
                    except Exception:
                        a = dho_area_for_height_ui(h, e, g, l, dmp, Tval, self.x_preview if getattr(self, 'x_preview', None) is not None else x)
                else:
                    a = dho_area_for_height_ui(h, e, g, l, dmp, Tval, self.x_preview if getattr(self, 'x_preview', None) is not None else x)
                y += dho_y(x, e, g, l, dmp, a, Tval)
                n += 1
            y += bg
            r = (y - ydata) / errs
            chi2 = float(np.sum(r * r))
            p = 4 + 3 * n
            dof = max(1, len(x) - p)
            return chi2 / dof
        except Exception:
            return None

    def save_fit(self):
        try:
            try:
                out_dir = self.user_dirs.get('results_dir') if hasattr(self, 'user_dirs') else None
            except Exception:
                out_dir = None
            if not out_dir:
                out_dir = os.path.join(self._default_base_dir(), 'FMO Results')
            os.makedirs(out_dir, exist_ok=True)
            param_dict = self.last_fit_params if self.last_fit_params is not None else self._build_param_dict_from_gui()
            fit_vals = {k: v['fit'] for k, v in param_dict.items() if isinstance(v, dict) and 'fit' in v}
            err_vals = {k: v.get('err') for k, v in param_dict.items() if isinstance(v, dict) and ('err' in v)}
            try:
                missing = [k for k in fit_vals.keys() if (k not in err_vals) or (err_vals[k] is None) or (not np.isfinite(err_vals[k]))]
                if missing:
                    computed_errs = self._compute_errors_via_numeric_jacobian(param_dict)
                    if computed_errs:
                        for k in missing:
                            if k in computed_errs and np.isfinite(computed_errs[k]):
                                err_vals[k] = computed_errs[k]
                                if k in param_dict and isinstance(param_dict[k], dict):
                                    param_dict[k]['err'] = float(computed_errs[k])
            except Exception:
                pass
            base_name = os.path.splitext(self.file_info.get('name', 'fit'))[0]
            out_path = os.path.join(out_dir, f"{base_name}{RESULTS_SUFFIX}")
            chi2r = self._compute_reduced_chi2_from_gui()
            meta_lines = []
            meta_lines.append(f"Source file: {self.file_info.get('path', '')}")
            for k in ['scan', 'H', 'K', 'L', 'T', 'B']:
                if k in self.file_info:
                    meta_lines.append(f"{k}: {self.file_info[k]}")
            if chi2r is not None and np.isfinite(chi2r):
                meta_lines.append(f"Chi2_reduced: {chi2r:.6f}")
            def key_sort(k):
                if k.startswith('phonon_energy_'):
                    return (2, int(k.split('_')[-1]), 0)
                if k.startswith('phonon_damping_'):
                    return (2, int(k.split('_')[-1]), 1)
                if k.startswith('phonon_amplitude_'):
                    return (2, int(k.split('_')[-1]), 2)
                core = {
                    'BG': (0, 0),
                    'gauss_fwhm': (0, 1),
                    'lorentz_fwhm': (0, 2),
                    'elastic_amplitude': (1, 0),
                }
                return core.get(k, (9, k))
            lines = [f"{k}: {fit_vals[k]:.8g}" for k in sorted(fit_vals.keys(), key=key_sort)]
            content = []
            content.append("# Fit Results")
            content.extend(f"# {m}" for m in meta_lines)
            content.append("")
            content.extend(lines)
            try:
                errs_sorted = [k for k in sorted(err_vals.keys(), key=key_sort) if err_vals.get(k) is not None and np.isfinite(err_vals[k])]
                if errs_sorted:
                    content.append("")
                    content.append("# Uncertainties (1-sigma from Jacobian of weighted residuals)")
                    for k in errs_sorted:
                        content.append(f"# {k}_err: {err_vals[k]:.8g}")
            except Exception:
                pass
            try:
                if isinstance(self.excluded_mask, np.ndarray) and np.any(self.excluded_mask):
                    idxs = np.where(self.excluded_mask)[0]
                    content.append(f"ExcludedIndices: {','.join(str(int(i)) for i in idxs)}")
            except Exception:
                pass
            try:
                if os.path.exists(out_path):
                    os.remove(out_path)
            except Exception:
                pass
            with open(out_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(content))
            prev = self.results_text.toPlainText()
            msg = (prev + '\n' if prev else '') + f"Saved fit to: {out_path}"
            try:
                fig_dir = self.user_dirs.get('figures_dir') if hasattr(self, 'user_dirs') else None
            except Exception:
                fig_dir = None
            if not fig_dir:
                fig_dir = os.path.join(self._default_base_dir(), 'FMO Figures')
            os.makedirs(fig_dir, exist_ok=True)
            fig_path = os.path.join(fig_dir, f"{base_name}{FIGURE_SUFFIX}")
            try:
                exporter = ImageExporter(self.plot_widget.plotItem)
                exporter.parameters()['width'] = 1600
                exporter.export(fig_path)
                msg += f"\nSaved figure to: {fig_path}"
            except Exception:
                pass
            self.results_text.setPlainText(msg)
        except Exception as e:
            self.results_text.setPlainText(f"Error saving fit: {e}")

    def _load_previous_fit_if_exists(self):
        if not self.file_info or 'name' not in self.file_info:
            return False
        try:
            out_dir = self.user_dirs.get('results_dir') if hasattr(self, 'user_dirs') else None
        except Exception:
            out_dir = None
        if not out_dir:
            out_dir = os.path.join(self._default_base_dir(), 'FMO Results')
        base_name = os.path.splitext(self.file_info.get('name', 'fit'))[0]
        fit_path = os.path.join(out_dir, f"{base_name}{RESULTS_SUFFIX}")
        if not os.path.isfile(fit_path):
            return False
        vals = {}
        excluded_indices = []
        try:
            with open(fit_path, 'r', encoding='utf-8') as f:
                for line in f:
                    s = line.strip()
                    if not s or s.startswith('#'):
                        if s.lower().startswith('# excludedindices:'):
                            try:
                                part = s.split(':', 1)[1].strip()
                                if part:
                                    excluded_indices = [int(tok) for tok in part.split(',') if tok.strip().isdigit()]
                            except Exception:
                                pass
                        continue
                    if ':' not in s:
                        continue
                    k, v = s.split(':', 1)
                    k = k.strip()
                    if k.lower() == 'excludedindices':
                        try:
                            part = v.strip()
                            if part:
                                excluded_indices = [int(tok) for tok in part.split(',') if tok.strip().isdigit()]
                        except Exception:
                            pass
                        continue
                    try:
                        fv = float(v.strip())
                    except Exception:
                        continue
                    vals[k] = fv
        except Exception:
            return False
        if not vals:
            if excluded_indices and self.energy is not None:
                try:
                    self.excluded_mask = np.zeros(len(self.energy), dtype=bool)
                    for i in excluded_indices:
                        if 0 <= i < len(self.excluded_mask):
                            self.excluded_mask[i] = True
                    self._update_data_plot()
                except Exception:
                    pass
            return bool(excluded_indices)
        param_dict = {k: {'fit': v} for k, v in vals.items()}
        self.show_fit_results(param_dict, force_apply=True)
        if excluded_indices and self.energy is not None:
            try:
                self.excluded_mask = np.zeros(len(self.energy), dtype=bool)
                for i in excluded_indices:
                    if 0 <= i < len(self.excluded_mask):
                        self.excluded_mask[i] = True
                self._update_data_plot()
            except Exception:
                pass
        return True

    # --- Misc helpers from matplotlib version adapted ---
    def _build_param_dict_from_gui(self):
        self.file_info['T'] = self.file_info.get('T', 1.5)
        gauss = float(self.gauss_fwhm_spinbox.value())
        lorentz = float(self.lorentz_fwhm_spinbox.value())
        bg = float(self.bg_spinbox.value())
        elastic_height = float(self.elastic_height_spinbox.value())
        elastic_area = elastic_area_from_height(elastic_height, gauss, lorentz)
        param_dict = {
            'gauss_fwhm': {'fit': gauss},
            'lorentz_fwhm': {'fit': lorentz},
            'elastic_amplitude': {'fit': elastic_area},
            'BG': {'fit': bg},
        }
        for i, info in enumerate(self.draggable_artists):
            e = abs(float(info['center']))
            h = max(0.0, float(info['height']))
            dmp = float(info.get('damping', 0.1))
            a = dho_area_for_height_ui(h, e, gauss, lorentz, dmp, self.file_info['T'], self.x_preview)
            param_dict[f'phonon_energy_{i}'] = {'fit': e}
            param_dict[f'phonon_amplitude_{i}'] = {'fit': a}
            param_dict[f'phonon_damping_{i}'] = {'fit': dmp}
        return param_dict

    def _compute_errors_via_numeric_jacobian(self, param_dict):
        """
        Stub for computing errors via numeric Jacobian.
        Returns None for now - errors are computed in FitWorker.
        """
        return None

    def on_toggle_exclude_mode(self, checked):
        self.exclude_mode = bool(checked)
        self.exclude_button.setText("Done Excluding" if checked else "Exclude Points")
        try:
            # While entering exclusion mode, freeze current view ranges
            try:
                vb = self.plot_widget.getViewBox()
                if vb is not None:
                    vb.enableAutoRange(enable=False)
            except Exception:
                pass
            # Remove any leftover selection rectangle from the custom ViewBox
            try:
                vb = self.plot_widget.getViewBox()
                if hasattr(vb, '_exclude_rect') and vb._exclude_rect is not None:
                    try:
                        vb.removeItem(vb._exclude_rect)
                    except Exception:
                        pass
                    vb._exclude_rect = None
                    vb._exclude_active = False
                    vb._exclude_start = None
            except Exception:
                pass
            self.update_previews()
        except Exception:
            pass

    def reset_preview_defaults(self):
        """Reset spinboxes to defaults, clear peaks, keep data and redraw previews."""
        try:
            # Clear peaks without resetting view
            self.clear_peaks(reset_view=False)
            # Reset parameter inputs to defaults
            if hasattr(self, '_default_params'):
                try:
                    was = self.gauss_fwhm_spinbox.blockSignals(True)
                    self.gauss_fwhm_spinbox.setValue(float(self._default_params.get('gauss', self.gauss_fwhm_spinbox.value())))
                    self.gauss_fwhm_spinbox.blockSignals(was)
                except Exception:
                    pass
                try:
                    was = self.lorentz_fwhm_spinbox.blockSignals(True)
                    self.lorentz_fwhm_spinbox.setValue(float(self._default_params.get('lorentz', self.lorentz_fwhm_spinbox.value())))
                    self.lorentz_fwhm_spinbox.blockSignals(was)
                except Exception:
                    pass
                try:
                    was = self.bg_spinbox.blockSignals(True)
                    if self.counts is not None and len(self.counts) > 0:
                        self.bg_spinbox.setValue(float(np.nanmin(self.counts)))
                    else:
                        self.bg_spinbox.setValue(float(self._default_params.get('bg', self.bg_spinbox.value())))
                    self.bg_spinbox.blockSignals(was)
                except Exception:
                    pass
                try:
                    was = self.elastic_height_spinbox.blockSignals(True)
                    self.elastic_height_spinbox.setValue(float(self._default_params.get('elastic_height', self.elastic_height_spinbox.value())))
                    self.elastic_height_spinbox.blockSignals(was)
                except Exception:
                    pass
            # Restore preview label
            try:
                if self.preview_line is not None:
                    self.preview_line.setName('Total Guess')
            except Exception:
                pass
            # Deselect and redraw
            self.set_selected(None, None)
            self.update_previews()
        except Exception:
            pass

    def _toggle_nearest_point_exclusion_xy(self, x, y, pixel_threshold=EXCLUDE_CLICK_PIXEL_THRESHOLD):
        if self.energy is None or self.counts is None:
            return
        try:
            # Find nearest point in data coordinates (simple distance in x)
            xdata = self.energy
            ydata = self.counts
            diffs = np.abs(xdata - x)
            best_i = int(np.argmin(diffs))
            
            # Verify it's within pixel threshold
            vb = self.plot_widget.getViewBox()
            if vb is not None:
                # Get pixel position of the point and click
                point_px = vb.mapViewToDevice(pg.Point(xdata[best_i], ydata[best_i]))
                click_px = vb.mapViewToDevice(pg.Point(x, y))
                pixel_dist = np.hypot(point_px.x() - click_px.x(), point_px.y() - click_px.y())
                
                if pixel_dist <= pixel_threshold:
                    if self.excluded_mask is None or len(self.excluded_mask) != len(xdata):
                        self.excluded_mask = np.zeros(len(xdata), dtype=bool)
                    self.excluded_mask[best_i] = not self.excluded_mask[best_i]
                    # Do not re-range when toggling a single point; keep view fixed
                    self._update_data_plot(do_range=False)
                    self.update_previews()
        except Exception:
            pass

    def _toggle_exclude_shortcut(self):
        try:
            self.exclude_button.setChecked(not self.exclude_button.isChecked())
        except Exception:
            # Fallback to direct toggle
            self.on_toggle_exclude_mode(not getattr(self, 'exclude_mode', False))

    def _nearest_target_xy(self, x, y, pixel_threshold=NEAREST_TARGET_PIXEL_THRESHOLD):
        if self.plot_widget is None:
            return (None, None)
        vb = self.plot_widget.getViewBox()
        if vb is None:
            return (None, None)
        
        best_kind, best_obj, best_dist = None, None, float('inf')
        elastic_height = self.elastic_height_spinbox.value()
        gauss_fwhm = self.gauss_fwhm_spinbox.value()
        lorentz_fwhm = self.lorentz_fwhm_spinbox.value()
        bg = self.bg_spinbox.value()
        
        # Check phonon markers - calculate distance in pixel space
        for info in self.draggable_artists:
            baseline_center = elastic_baseline_at_energy(info['center'], elastic_height, gauss_fwhm, lorentz_fwhm, bg)
            marker_y = baseline_center + info['height']
            # Convert both points to device (pixel) coordinates
            marker_px = vb.mapViewToDevice(pg.Point(info['center'], marker_y))
            click_px = vb.mapViewToDevice(pg.Point(x, y))
            d = np.hypot(marker_px.x() - click_px.x(), marker_px.y() - click_px.y())
            if d < best_dist:
                best_kind, best_obj, best_dist = 'phonon', info, d
                
        # Elastic proximity (distance to curve at x) - calculate in pixel space
        try:
            if self.elastic_line is not None:
                # Interpolate y on elastic baseline
                X, Y = self.elastic_line.getData()
                X = X if X is not None else []
                Y = Y if Y is not None else []
                if len(X) > 1:
                    idx = np.searchsorted(X, x)
                    if 0 < idx < len(X):
                        t = (x - X[idx-1]) / (X[idx] - X[idx-1] + 1e-12)
                        y_el = Y[idx-1] * (1 - t) + Y[idx] * t
                    else:
                        y_el = Y[0] if idx <= 0 else Y[-1]
                    # Convert to pixel coordinates for distance calculation
                    curve_px = vb.mapViewToDevice(pg.Point(x, y_el))
                    click_px = vb.mapViewToDevice(pg.Point(x, y))
                    d = np.hypot(curve_px.x() - click_px.x(), curve_px.y() - click_px.y())
                    if d < best_dist:
                        best_kind, best_obj, best_dist = 'elastic', None, d
        except Exception:
            pass
            
        return (best_kind, best_obj) if best_dist <= pixel_threshold else (None, None)

    # --- Reset zoom similar to matplotlib version ---
    def _reset_zoom(self):
        try:
            if self.energy is None or self.counts is None or len(self.energy) == 0:
                vb = self.plot_widget.getViewBox()
                if vb:
                    vb.enableAutoRange()
                return
            x = np.asarray(self.energy, dtype=float)
            y = np.asarray(self.counts, dtype=float)
            if isinstance(self.excluded_mask, np.ndarray) and len(self.excluded_mask) == len(y):
                y = y[~self.excluded_mask]
            xmin, xmax = float(np.nanmin(x)), float(np.nanmax(x))
            ymin, ymax = float(np.nanmin(y)), float(np.nanmax(y))
            # Include model lines if present to avoid clipping on reset
            try:
                for item in (self.background_line, self.elastic_line, self.preview_line):
                    if item is None:
                        continue
                    X, Y = item.getData()
                    if X is None or Y is None or len(X) == 0:
                        continue
                    xmin = min(xmin, float(np.nanmin(X)))
                    xmax = max(xmax, float(np.nanmax(X)))
                    ymin = min(ymin, float(np.nanmin(Y)))
                    ymax = max(ymax, float(np.nanmax(Y)))
            except Exception:
                pass
            xr = xmax - xmin if np.isfinite(xmax - xmin) and (xmax - xmin) > 0 else 1.0
            yr = ymax - ymin if np.isfinite(ymax - ymin) and (ymax - ymin) > 0 else 1.0
            
            vb = self.plot_widget.getViewBox()
            if vb:
                vb.enableAutoRange(enable=False)
                vb.setXRange(xmin - X_MARGIN_FRAC * xr, xmax + X_MARGIN_FRAC * xr, padding=0)
                vb.setYRange(ymin - Y_MARGIN_LOW * yr, ymax + Y_MARGIN_HIGH * yr, padding=0)
        except Exception:
            pass

    def toggle_help_panel(self):
        """Show/hide the right-hand help panel and adjust splitter sizes."""
        try:
            current = bool(self.help_widget.isVisible())
        except Exception:
            current = True
        # Determine target width
        try:
            help_w = int(getattr(self, '_help_panel_width', 0))
            if help_w <= 0:
                help_w = int(self.help_widget.sizeHint().width())
                if help_w <= 0:
                    help_w = DEFAULT_HELP_PANEL_WIDTH
        except Exception:
            help_w = DEFAULT_HELP_PANEL_WIDTH
        # Parent splitter that contains plot and help
        try:
            right_splitter = self.help_widget.parent()
        except Exception:
            right_splitter = None
        if current:
            try:
                self.help_widget.setVisible(False)
            except Exception:
                pass
            try:
                if isinstance(right_splitter, QSplitter):
                    sizes = right_splitter.sizes()
                    if len(sizes) >= 2:
                        right_splitter.setSizes([sum(sizes), 0])
            except Exception:
                pass
            try:
                self.help_toggle_button.setText("Show Help")
            except Exception:
                pass
        else:
            try:
                self.help_widget.setVisible(True)
            except Exception:
                pass
            try:
                if isinstance(right_splitter, QSplitter):
                    sizes = right_splitter.sizes()
                    total = sum(sizes) if sizes and sum(sizes) > 0 else (DEFAULT_WINDOW_WIDTH - MIN_LEFT_PANEL_WIDTH)
                    right_splitter.setSizes([max(300, total - help_w), help_w])
            except Exception:
                pass
            try:
                self.help_toggle_button.setText("Hide Help")
            except Exception:
                pass

    # ... (many more methods need conversion)

if __name__ == "__main__":
    # Reuse the existing QApplication singleton created above
    app = QApplication.instance() or pg.mkQApp("Interactive Peak Fitter (PyQtGraph)")
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
