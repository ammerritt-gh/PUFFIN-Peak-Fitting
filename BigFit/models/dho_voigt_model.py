# dho_voigt_model.py
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional

# import the functions you already wrote (assumed available)
from .models import Voigt, voigt_area_for_height, convolute_voigt_dho
# (use your actual module path)
# from .voigt import Voigt, voigt_area_for_height
# from .convolutions import convolute_voigt_dho
# from .utils import kB

@dataclass
class Phonon:
    center: float            # phonon energy (meV)
    height: Optional[float]  # UI prefers height; if None, area must be set
    area: Optional[float]    # area for convolution; if None, computed from height
    damping: float = 0.1
    fix_E: bool = False
    fix_D: bool = False
    fix_H: bool = False

    def snapshot(self):
        return dict(center=self.center, height=self.height, area=self.area, damping=self.damping,
                    fix_E=self.fix_E, fix_D=self.fix_D, fix_H=self.fix_H)


class DhoVoigtComposite:
    """
    Composite model: elastic Voigt (centered at elastic_center) + sum of DHO phonons,
    each convolved with the Voigt instrument resolution (gauss_fwhm, lorentz_fwhm).

    This class delegates convolution to your `convolute_voigt_dho` function,
    but centralizes parameter storage, area<->height helpers, and evaluation.
    """

    def __init__(self,
                 gauss_fwhm: float = 1.14,
                 lorentz_fwhm: float = 0.28,
                 bg: float = 0.0,
                 elastic_height: float = 100.0,
                 elastic_center: float = 0.0,
                 T: float = 10.0,
                 convolution_dx: float = 0.05,
                 convolution_pad: float = 20.0):
        self.gauss_fwhm = float(gauss_fwhm)
        self.lorentz_fwhm = float(lorentz_fwhm)
        self.bg = float(bg)
        self.elastic_height = float(elastic_height)
        self.elastic_center = float(elastic_center)
        self.T = float(T)

        # phonons list
        self.peaks: List[Phonon] = []

        # caching grid/resolution kernel params for speed (invalidated when gauss/lorentz change)
        self._cache = {'gauss': None, 'lorentz': None, 'kernel': None,
                       'dx': float(convolution_dx), 'pad': float(convolution_pad)}

    # -------------------------
    # Peak management
    # -------------------------
    def add_peak(self, center: float, height: Optional[float] = None,
                 area: Optional[float] = None, damping: float = 0.1,
                 fix_E=False, fix_D=False, fix_H=False) -> Phonon:
        p = Phonon(center=float(center), height=(None if height is None else float(height)),
                   area=(None if area is None else float(area)), damping=float(damping),
                   fix_E=bool(fix_E), fix_D=bool(fix_D), fix_H=bool(fix_H))
        # if area not provided but height is, compute area consistent with current resolution
        if p.area is None and p.height is not None:
            p.area = self.area_for_phonon_height(p.height, p.center)
        self.peaks.append(p)
        return p

    def remove_peak(self, index: int):
        del self.peaks[index]

    def clear_peaks(self):
        self.peaks = []

    # -------------------------
    # Conversion helpers
    # -------------------------
    def elastic_area_for_height(self, height: float) -> float:
        """Return area for a Voigt elastic peak with given height using current resolution."""
        return float(voigt_area_for_height(float(height), self.gauss_fwhm, self.lorentz_fwhm))

    def area_for_phonon_height(self, height: float, phonon_energy: float,
                               x_grid: Optional[np.ndarray] = None) -> float:
        """
        Estimate the DHO area that yields the requested height at phonon_energy.
        Uses your convolute_voigt_dho under the hood; uses a small local grid if x_grid is None.
        """
        # Delegate to your dho_voigt_area_for_height if you have it; else use local approach:
        if x_grid is None:
            x_grid = np.linspace(phonon_energy - 5.0, phonon_energy + 5.0, 1200)
        # call convolute with test area = 1.0 (convoluted gives y(x) for that area)
        y = convolute_voigt_dho(
            x_grid, float(phonon_energy), float(self.elastic_center),
            float(self.gauss_fwhm), float(self.lorentz_fwhm),
            float(0.1), 1.0, 0.0, 0.0, float(self.T), peak='stokes+astokes'
        )
        # find height at center + phonon_energy
        target_x = float(phonon_energy) + float(self.elastic_center)
        idx = int(np.argmin(np.abs(x_grid - target_x)))
        y_at = float(y[idx]) if (y is not None and len(y) > idx) else 0.0
        return (1.0 * (height / y_at)) if y_at > 0 else 0.0

    # -------------------------
    # Evaluation
    # -------------------------
    def evaluate(self, x: np.ndarray) -> np.ndarray:
        """
        Return y(x) = sum_{peaks} convoluted_DHO(x) + elastic Voigt(x) + BG
        - x: 1D numpy array
        """
        x = np.asarray(x, dtype=float)
        # Elastic: convert height -> area and evaluate Voigt directly at elastic_center
        elastic_area = self.elastic_area_for_height(self.elastic_height)
        y = Voigt(x, Area=elastic_area, gauss_fwhm=self.gauss_fwhm,
                  lorentz_fwhm=self.lorentz_fwhm, center=self.elastic_center)

        # Add each phonon contribution computed with convolution
        for pk in self.peaks:
            area = float(pk.area) if (pk.area is not None and pk.area > 0) else \
                   (self.area_for_phonon_height(pk.height, pk.center) if pk.height is not None else 0.0)

            # Use your convolution helper; it already returns an array interpolated to x
            y += convolute_voigt_dho(
                x, float(pk.center), float(self.elastic_center),
                float(self.gauss_fwhm), float(self.lorentz_fwhm),
                float(pk.damping), float(area), float(elastic_area), float(self.bg),
                float(self.T), peak='stokes+astokes'
            )
        # Add background (if not already added by convolute – we passed BG but ensure again)
        # Note: we passed BG into convolute_voigt_dho; it adds BG. If you prefer to handle BG once:
        # y = y + 0.0  # BG already included
        return y

    # -------------------------
    # Snapshot / serialization
    # -------------------------
    def snapshot(self) -> dict:
        return {
            'gauss_fwhm': self.gauss_fwhm,
            'lorentz_fwhm': self.lorentz_fwhm,
            'bg': self.bg,
            'elastic_height': self.elastic_height,
            'elastic_center': self.elastic_center,
            'T': self.T,
            'peaks': [p.snapshot() for p in self.peaks],
        }

    @classmethod
    def from_snapshot(cls, snap: dict):
        inst = cls(
            gauss_fwhm=snap.get('gauss_fwhm', 1.14),
            lorentz_fwhm=snap.get('lorentz_fwhm', 0.28),
            bg=snap.get('bg', 0.0),
            elastic_height=snap.get('elastic_height', 100.0),
            elastic_center=snap.get('elastic_center', 0.0),
            T=snap.get('T', 10.0)
        )
        for pk in snap.get('peaks', []):
            inst.add_peak(center=pk.get('center', 0.0), height=pk.get('height', None),
                          area=pk.get('area', None), damping=pk.get('damping', 0.1))
        return inst

