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