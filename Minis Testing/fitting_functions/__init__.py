# Directory structure for the package:
# FittingFunctions/
# ├── __init__.py
# ├── fitting_functions.py

# __init__.py
from .functions import (
    Gaussian,
    Lorentzian,
    Voigt,
    Voigt_wBG,
    voigt_area_for_height,
    voigt_height_for_area,
    dho_voigt_area_for_height,
    dho_voigt_height_for_area,
    Stokes_DHO,
    antiStokes_DHO,
    weighted_discrete_delta,
    narrow_gaussian_delta,
    convolute_voigt_dho,
    convolute_gaussian_dho
)