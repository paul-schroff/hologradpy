"""Backend-agnostic analytic field profiles on a 2D grid.

Pure functions and small helpers that build amplitude/intensity profiles, phase
profiles and Zernike polynomials from coordinate grids. They are used well beyond the
propagation backend (calibration, holography, analysis), so they live at the top level
rather than inside :mod:`hologradpy.optics`. Coordinate grids come from the
:mod:`hologradpy.grids` leaf.
"""

from .amplitude import (
    get_focal_spot_radius,
    gaussian_beam_intensity,
    super_gaussian,
    gaussian_spot_array,
    gaussian_ring,
    top_hat_gaussian_shoulders,
    top_hat_2D,
    gaussian_blur,
    laser_speckle_intensity,
    checkerboard,
)
from .masks import (
    rectangular_mask,
    circular_mask,
)
from .phase import (
    tilt_to_angle,
    lens_phase,
    spherical_surface,
    doublet_lens,
    linear_phase,
    quadratic_phase,
    analytic_phase_guess,
    binary_phase_grating,
)
from .zernike import (
    ZernikeConventionHandler,
    Zernike,
    Conventions,
    make_per_wavelength_coefficients,
)

__all__ = [
    "get_focal_spot_radius",
    "gaussian_beam_intensity",
    "super_gaussian",
    "gaussian_spot_array",
    "gaussian_ring",
    "top_hat_gaussian_shoulders",
    "top_hat_2D",
    "rectangular_mask",
    "circular_mask",
    "gaussian_blur",
    "laser_speckle_intensity",
    "checkerboard",
    "tilt_to_angle",
    "lens_phase",
    "spherical_surface",
    "doublet_lens",
    "linear_phase",
    "quadratic_phase",
    "analytic_phase_guess",
    "binary_phase_grating",
    "ZernikeConventionHandler",
    "Zernike",
    "Conventions",
    "make_per_wavelength_coefficients",
]
