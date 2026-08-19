"""Optical-lattice model/fit and the pointing-compensation coordinate shift.

``optical_lattice_fringes`` models the 2D interference of the four corner
superpixels and must be invertible by ``fit_optical_lattice_fringes``. The
pointing correction works by converting the lattice phase drift to a camera
displacement and fitting the main fringes on shifted coordinates; the second
test validates that coordinate-shift mechanism in isolation.
"""

import numpy as np
import pytest

from hologradpy.analysis.fitting import (
    optical_lattice_fringes,
    fit_optical_lattice_fringes,
    interferometric_fringes,
    fit_interferometric_fringes,
)

WAVENUMBER = 2 * np.pi / 800e-9
FOCAL_LENGTH = 0.25


def _grid(n: int = 64, half_extent: float = 60e-6):
    coords = np.linspace(-half_extent, half_extent, n)
    return np.meshgrid(coords, coords)


def test_fit_optical_lattice_recovers_phases():
    x, y = _grid()
    separation_x, separation_y = 8e-3, 6e-3
    true_phase_x, true_phase_y, amplitude = 0.7, -1.1, 2.0

    data = optical_lattice_fringes(
        x,
        y,
        separation_x,
        separation_y,
        WAVENUMBER,
        FOCAL_LENGTH,
        true_phase_x,
        true_phase_y,
        amplitude,
    )
    popt, _ = fit_optical_lattice_fringes(
        x,
        y,
        data,
        separation_x,
        separation_y,
        WAVENUMBER,
        FOCAL_LENGTH,
        amplitude_guess=amplitude,
    )

    assert popt[0] == pytest.approx(true_phase_x, abs=1e-3)
    assert popt[1] == pytest.approx(true_phase_y, abs=1e-3)
    assert abs(popt[2]) == pytest.approx(amplitude, abs=1e-2)


def test_coordinate_shift_recovers_phase_under_drift():
    """A pointing shift moves the camera pattern by a displacement; fitting the
    main fringes on coordinates ``grid - displacement`` recovers the true phase,
    while the naive (unshifted) fit is biased.
    """
    x, y = _grid()
    separation_x, separation_y = 2e-3, 1.5e-3  # coarse main pattern
    true_phase, amplitude = 0.5, 3.0
    drift_x, drift_y = 10e-6, -8e-6

    # The camera pattern is shifted by the pointing displacement.
    data = interferometric_fringes(
        x - drift_x,
        y - drift_y,
        separation_x,
        separation_y,
        WAVENUMBER,
        FOCAL_LENGTH,
        true_phase,
        amplitude,
    )

    popt_corrected, _ = fit_interferometric_fringes(
        x - drift_x,
        y - drift_y,
        data,
        separation_x,
        separation_y,
        WAVENUMBER,
        FOCAL_LENGTH,
        amplitude_guess=amplitude,
    )
    popt_naive, _ = fit_interferometric_fringes(
        x,
        y,
        data,
        separation_x,
        separation_y,
        WAVENUMBER,
        FOCAL_LENGTH,
        amplitude_guess=amplitude,
    )

    assert popt_corrected[0] == pytest.approx(true_phase, abs=1e-3)
    # The uncorrected fit is biased by the pointing-induced phase.
    assert abs(popt_naive[0] - true_phase) > 0.05
