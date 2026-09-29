"""Tests for fitting a Gaussian beam intensity to the pixels kept by a mask."""

from __future__ import annotations

import numpy as np
import pytest

from hologradpy.analysis.fitting import fit_gaussian_beam_intensity
from hologradpy.profiles.amplitude import gaussian_beam_intensity

SHAPE = (81, 81)
SPOT = (40.3, 38.7)  # (x, y) in pixels
SPOT_RADIUS = 6.0
NEIGHBOUR = (60.0, 40.0)  # a brighter spot beside it, (x, y) in pixels
NEIGHBOUR_MASK_RADIUS = 12.0


def _spot_beside_a_brighter_neighbour() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    y, x = np.indices(SHAPE, dtype=np.float64)
    data = gaussian_beam_intensity(x, y, SPOT_RADIUS, *SPOT, 100.0, 5.0)
    data += gaussian_beam_intensity(x, y, 5.0, *NEIGHBOUR, 2000.0)
    return x, y, data


def test_a_masked_neighbour_leaves_the_fit_on_the_spot() -> None:
    """The fit starts on the spot and fits it alone. This holds although the neighbour
    is brighter and lies inside the blur used to find the starting peak.
    """
    x, y, data = _spot_beside_a_brighter_neighbour()
    kept = (x - NEIGHBOUR[0]) ** 2 + (y - NEIGHBOUR[1]) ** 2 > NEIGHBOUR_MASK_RADIUS**2

    popt, _ = fit_gaussian_beam_intensity(x, y, data, 5.0, mask=kept)

    radius, shift_x, shift_y, intensity, offset = popt
    assert (shift_x, shift_y) == pytest.approx(SPOT, abs=0.01)
    assert radius == pytest.approx(SPOT_RADIUS, rel=1e-3)
    assert intensity == pytest.approx(100.0, rel=1e-3)
    assert offset == pytest.approx(5.0, abs=0.05)


def test_without_a_mask_the_fit_starts_on_the_brighter_neighbour() -> None:
    x, y, data = _spot_beside_a_brighter_neighbour()

    popt, _ = fit_gaussian_beam_intensity(x, y, data, 5.0)

    assert np.hypot(popt[1] - SPOT[0], popt[2] - SPOT[1]) > 5.0


def test_a_mask_of_every_pixel_fits_as_no_mask() -> None:
    x, y, data = _spot_beside_a_brighter_neighbour()
    data = gaussian_beam_intensity(x, y, SPOT_RADIUS, *SPOT, 100.0, 5.0)

    unmasked, _ = fit_gaussian_beam_intensity(x, y, data, 5.0)
    masked, _ = fit_gaussian_beam_intensity(
        x, y, data, 5.0, mask=np.ones(SHAPE, dtype=bool)
    )

    np.testing.assert_allclose(masked, unmasked, rtol=1e-6)
