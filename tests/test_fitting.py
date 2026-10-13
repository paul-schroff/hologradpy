"""Tests for the Gaussian beam: drawing it, fitting it, also to the pixels kept by a mask,
and the uncertainty of its quantities.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from hologradpy.analysis.uncertainties import standard_deviation
from hologradpy.profiles.gaussian import (
    Gaussian,
    GaussianBeam,
    GaussianFit,
    fit_focus,
    radius_along_beam,
)
from hologradpy.roi import ROI

SHAPE = (81, 81)
# A grid in pixel units: x to the right, y up, and 0 at the centre pixel.
GAUSSIAN = Gaussian(SHAPE, (1.0, 1.0))
X = GAUSSIAN.x.numpy().astype(np.float64)
Y = GAUSSIAN.y.numpy().astype(np.float64)
SPOT = (0.3, 1.3)  # (x, y) in pixels
SPOT_RADIUS = 6.0
NEIGHBOUR = (20.0, 0.0)  # a brighter spot beside it, (x, y) in pixels
NEIGHBOUR_MASK_RADIUS = 12.0


def _spot_beside_a_brighter_neighbour() -> np.ndarray:
    spot = GaussianBeam(SPOT_RADIUS, None, *SPOT, 100.0, 5.0)
    neighbour = GaussianBeam(5.0, None, *NEIGHBOUR, 2000.0)
    return spot.evaluate(X, Y) + neighbour.evaluate(X, Y)


def _turned_ellipse(angle_degrees: float, offset: float = 5.0) -> GaussianBeam:
    return GaussianBeam(9.0, 4.0, *SPOT, 100.0, offset, angle_degrees)


def test_a_masked_neighbour_leaves_the_fit_on_the_spot() -> None:
    """The fit starts on the spot and fits it alone. This holds although the neighbour
    is brighter and lies inside the blur used to find the starting peak.
    """
    kept = (X - NEIGHBOUR[0]) ** 2 + (Y - NEIGHBOUR[1]) ** 2 > NEIGHBOUR_MASK_RADIUS**2

    beam = GAUSSIAN.fit(_spot_beside_a_brighter_neighbour(), 5.0, mask=kept).beam

    assert (beam.shift_x, beam.shift_y) == pytest.approx(SPOT, abs=0.01)
    assert beam.beam_radius_major == pytest.approx(SPOT_RADIUS, rel=1e-3)
    assert beam.beam_radius_minor == pytest.approx(SPOT_RADIUS, rel=1e-3)
    assert beam.intensity == pytest.approx(100.0, rel=1e-3)
    assert beam.offset == pytest.approx(5.0, abs=0.05)


def test_without_a_mask_the_fit_starts_on_the_brighter_neighbour() -> None:
    beam = GAUSSIAN.fit(_spot_beside_a_brighter_neighbour(), 5.0).beam

    assert np.hypot(beam.shift_x - SPOT[0], beam.shift_y - SPOT[1]) > 5.0


def test_a_mask_of_every_pixel_fits_as_no_mask() -> None:
    data = GaussianBeam(SPOT_RADIUS, None, *SPOT, 100.0, 5.0).evaluate(X, Y)

    unmasked = GAUSSIAN.fit(data, 5.0).beam
    masked = GAUSSIAN.fit(data, 5.0, mask=np.ones(SHAPE, dtype=bool)).beam

    for beam in (masked, unmasked):
        assert (beam.shift_x, beam.shift_y) == pytest.approx(SPOT, abs=1e-5)
        assert (beam.mean_radius, beam.intensity) == pytest.approx(
            (SPOT_RADIUS, 100.0), rel=1e-6
        )
    assert masked.offset == pytest.approx(unmasked.offset, abs=1e-6)


@pytest.mark.parametrize("angle_degrees", [-60.0, -15.0, 0.0, 30.0, 90.0])
def test_a_turned_ellipse_is_recovered_through_noise(angle_degrees) -> None:
    """The radii, the angle of the long axis, the centre, the peak and the offset."""
    noise = np.random.default_rng(1).normal(0.0, 1.0, SHAPE)
    data = _turned_ellipse(angle_degrees).evaluate(X, Y)

    fit = GAUSSIAN.fit(data + noise, 5.0)

    beam = fit.beam
    assert (beam.beam_radius_major, beam.beam_radius_minor) == pytest.approx(
        (9.0, 4.0), rel=0.01
    )
    # The long axis is an axis, so the angle is found to within half a turn.
    assert -90.0 < beam.angle_degrees <= 90.0
    turn = (beam.angle_degrees - angle_degrees + 90.0) % 180.0 - 90.0
    assert turn == pytest.approx(0.0, abs=0.5)
    assert (beam.shift_x, beam.shift_y) == pytest.approx(SPOT, abs=0.05)
    assert beam.intensity == pytest.approx(100.0, rel=0.02)
    assert beam.offset == pytest.approx(5.0, abs=0.2)
    assert 0.0 < standard_deviation(fit, lambda fit: fit.beam.mean_radius) < 0.05


def test_a_round_spot_gives_equal_radii() -> None:
    data = GaussianBeam(SPOT_RADIUS, None, *SPOT, 100.0, 5.0).evaluate(X, Y)

    beam = GAUSSIAN.fit(data, 4.0).beam

    assert beam.beam_radius_major == pytest.approx(beam.beam_radius_minor, rel=1e-6)
    assert beam.mean_radius == pytest.approx(SPOT_RADIUS, rel=1e-6)


def test_get_intensity_gives_back_the_fitted_intensity() -> None:
    """The fit is the inverse of get_intensity."""
    data = _turned_ellipse(30.0).evaluate(X, Y)

    fit = GAUSSIAN.fit(data, 5.0)

    # get_intensity draws on the float32 grid.
    drawn = GAUSSIAN.get_intensity(fit.beam).numpy()
    np.testing.assert_allclose(drawn, data, atol=1e-4)


def test_a_fit_on_a_region_finds_the_beam_in_the_whole_grid() -> None:
    """The region's grid is the whole grid, cropped, so the centre is the whole grid's
    position.
    """
    beam = GaussianBeam(5.0, 3.0, 20.3, -15.6, 100.0, 2.0, 25.0)
    roi = ROI.from_bounds(41, 71, 46, 76)  # Rows 41 to 70 and columns 46 to 75.

    region = Gaussian(SHAPE, (1.0, 1.0), roi)
    fitted = region.fit(roi.crop(beam.evaluate(X, Y)), 3.0, blur_sigma=2.0).beam

    assert (fitted.shift_x, fitted.shift_y) == pytest.approx((20.3, -15.6), abs=1e-5)
    assert (
        fitted.beam_radius_major,
        fitted.beam_radius_minor,
        fitted.angle_degrees,
    ) == pytest.approx((5.0, 3.0, 25.0), rel=1e-5)


def test_the_x_and_y_radii_are_those_of_the_beam_summed_over_the_other_axis() -> None:
    """Each radius is twice the standard deviation of the summed profile."""
    beam = _turned_ellipse(30.0, offset=0.0)
    data = beam.evaluate(X, Y)

    for radius, position, profile in (
        (beam.beam_radius_x, X[0], data.sum(axis=0)),
        (beam.beam_radius_y, Y[:, 0], data.sum(axis=1)),
    ):
        mean = np.average(position, weights=profile)
        variance = np.average((position - mean) ** 2, weights=profile)
        assert radius == pytest.approx(2.0 * np.sqrt(variance), rel=1e-6)


def test_the_mean_radius_is_the_geometric_mean_of_the_radii() -> None:
    assert GaussianBeam(9.0, 4.0, angle_degrees=30.0).mean_radius == pytest.approx(6.0)


def test_every_radius_of_a_round_beam_is_its_radius() -> None:
    beam = GaussianBeam(SPOT_RADIUS, angle_degrees=30.0)

    radii = (beam.mean_radius, beam.beam_radius_x, beam.beam_radius_y)
    assert radii == pytest.approx((SPOT_RADIUS,) * 3)


def test_a_minor_radius_above_the_major_one_is_rejected() -> None:
    with pytest.raises(ValueError, match="larger than the major radius"):
        GaussianBeam(4.0, 9.0)


def test_the_mean_radius_std_is_its_analytic_propagation() -> None:
    """The mean radius is 1/sqrt(a c), with gradient -m/2 (1/a, 0, 1/c) in (a, s, c)."""
    noise = np.random.default_rng(1).normal(0.0, 1.0, SHAPE)
    data = _turned_ellipse(30.0).evaluate(X, Y)

    fit = GAUSSIAN.fit(data + noise, 5.0)

    a, _, c = fit.parameters[:3]
    mean_radius = 1.0 / math.sqrt(abs(a * c))
    gradient = -0.5 * mean_radius * np.array([1.0 / a, 0.0, 1.0 / c])
    expected = math.sqrt(gradient @ fit.covariance[:3, :3] @ gradient)
    std = standard_deviation(fit, lambda fit: fit.beam.mean_radius)
    assert std == pytest.approx(expected, rel=1e-4)


def test_the_angle_std_wraps_at_90_degrees() -> None:
    """A beam turned by exactly 90 degrees sits at the edge of (-90, 90]. Small changes
    of the parameters turn it to either side of the edge. With the period of 180
    degrees, the changes wrap.
    """
    # L = [[1/4, 0], [0, 1/9]] is a 9 x 4 beam with its long axis along y.
    parameters = np.array([1 / 4, 0.0, 1 / 9, 0.0, 0.0, 1.0, 0.0])
    fit = GaussianFit(parameters, np.diag(np.full(7, 1e-6)))

    std = standard_deviation(fit, lambda fit: fit.beam.angle_degrees, period=180.0)

    assert fit.beam.angle_degrees == pytest.approx(90.0)
    # The shear's standard deviation of 1e-3 turns the axis by a / (a^2 - c^2) radians
    # per unit of shear, which is 0.29 degrees.
    assert std == pytest.approx(0.2848, rel=0.01)


# A waist of 20 um, 0.4 mm along the beam, with a Rayleigh range of 1.2 mm.
FOCUS = (20e-6, 0.4e-3, 1.2e-3)
FOCUS_POSITIONS = np.linspace(-3e-3, 3e-3, 13)


def test_a_focus_is_recovered_from_its_radii() -> None:
    radii = radius_along_beam(FOCUS_POSITIONS, *FOCUS)

    fit = fit_focus(FOCUS_POSITIONS, radii)

    np.testing.assert_allclose(
        (fit.waist_radius, fit.waist_position, fit.rayleigh_range), FOCUS, rtol=1e-6
    )
    np.testing.assert_allclose(fit.radius(FOCUS_POSITIONS), radii, rtol=1e-6)


def test_a_focus_is_recovered_through_noise() -> None:
    """Radii with 1% noise give the focus back to within the noise. The waist
    position's standard deviation is a fraction of the Rayleigh range.
    """
    noise = np.random.default_rng(0).standard_normal(FOCUS_POSITIONS.size)
    radii = radius_along_beam(FOCUS_POSITIONS, *FOCUS) * (1.0 + 0.01 * noise)

    fit = fit_focus(FOCUS_POSITIONS, radii)
    std = standard_deviation(fit, lambda fit: fit.waist_position)

    assert fit.waist_radius == pytest.approx(FOCUS[0], rel=0.02)
    assert fit.waist_position == pytest.approx(FOCUS[1], abs=0.05e-3)
    assert fit.rayleigh_range == pytest.approx(FOCUS[2], rel=0.05)
    assert 0 < std < 0.05 * FOCUS[2]
