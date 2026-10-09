"""Tests for the analytic amplitude profiles.

`top_hat_1D` is the line a trap wants: flat along a chosen axis, Gaussian across the
other.
It builds on `top_hat_gaussian_shoulders`, which reached for `xp.sqrt(2)` and so raised
on torch inputs, taking `top_hat_2D` with it. These pin the shape it produces, that it
agrees between numpy and torch, and that the torch path works at all.

`gaussian_beam_intensity_1D` is the Gaussian across a line trap. A round beam equals
the product of two of it, and these pin that too.

`gaussian_beam_intensity` is also elliptical: a major radius along the beam's long axis
and a minor radius across it, with the long axis turned by an angle from +x towards +y.
The spot array, the blur and the speckle grain pass the radii and the angle on.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from hologradpy.profiles.amplitude import (
    gaussian_blur,
    gaussian_beam_intensity,
    gaussian_beam_intensity_1D,
    gaussian_spot_array,
    laser_speckle_intensity,
    top_hat_1D,
    top_hat_2D,
    top_hat_gaussian_shoulders,
)

EXTENT = 3e-4
SAMPLES = 601
PLATEAU = 200e-6
SHOULDER = 8e-6
WAIST = 30e-6

AXIS = np.linspace(-EXTENT, EXTENT, SAMPLES)


def _numpy_grid():
    return np.meshgrid(AXIS, AXIS)


def _torch_grid():
    axis = torch.linspace(-EXTENT, EXTENT, SAMPLES, dtype=torch.float64)
    rows, columns = torch.meshgrid(axis, axis, indexing="ij")
    return columns, rows


@pytest.mark.parametrize("grid", [_numpy_grid, _torch_grid], ids=["numpy", "torch"])
@pytest.mark.parametrize("axis", ["x", "y"])
def test_the_line_is_flat_along_its_axis_and_gaussian_across_it(grid, axis) -> None:
    """The plateau spans the width asked for, and the waist is the 1/e^2 radius."""
    x, y = grid()
    profile = np.asarray(top_hat_1D(x, y, PLATEAU, SHOULDER, WAIST, axis=axis))

    middle = SAMPLES // 2
    down_the_rows = profile[:, middle]
    along_the_columns = profile[middle, :]
    flat, narrow = (
        (down_the_rows, along_the_columns)
        if axis == "y"
        else (along_the_columns, down_the_rows)
    )

    lit = AXIS[flat > 0.5 * flat.max()]
    assert (lit[-1] - lit[0]) == pytest.approx(PLATEAU, abs=4 * SHOULDER)

    # A Gaussian falls to 1/e^2 of its peak one waist out from the centre.
    waist_high = AXIS[narrow > np.exp(-2) * narrow.max()]
    assert (waist_high[-1] - waist_high[0]) == pytest.approx(2 * WAIST, rel=0.05)


def test_the_two_axes_are_transposes_of_one_another() -> None:
    """Turning the line on its side is the only difference the axis makes."""
    x, y = _numpy_grid()
    upright = np.asarray(top_hat_1D(x, y, PLATEAU, SHOULDER, WAIST, axis="y"))
    on_its_side = np.asarray(top_hat_1D(x, y, PLATEAU, SHOULDER, WAIST, axis="x"))

    assert np.abs(upright - on_its_side.T).max() < 1e-12


def test_an_unknown_axis_is_refused() -> None:
    """Silently falling back to a default would give the wrong shape."""
    x, y = _numpy_grid()

    with pytest.raises(ValueError, match="'x' or 'y'"):
        top_hat_1D(x, y, PLATEAU, SHOULDER, WAIST, axis="z")


def test_numpy_and_torch_agree() -> None:
    """One implementation serves both namespaces, so they must not drift apart."""
    from_numpy = np.asarray(top_hat_1D(*_numpy_grid(), PLATEAU, SHOULDER, WAIST))
    from_torch = np.asarray(top_hat_1D(*_torch_grid(), PLATEAU, SHOULDER, WAIST))

    assert np.abs(from_numpy - from_torch).max() < 1e-12


def test_the_intensity_sets_the_peak() -> None:
    """The peak is what was asked for."""
    x, y = _numpy_grid()
    profile = np.asarray(top_hat_1D(x, y, PLATEAU, SHOULDER, WAIST, intensity=2.5))

    assert profile.max() == pytest.approx(2.5, rel=1e-6)


@pytest.mark.parametrize("grid", [_numpy_grid, _torch_grid], ids=["numpy", "torch"])
def test_the_amplitude_scales_the_shoulder_profiles(grid) -> None:
    """``amplitude`` sets the peak, leaving the edge where it was.

    It once divided into the erf argument, which left the peak at one and widened the
    shoulders instead.
    """
    x, y = grid()

    plain = np.asarray(top_hat_gaussian_shoulders(y, 0.0, PLATEAU, SHOULDER, 1.0))
    scaled = np.asarray(top_hat_gaussian_shoulders(y, 0.0, PLATEAU, SHOULDER, 2.5))

    assert scaled.max() == pytest.approx(2.5, rel=1e-6)
    assert np.abs(scaled - 2.5 * plain).max() < 1e-12

    flat = np.asarray(
        top_hat_2D(x, y, 0.0, 0.0, PLATEAU, PLATEAU, SHOULDER, SHOULDER, 2.5)
    )
    assert flat.max() == pytest.approx(2.5, rel=1e-6)


@pytest.mark.parametrize("axis", ["x", "y"])
def test_the_line_sits_where_it_is_shifted_to(axis) -> None:
    """Both offsets move the line, whichever axis it runs along."""
    x, y = _numpy_grid()
    profile = np.asarray(
        top_hat_1D(
            x, y, PLATEAU, SHOULDER, WAIST, axis=axis, shift_x=60e-6, shift_y=-40e-6
        )
    )

    rows, columns = np.nonzero(profile > 0.5 * profile.max())

    assert AXIS[columns].mean() == pytest.approx(60e-6, abs=2e-6)
    assert AXIS[rows].mean() == pytest.approx(-40e-6, abs=2e-6)


@pytest.mark.parametrize("grid", [_numpy_grid, _torch_grid], ids=["numpy", "torch"])
def test_the_shoulder_profiles_accept_both_namespaces(grid) -> None:
    """torch.sqrt takes only tensors, so the square root of two has to be a float."""
    x, y = grid()

    assert float(np.asarray(top_hat_gaussian_shoulders(
        y, 0.0, PLATEAU, SHOULDER, 1.0
    )).max()) == pytest.approx(1.0, rel=1e-6)

    assert float(np.asarray(top_hat_2D(
        x, y, 0.0, 0.0, PLATEAU, PLATEAU, SHOULDER, SHOULDER, 1.0
    )).max()) == pytest.approx(1.0, rel=1e-6)


@pytest.mark.parametrize("grid", [_numpy_grid, _torch_grid], ids=["numpy", "torch"])
def test_the_1D_gaussian_has_the_peak_and_waist_asked_for(grid) -> None:
    """`beam_radius` is the 1/e^2 intensity radius, and `intensity` is the peak."""
    x, _ = grid()
    axis = x[0] if hasattr(x, "shape") and x.ndim == 2 else x
    profile = np.asarray(
        gaussian_beam_intensity_1D(axis, WAIST, shift=40e-6, intensity=3.0)
    )

    assert profile.max() == pytest.approx(3.0, rel=1e-9)
    assert AXIS[profile.argmax()] == pytest.approx(40e-6, abs=2e-6)

    lit = AXIS[profile > np.exp(-2) * profile.max()]
    assert (lit[-1] - lit[0]) == pytest.approx(2 * WAIST, rel=0.05)


def test_the_offset_lifts_the_1D_gaussian() -> None:
    """The offset is added after scaling, so it sets the floor."""
    axis = np.linspace(-EXTENT, EXTENT, SAMPLES)
    profile = gaussian_beam_intensity_1D(axis, WAIST, intensity=2.0, offset=0.25)

    assert profile.min() == pytest.approx(0.25, abs=1e-9)
    assert profile.max() == pytest.approx(2.25, rel=1e-9)


def test_a_round_beam_is_the_product_of_two_1D_gaussians() -> None:
    """A round beam is separable, so it equals the product of two 1D Gaussians."""
    x, y = _numpy_grid()
    round_beam = gaussian_beam_intensity(
        x, y, WAIST, shift_x=40e-6, shift_y=-25e-6, intensity=3.0, offset=0.25
    )
    separated = (
        3.0
        * gaussian_beam_intensity_1D(x, WAIST, 40e-6)
        * gaussian_beam_intensity_1D(y, WAIST, -25e-6)
        + 0.25
    )

    assert np.abs(round_beam - separated).max() < 1e-12


def test_the_line_uses_the_same_gaussian_across_it() -> None:
    """The narrow direction of a line is that same 1D Gaussian, unscaled."""
    x, y = _numpy_grid()
    line = np.asarray(top_hat_1D(x, y, PLATEAU, SHOULDER, WAIST, shift_x=40e-6))

    middle = SAMPLES // 2
    across = line[middle, :] / line[middle, :].max()
    expected = np.asarray(gaussian_beam_intensity_1D(AXIS, WAIST, shift=40e-6))

    assert np.abs(across - expected).max() < 1e-9


def test_a_spot_array_is_centred_where_it_is_shifted_to() -> None:
    """On a grid in metres, the array is centred on its shift, and its outer columns
    sit one spot separation either side.
    """
    x, y = _numpy_grid()
    spots = np.asarray(
        gaussian_spot_array(
            x,
            y,
            number_of_rows=2,
            number_of_columns=3,
            shift_x=30e-6,
            shift_y=-20e-6,
            spot_separation=80e-6,
            beam_radius_major=10e-6,
        )
    )

    rows, columns = np.nonzero(spots > 0.5 * spots.max())

    assert AXIS[columns].mean() == pytest.approx(30e-6, abs=1e-6)
    assert AXIS[rows].mean() == pytest.approx(-20e-6, abs=1e-6)
    assert AXIS[columns].min() == pytest.approx(30e-6 - 80e-6, abs=10e-6)


@pytest.mark.parametrize("beam_radius", [2.0, 3.0, 4.5], ids=["7", "9", "13"])
def test_the_blur_leaves_an_impulse_where_it_found_it(beam_radius) -> None:
    """A symmetric blur must not move the thing it blurs.

    ``kernel_size`` is ``3 * beam_radius // 2 * 2 + 1``, always odd, and the kernel is
    applied with ``padding="same"``, which centres on ``size // 2``. When the kernel
    grid put its peak a sample away from that, every blurred image came out shifted by
    one pixel up and to the left, and the kernel itself was lopsided.
    """
    samples = 33
    middle = samples // 2
    image = torch.zeros(samples, samples)
    image[middle, middle] = 1.0

    blurred = gaussian_blur(image, beam_radius_major=beam_radius)

    peak = np.unravel_index(int(blurred.argmax()), blurred.shape)
    assert peak == (middle, middle)
    # Exactly symmetric, not nearly: any residual is the kernel being lopsided.
    assert float((blurred - torch.flip(blurred, dims=(-2, -1))).abs().max()) == 0.0


def _point(distance, direction_degrees, major, minor, angle_degrees):
    """The intensity of a beam at ``distance`` from its centre, towards
    ``direction_degrees`` from +x towards +y.
    """
    direction = np.radians(direction_degrees)
    x = np.array([distance * np.cos(direction)])
    y = np.array([distance * np.sin(direction)])
    return float(
        gaussian_beam_intensity(
            x, y, major, beam_radius_minor=minor, angle_degrees=angle_degrees
        )[0]
    )


@pytest.mark.parametrize("grid", [_numpy_grid, _torch_grid], ids=["numpy", "torch"])
def test_an_elliptical_beam_has_its_major_radius_along_x_without_an_angle(grid):
    x, y = grid()
    beam = np.asarray(
        gaussian_beam_intensity(x, y, WAIST, beam_radius_minor=WAIST / 2)
    )

    middle = SAMPLES // 2
    along_x = AXIS[beam[middle, :] > np.exp(-2)]
    along_y = AXIS[beam[:, middle] > np.exp(-2)]

    assert (along_x[-1] - along_x[0]) == pytest.approx(2 * WAIST, rel=0.05)
    assert (along_y[-1] - along_y[0]) == pytest.approx(WAIST, rel=0.05)


@pytest.mark.parametrize("angle_degrees", [30.0, 45.0, -60.0])
def test_the_angle_turns_the_long_axis_from_x_towards_y(angle_degrees) -> None:
    """The beam drops to 1/e^2 at its major radius along the angle, and at its minor
    radius a quarter turn further on.
    """
    major, minor = 30e-6, 10e-6

    along = _point(major, angle_degrees, major, minor, angle_degrees)
    across = _point(minor, angle_degrees + 90.0, major, minor, angle_degrees)

    assert along == pytest.approx(np.exp(-2), rel=1e-9)
    assert across == pytest.approx(np.exp(-2), rel=1e-9)
    # Mirrored about x, the major radius lands on the narrow side.
    assert _point(major, -angle_degrees, major, minor, angle_degrees) < 0.1 * along


def test_a_quarter_turn_puts_the_long_axis_along_y() -> None:
    """On a square grid, a beam turned a quarter turn is the upright one transposed."""
    x, y = _numpy_grid()
    turned = gaussian_beam_intensity(
        x, y, WAIST, beam_radius_minor=WAIST / 3, angle_degrees=90.0
    )
    upright = gaussian_beam_intensity(x, y, WAIST, beam_radius_minor=WAIST / 3)

    assert np.abs(turned - upright.T).max() < 1e-12


def test_a_round_beam_is_the_same_at_any_angle() -> None:
    x, y = _numpy_grid()
    round_beam = gaussian_beam_intensity(x, y, WAIST, shift_x=40e-6)

    for minor in (None, WAIST):
        turned = gaussian_beam_intensity(
            x, y, WAIST, shift_x=40e-6, beam_radius_minor=minor, angle_degrees=37.0
        )
        assert np.array_equal(turned, round_beam)


def test_numpy_and_torch_agree_on_an_elliptical_beam() -> None:
    options = dict(
        shift_x=40e-6,
        shift_y=-25e-6,
        intensity=2.0,
        beam_radius_minor=WAIST / 2,
        angle_degrees=25.0,
    )
    on_numpy = gaussian_beam_intensity(*_numpy_grid(), WAIST, **options)
    on_torch = gaussian_beam_intensity(*_torch_grid(), WAIST, **options)

    assert np.abs(on_numpy - on_torch.numpy()).max() < 1e-12


def test_a_minor_radius_above_the_major_radius_is_refused() -> None:
    with pytest.raises(ValueError, match="larger than the major radius"):
        gaussian_beam_intensity(*_numpy_grid(), WAIST, beam_radius_minor=2 * WAIST)


def test_the_minor_radius_follows_the_major_radius() -> None:
    """The second positional number is the minor radius, and the shifts follow it."""
    x, y = _numpy_grid()
    positional = gaussian_beam_intensity(x, y, WAIST, WAIST / 2, 40e-6, -25e-6)
    named = gaussian_beam_intensity(
        x, y, WAIST, beam_radius_minor=WAIST / 2, shift_x=40e-6, shift_y=-25e-6
    )

    assert np.array_equal(positional, named)


def test_the_spots_of_an_array_take_the_radii_and_the_angle() -> None:
    """Turned a quarter turn, each spot's long axis runs along y."""
    x, y = _numpy_grid()
    spots = np.asarray(
        gaussian_spot_array(
            x,
            y,
            number_of_rows=1,
            number_of_columns=1,
            shift_x=0.0,
            shift_y=0.0,
            spot_separation=80e-6,
            beam_radius_major=WAIST,
            beam_radius_minor=WAIST / 2,
            angle_degrees=90.0,
        )
    )

    middle = SAMPLES // 2
    along_x = AXIS[spots[middle, :] > np.exp(-2)]
    along_y = AXIS[spots[:, middle] > np.exp(-2)]

    assert (along_y[-1] - along_y[0]) == pytest.approx(2 * WAIST, rel=0.05)
    assert (along_x[-1] - along_x[0]) == pytest.approx(WAIST, rel=0.05)


@pytest.mark.parametrize("angle_degrees", [0.0, 30.0])
def test_an_elliptical_blur_spreads_an_impulse_along_its_axes(angle_degrees) -> None:
    """The blurred impulse is the kernel: centred, point symmetric, and 1/e^2 at each
    radius along its axis.
    """
    samples = 33
    middle = samples // 2
    image = torch.zeros(samples, samples)
    image[middle, middle] = 1.0

    blurred = gaussian_blur(
        image, 4.0, beam_radius_minor=2.0, angle_degrees=angle_degrees
    ).double()
    blurred /= blurred.max()

    peak = np.unravel_index(int(blurred.argmax()), blurred.shape)
    assert peak == (middle, middle)
    assert float((blurred - torch.flip(blurred, dims=(-2, -1))).abs().max()) == 0.0
    if angle_degrees == 0.0:
        along = float(blurred[middle, middle + 4])
        across = float(blurred[middle + 2, middle])
        assert along == pytest.approx(np.exp(-2), rel=1e-5)
        assert across == pytest.approx(np.exp(-2), rel=1e-5)


def test_elongated_speckle_grains_vary_slowly_along_their_long_axis() -> None:
    """Grains 8 pixels long in x and 2 across change much less from column to column
    than from row to row.
    """
    speckle = laser_speckle_intensity(
        (256, 256),
        1.0,
        8.0,
        grain_radius_minor=2.0,
        generator=torch.Generator().manual_seed(0),
    )

    along_x = float(speckle.diff(dim=1).pow(2).mean())
    along_y = float(speckle.diff(dim=0).pow(2).mean())

    assert along_x < 0.25 * along_y
