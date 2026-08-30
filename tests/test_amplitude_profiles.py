"""Tests for the analytic amplitude profiles.

`top_hat_1D` is the line a trap wants: flat along a chosen axis, Gaussian across the
other.
It builds on `top_hat_gaussian_shoulders`, which reached for `xp.sqrt(2)` and so raised
on torch inputs, taking `top_hat_2D` with it. These pin the shape it produces, that it
agrees between numpy and torch, and that the torch path works at all.

`gaussian_beam_intensity_1D` is the one Gaussian both a round beam and a line trap are
built from, so these also pin that the round beam stays the product of two of it.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from hologradpy.profiles.amplitude import (
    gaussian_blur,
    gaussian_beam_intensity,
    gaussian_beam_intensity_1D,
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
    """The 2D profile is separable, which is what lets it share the 1D one."""
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

    blurred = gaussian_blur(image, beam_radius=beam_radius)

    peak = np.unravel_index(int(blurred.argmax()), blurred.shape)
    assert peak == (middle, middle)
    # Exactly symmetric, not nearly: any residual is the kernel being lopsided.
    assert float((blurred - torch.flip(blurred, dims=(-2, -1))).abs().max()) == 0.0
