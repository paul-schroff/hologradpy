"""Tests for the coordinate grids, and for the centring convention they define.

The origin of a plane of length ``n`` is the sample at index ``n // 2``. That is what
``torch.fft.fftshift`` implements, so a plane centred this way survives a round trip
through any transform in the package, and it is what ``to_canvas`` preserves and
``ROI.centered`` crops about.

``grids.py`` used to centre at ``ceil(n / 2)`` instead, one sample away. Even sizes hide
the difference entirely, which is why it went unnoticed: every resolution in the suite
and the examples is even. So everything here is parametrised over odd *and* even, and
over a non-square shape with unequal pitches, since a transposed axis would otherwise
pass.
"""

from __future__ import annotations

import pytest
import torch

from hologradpy.grids import (
    get_frequency_grid,
    get_pixel_grid,
    get_spatial_grid,
    metres_to_pixel,
    pixel_to_metres,
    plane_center,
)
from hologradpy.fourier_transforms import FastFourierTransform
from hologradpy.roi import ROI
from hologradpy.utils import to_canvas

# Both parities on both axes, a square odd case, and the degenerate single sample.
RESOLUTIONS = [(8, 10), (7, 9), (7, 10), (8, 9), (9, 9), (1, 1)]
PIXEL_SIZE = (2e-6, 3e-6)  # (height, width), deliberately unequal


@pytest.mark.parametrize("resolution", RESOLUTIONS)
def test_the_grid_is_the_one_fftshift_implements(resolution) -> None:
    """The whole convention in one assertion.

    ``fftshift(fftfreq(n) * n)`` is the centred bin index of an n-point DFT, and the
    pixel grid has to be the same thing, or a plane built on it is not the plane the
    transforms think it is.
    """
    height, width = resolution
    grid_x, grid_y = get_pixel_grid(resolution, dtype=torch.float64)

    for axis, length in ((grid_x[0], width), (grid_y[:, 0], height)):
        wanted = torch.fft.fftshift(torch.fft.fftfreq(length) * length)
        # allclose, not equal: fftfreq(7) * 7 lands on 3.0000002, not 3.
        assert torch.allclose(axis, wanted.to(torch.float64))


@pytest.mark.parametrize("resolution", RESOLUTIONS)
def test_the_origin_is_at_half_the_length_rounded_down(resolution) -> None:
    """Stated the other way round, so a revert fails at odd sizes.

    The convention this replaced put the origin at ``ceil(n / 2)``, which agrees at
    even sizes. Asserting the index directly is what makes an odd size fail loudly.
    """
    height, width = resolution
    grid_x, grid_y = get_pixel_grid(resolution)

    assert int(grid_x[0][width // 2]) == 0
    assert int(grid_y[:, 0][height // 2]) == 0
    # The index the old convention used, where it exists: for width 1 it is off the
    # end of the array, which is its own kind of wrong.
    if width % 2 and width > 1:
        assert int(grid_x[0][-(-width // 2)]) != 0


@pytest.mark.parametrize("resolution", RESOLUTIONS)
def test_plane_center_is_where_the_grid_is_zero(resolution) -> None:
    """``plane_center`` no longer derives from the grid, so this stops them drifting."""
    center_x, center_y = plane_center(resolution)
    grid_x, grid_y = get_pixel_grid(resolution)

    assert int(grid_x[center_y, center_x]) == 0
    assert int(grid_y[center_y, center_x]) == 0


@pytest.mark.parametrize("source", [6, 7, 8, 9])
@pytest.mark.parametrize("target", [4, 5, 10, 11])
def test_the_origin_survives_to_canvas(source, target) -> None:
    """Padding and cropping keep the origin on the origin, at every parity."""
    marker = torch.zeros(source, source)
    center_x, center_y = plane_center((source, source))
    marker[center_y, center_x] = 1.0

    moved = to_canvas(marker, (target, target))

    wanted_x, wanted_y = plane_center((target, target))
    assert moved[wanted_y, wanted_x] == 1.0


@pytest.mark.parametrize("resolution", [7, 8])
@pytest.mark.parametrize("size", [3, 4, 5])
def test_the_origin_survives_a_centered_crop(resolution, size) -> None:
    """``ROI.centered`` about the origin puts the origin at the crop's own origin."""
    marker = torch.zeros(resolution, resolution)
    center_x, center_y = plane_center((resolution, resolution))
    marker[center_y, center_x] = 1.0

    cropped = ROI.centered((center_y, center_x), (size, size)).crop(marker)

    wanted_x, wanted_y = plane_center((size, size))
    assert cropped[wanted_y, wanted_x] == 1.0


@pytest.mark.parametrize("resolution", RESOLUTIONS)
def test_the_origin_in_metres_is_the_center_pixel(resolution) -> None:
    """Zero metres is the origin sample, whatever the parity."""
    assert metres_to_pixel((0.0, 0.0), PIXEL_SIZE, resolution) == plane_center(
        resolution
    )


@pytest.mark.parametrize("resolution", [(8, 10), (7, 9), (7, 10)])
def test_pixels_and_metres_round_trip(resolution) -> None:
    """Deliberately off centre and not symmetric, so an origin error cannot cancel."""
    pixel = (2.0, 5.0)

    metres = pixel_to_metres(pixel, PIXEL_SIZE, resolution)
    back = metres_to_pixel(metres, PIXEL_SIZE, resolution)

    assert back == pytest.approx(pixel)


@pytest.mark.parametrize("resolution", RESOLUTIONS)
def test_pixel_to_metres_agrees_with_the_grid(resolution) -> None:
    """The inverse map against the forward one.

    This is the assertion that would have caught the mismatch: ``pixel_to_metres``
    went through ``plane_center`` while the grid came from ``get_pixel_grid``, and the
    two disagreed at odd sizes.
    """
    grid_x, grid_y = get_spatial_grid(resolution, PIXEL_SIZE)
    height, width = resolution
    row, column = height // 2 + (height > 1), width // 2 + (width > 1)

    metres = pixel_to_metres((float(column), float(row)), PIXEL_SIZE, resolution)

    # One side is float32, the other Python float, so this needs a relative tolerance.
    assert metres[0] == pytest.approx(float(grid_x[row, column]), rel=1e-6)
    assert metres[1] == pytest.approx(float(grid_y[row, column]), rel=1e-6)


@pytest.mark.parametrize("resolution", RESOLUTIONS)
def test_the_spatial_grid_is_zero_at_the_origin(resolution) -> None:
    """And steps by the pitch, per axis, with the two pitches different."""
    grid_x, grid_y = get_spatial_grid(resolution, PIXEL_SIZE)
    center_x, center_y = plane_center(resolution)

    assert float(grid_x[center_y, center_x]) == 0.0
    assert float(grid_y[center_y, center_x]) == 0.0

    height, width = resolution
    if width > 1:
        assert float(grid_x[0, 1] - grid_x[0, 0]) == pytest.approx(
            PIXEL_SIZE[1], rel=1e-6
        )
    if height > 1:
        assert float(grid_y[1, 0] - grid_y[0, 0]) == pytest.approx(
            PIXEL_SIZE[0], rel=1e-6
        )


@pytest.mark.parametrize("resolution", [(8, 10), (7, 9), (7, 10)])
def test_the_frequency_grid_is_the_transform_it_claims_to_be(resolution) -> None:
    """Turns the claim in AngularSpectrumMethod's docstring into a contract."""
    grid_x, grid_y = get_frequency_grid(resolution, PIXEL_SIZE)
    omega = FastFourierTransform(resolution).frequencies

    wanted_x = omega[0].reshape(resolution) / PIXEL_SIZE[1]
    wanted_y = omega[1].reshape(resolution) / PIXEL_SIZE[0]

    assert torch.allclose(grid_x, wanted_x, rtol=1e-6)
    assert torch.allclose(grid_y, wanted_y, rtol=1e-6)
