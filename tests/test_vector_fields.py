"""Integrating a 2D vector field back into the scalar potential it came from.

The failure mode is quiet: a wrong sign, a transposed pixel size or a mishandled origin
all return something that still looks like a wavefront and is wrong by a low-order term.
These tests pin the round trip on analytic fields where the answer is known, on a
deliberately non-square grid with unequal pixel spacing so a transposed axis cannot
pass.

The integrator walks a path rather than solving a least-squares problem, following
``scalarPotentialN`` in SLMTools. That makes it exact for a conservative field and
path-dependent for anything else, which is a property worth knowing rather than a bug:
the maps it is given are gradients up to discretization error.
"""

from __future__ import annotations

import pytest
import torch

from hologradpy.grids import get_spatial_grid
from hologradpy.vector_fields import (
    forward_difference,
    gradient,
    mean_curvature,
    integrate_along_path,
)

RESOLUTION = (48, 64)
PIXEL_SIZE = (2e-4, 1e-4)  # (y, x), deliberately unequal


def _grid() -> tuple[torch.Tensor, torch.Tensor]:
    return get_spatial_grid(RESOLUTION, PIXEL_SIZE)


def _grid64() -> tuple[torch.Tensor, torch.Tensor]:
    """An exactly uniform grid in float64, on the same origin convention.

    :func:`~hologradpy.grids.get_spatial_grid` returns float32, whose steps come out a
    part in ten million off the pitch they represent. Casting that is no help, the
    jitter already being in the coordinates, and differencing twice turns it into a
    curvature of order a hundredth. The operator assumes a uniform lattice, so testing
    it means handing it one.
    """
    spacing_y, spacing_x = PIXEL_SIZE
    rows = (
        torch.arange(RESOLUTION[0], dtype=torch.float64) - RESOLUTION[0] // 2
    ) * spacing_y
    columns = (
        torch.arange(RESOLUTION[1], dtype=torch.float64) - RESOLUTION[1] // 2
    ) * spacing_x
    return columns.expand(RESOLUTION), rows[:, None].expand(RESOLUTION)


def _centred(field: torch.Tensor) -> torch.Tensor:
    return field - field.mean()


def test_a_gradient_integrates_back_to_its_potential():
    """The core contract, on a field with linear, quadratic and cross terms."""
    x, y = _grid()
    expected = _centred(0.3 * x + 0.7 * y + 2.0 * (x**2 + y**2) + 0.5 * x * y)

    recovered = _centred(
        integrate_along_path(*gradient(expected, PIXEL_SIZE), PIXEL_SIZE)
    )

    relative = (recovered - expected).abs().max() / expected.abs().max()
    assert float(relative) < 0.05


def test_the_potential_of_a_radial_field_is_a_paraboloid():
    """The shaping-lens case, and the one that would catch a flipped sign."""
    x, y = _grid()
    recovered = _centred(integrate_along_path(x, y, PIXEL_SIZE))
    expected = _centred((x**2 + y**2) / 2)

    overlap = (recovered * expected).sum() / (expected * expected).sum()
    assert float(overlap) == pytest.approx(1.0, abs=0.05)


def test_a_constant_field_integrates_to_a_tilt():
    """For a ray map the constant part is the tilt that puts light off axis."""
    x, y = _grid()
    constant_x, constant_y = 3.0, -2.0

    potential = integrate_along_path(
        torch.full_like(x, constant_x), torch.full_like(y, constant_y), PIXEL_SIZE
    )

    expected = constant_x * x + constant_y * y
    torch.testing.assert_close(
        _centred(potential), _centred(expected), rtol=1e-4, atol=1e-9
    )


def test_the_pixel_size_is_applied_to_the_matching_axis():
    """Swapping (y, x) would rescale each axis by the other's pitch."""
    x, y = _grid()
    field = _centred(x**2 + y**2)

    correct = integrate_along_path(*gradient(field, PIXEL_SIZE), PIXEL_SIZE)
    swapped = integrate_along_path(*gradient(field, PIXEL_SIZE), PIXEL_SIZE[::-1])

    assert not torch.allclose(correct, swapped, rtol=0.1)


def test_the_potential_is_zero_at_the_origin():
    """The gauge is fixed where hologradpy.grids puts the zero of its grids."""
    x, y = _grid()
    potential = integrate_along_path(x, y, PIXEL_SIZE)

    origin = (RESOLUTION[0] // 2, RESOLUTION[1] // 2)
    assert float(potential[origin].abs()) < 1e-12


def test_shape_and_dtype_survive():
    x, y = _grid()
    potential = integrate_along_path(x, y, PIXEL_SIZE)

    assert potential.shape == x.shape
    assert potential.dtype == x.dtype
    assert not potential.is_complex()


# --- differencing and curvature -----------------------------------------------------


def test_forward_difference_returns_x_before_y():
    """The naming this module lives or dies by, and the one that was back to front.

    A field that varies only along x has no y difference at all, so a swapped pair
    shows up as a zero where the signal should be.
    """
    x, _ = _grid()
    ramp = x.expand(RESOLUTION)

    difference_x, difference_y = forward_difference(ramp)

    assert float(difference_x.abs().max()) > 0
    assert float(difference_y.abs().max()) == 0


def test_forward_difference_shortens_the_axis_it_differences():
    x, _ = _grid()
    difference_x, difference_y = forward_difference(x**2)

    assert difference_x.shape == (RESOLUTION[0], RESOLUTION[1] - 1)
    assert difference_y.shape == (RESOLUTION[0] - 1, RESOLUTION[1])


def test_a_plane_has_no_mean_curvature():
    """A plane is flat, in float64.

    A second difference at this pitch amplifies float32's rounding by enough to give a
    plane a curvature of order a tenth, which the operator documents and this test
    would otherwise trip over.
    """
    x, y = _grid64()
    curvature = mean_curvature(0.3 * x - 0.7 * y, PIXEL_SIZE)

    assert float(curvature.abs().max()) == pytest.approx(0.0, abs=1e-6)


def test_the_mean_curvature_of_a_paraboloid_is_one_over_its_radius():
    """At the apex the slopes vanish, so the curvature is exactly ``1 / radius``.

    Away from it the slope terms bite, which is why only the centre sample is asserted.
    A pixel size applied to the wrong axis, or left off the second derivative, misses
    this by the ratio of the two pitches.
    """
    x, y = _grid64()
    radius = 1e-2
    curvature = mean_curvature((x**2 + y**2) / (2 * radius), PIXEL_SIZE)

    apex = (RESOLUTION[0] // 2, RESOLUTION[1] // 2)
    assert float(curvature[apex]) == pytest.approx(1 / radius, rel=1e-3)


def test_the_mean_curvature_pixel_size_is_applied_to_the_matching_axis():
    x, y = _grid64()
    radius = 1e-2
    surface = (x**2 + y**2) / (2 * radius)
    apex = (RESOLUTION[0] // 2, RESOLUTION[1] // 2)

    swapped = mean_curvature(surface, PIXEL_SIZE[::-1])

    assert float(swapped[apex]) != pytest.approx(1 / radius, rel=0.1)
