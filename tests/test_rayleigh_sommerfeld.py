"""Tests for the Rayleigh-Sommerfeld propagator.

This one exists to be the reference the fast propagators are judged against, so it is
itself judged against a closed-form solution rather than against another propagator. A
uniformly lit circular aperture has an exact on-axis field,
``exp(ikz) - (z / sqrt(a^2 + z^2)) exp(ik sqrt(a^2 + z^2))``, and that is what these
compare with.
"""

from __future__ import annotations

import pytest
import torch

from hologradpy.optics.complex_amplitude import ComplexAmplitude, FieldGeometry
from hologradpy.optics.modules.propagators import (
    AngularSpectrumMethod,
    RayleighSommerfeld,
)

WAVELENGTH = 633e-9
APERTURE = 100e-6
DISTANCE = 20e-3


def _aperture(pitch: float, samples: int) -> ComplexAmplitude:
    geometry = FieldGeometry(
        resolution=(samples, samples),
        pixel_size=torch.tensor([pitch, pitch], dtype=torch.float64),
        wavelength=torch.tensor(WAVELENGTH, dtype=torch.float64),
    )
    x, y = geometry.get_spatial_grid()
    lit = (torch.sqrt(x**2 + y**2) <= APERTURE).to(torch.complex128)
    # Double precision throughout: k d runs to 2e5 radians, and this is the reference
    # the others are judged against.
    return ComplexAmplitude.from_geometry(geometry, data=lit, dtype=torch.complex128)


def _on_axis(distance: float = DISTANCE) -> torch.Tensor:
    """The exact field on the axis, which the sum has to reproduce."""
    wavenumber = 2 * torch.pi / WAVELENGTH
    edge = (APERTURE**2 + distance**2) ** 0.5
    return torch.exp(1j * torch.tensor(wavenumber * distance)) - (
        distance / edge
    ) * torch.exp(1j * torch.tensor(wavenumber * edge))


@pytest.mark.parametrize("distance", [5e-3, 20e-3, 50e-3])
def test_it_matches_the_exact_on_axis_field(distance) -> None:
    """Amplitude and phase, against the closed form, at three distances."""
    field = _aperture(2e-6, 240)

    got = torch.as_tensor(
        RayleighSommerfeld(distance, resolution_out=(1, 1))(field)
    ).reshape(())

    # Loose enough to hold at the default float32 geometry, where the phase
    # k d costs about 1e-2 radians. See the class docstring.
    assert abs(got - _on_axis(distance)) / abs(_on_axis(distance)) < 1e-2


def test_it_holds_up_across_the_aperture_sampling() -> None:
    """The one approximation is the pixelated aperture, and it stays small.

    The residual is dominated by which pixels the staircase edge of the circle takes
    in. That does not even fall monotonically with the pitch, so what is pinned is
    that every sampling agrees with the closed form, not that finer is better.
    """
    analytic = _on_axis()
    errors = []
    for pitch, samples in ((8e-6, 30), (4e-6, 60), (1e-6, 240)):
        propagator = RayleighSommerfeld(DISTANCE, resolution_out=(1, 1))
        got = torch.as_tensor(propagator(_aperture(pitch, samples))).reshape(())
        errors.append(float(abs(got - analytic) / abs(analytic)))

    assert max(errors) < 1e-2


def test_it_agrees_with_the_angular_spectrum_where_both_are_valid() -> None:
    """Two independent routes to the same field, which is the point of a benchmark."""
    field = _aperture(2e-6, 128)

    direct = torch.as_tensor(RayleighSommerfeld(DISTANCE)(field))
    spectrum = torch.as_tensor(
        AngularSpectrumMethod(DISTANCE, padded_resolution=(512, 512))(field)
    )

    middle = slice(48, 80)
    a, b = direct[middle, middle], spectrum[middle, middle]
    assert float((a - b).abs().max() / b.abs().max()) < 0.05


def test_the_output_grid_can_differ_from_the_input() -> None:
    """A benchmark has to sample wherever the method under test samples."""
    field = _aperture(2e-6, 64)
    propagator = RayleighSommerfeld(
        DISTANCE, pixel_size_out=(1e-6, 1e-6), resolution_out=(16, 24)
    )

    out = propagator(field)

    assert tuple(out.shape[-2:]) == (16, 24)
    assert float(out.pixel_size.reshape(-1, 2)[0][0]) == pytest.approx(1e-6)


def test_the_block_size_does_not_change_the_answer() -> None:
    """Blocking is a memory device, so it must not be visible in the result.

    Both sides sum explicitly: the convolution route does not block at all, so it
    would pass this without testing anything.
    """
    field = _aperture(4e-6, 48)

    whole = torch.as_tensor(
        RayleighSommerfeld(DISTANCE, convolution=False)(field)
    )
    split = torch.as_tensor(
        RayleighSommerfeld(DISTANCE, convolution=False, block=1000)(field)
    )

    assert (whole - split).abs().max() < 1e-12 * whole.abs().max()


def test_the_convolution_route_is_the_summed_one() -> None:
    """The fast route has to be the same integral, not an approximation of it.

    The kernel depends on the two points only through their separation, so when both
    planes sample one lattice the sum is a convolution exactly. What is left between
    them is the order the terms are added in.
    """
    field = _aperture(2e-6, 64)

    summed = torch.as_tensor(RayleighSommerfeld(DISTANCE, convolution=False)(field))
    convolved = torch.as_tensor(RayleighSommerfeld(DISTANCE)(field))

    assert float((summed - convolved).abs().max() / summed.abs().max()) < 1e-9


def test_the_routes_agree_on_a_smaller_output_window() -> None:
    """A cropped output plane still shares the lattice, so it still convolves."""
    field = _aperture(2e-6, 64)
    options = dict(resolution_out=(20, 12))

    summed = torch.as_tensor(
        RayleighSommerfeld(DISTANCE, convolution=False, **options)(field)
    )
    convolved = torch.as_tensor(RayleighSommerfeld(DISTANCE, **options)(field))

    assert convolved.shape[-2:] == (20, 12)
    assert float((summed - convolved).abs().max() / summed.abs().max()) < 1e-9


def test_a_different_pitch_falls_back_to_summing() -> None:
    """The convolution needs one lattice. A different output pitch is not one."""
    field = _aperture(2e-6, 32)
    same = RayleighSommerfeld(DISTANCE)
    different = RayleighSommerfeld(DISTANCE, pixel_size_out=(1e-6, 1e-6))
    # The output geometry is lazy, so both have to have run before it can be asked
    # what lattice they land on.
    same(field)
    different(field)

    assert same._shares_a_lattice(field)
    assert not different._shares_a_lattice(field)


def test_the_adjoint_of_the_convolution_route_is_its_transpose() -> None:
    """The fast route needs its own check: it correlates rather than convolving."""
    geometry = FieldGeometry(
        resolution=(24, 24),
        pixel_size=torch.tensor([2e-6, 2e-6], dtype=torch.float64),
        wavelength=torch.tensor(WAVELENGTH, dtype=torch.float64),
    )
    generator = torch.Generator().manual_seed(0)
    a = ComplexAmplitude.from_geometry(
        geometry,
        data=torch.randn(24, 24, generator=generator, dtype=torch.complex128),
        dtype=torch.complex128,
    )
    b = ComplexAmplitude.from_geometry(
        geometry,
        data=torch.randn(24, 24, generator=generator, dtype=torch.complex128),
        dtype=torch.complex128,
    )
    propagator = RayleighSommerfeld(1e-3)

    left = (torch.as_tensor(propagator(a)).conj() * torch.as_tensor(b)).sum()
    right = (torch.as_tensor(a).conj() * torch.as_tensor(propagator.adjoint(b))).sum()

    assert abs(left - right) / abs(left) < 1e-9


def test_the_adjoint_is_the_conjugate_transpose() -> None:
    """So it composes into a model that differentiates through it."""
    geometry = FieldGeometry(
        resolution=(24, 24),
        pixel_size=torch.tensor([2e-6, 2e-6], dtype=torch.float64),
        wavelength=torch.tensor(WAVELENGTH, dtype=torch.float64),
    )
    generator = torch.Generator().manual_seed(0)
    a = ComplexAmplitude.from_geometry(
        geometry,
        data=torch.randn(24, 24, generator=generator, dtype=torch.complex128),
    )
    b = ComplexAmplitude.from_geometry(
        geometry,
        data=torch.randn(24, 24, generator=generator, dtype=torch.complex128),
    )
    propagator = RayleighSommerfeld(1e-3)

    left = (torch.as_tensor(propagator(a)).conj() * torch.as_tensor(b)).sum()
    right = (torch.as_tensor(a).conj() * torch.as_tensor(propagator.adjoint(b))).sum()

    assert abs(left - right) / abs(left) < 1e-12
