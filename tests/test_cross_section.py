"""Tests for sampling a propagated field along the axis it travels down.

Both propagators grew a ``propagate_to(field, geometry)``. The point of having
two is that they fail differently. The Rayleigh-Sommerfeld sum is exact and slow,
so it is what the angular-spectrum route is judged against, exactly as for the
plane-to-plane propagation these generalise.

What has to hold, and does not follow from either implementation on its own:

- sampling a transverse plane reproduces propagating to it, so the two verbs cannot
  drift apart;
- every row of a section is what propagating to that plane would have given, which is
  the whole claim the NUFFT route makes;
- the section is differentiable, since an analysis tool that drops the graph cannot
  be optimized through.
"""

from __future__ import annotations

import warnings

import pytest
import torch

from hologradpy.optics.complex_amplitude import ComplexAmplitude, FieldGeometry
from hologradpy.optics.modules.propagators import (
    AngularSpectrumMethod,
    RayleighSommerfeld,
)
from hologradpy.optics.modules.propagators.angular_spectrum_method import (
    QUANTILE_BUCKETS,
    _phase_step_quantile,
)

pytest.importorskip("pytorch_finufft")

# Several sections below step a millimetre at a time on purpose, to keep the
# comparisons cheap. Whether they resolve the field is what two tests of their
# own are for.
pytestmark = pytest.mark.filterwarnings(
    "ignore:This section does not resolve:RuntimeWarning"
)

WAVELENGTH = 633e-9
APERTURE = 100e-6
PITCH = 2e-6
PADDED = (512, 512)


def _geometry(pitch: float, samples: int) -> FieldGeometry:
    return FieldGeometry(
        resolution=(samples, samples),
        pixel_size=torch.tensor([pitch, pitch], dtype=torch.float64),
        wavelength=torch.tensor(WAVELENGTH, dtype=torch.float64),
    )


def _aperture(pitch: float = PITCH, samples: int = 128) -> ComplexAmplitude:
    """A uniformly lit circular aperture, in double as the reference route needs."""
    geometry = _geometry(pitch, samples)
    x, y = geometry.get_spatial_grid()
    lit = (torch.sqrt(x**2 + y**2) <= APERTURE).to(torch.complex128)
    return ComplexAmplitude.from_geometry(geometry, data=lit, dtype=torch.complex128)


def _on_axis(distance: float) -> torch.Tensor:
    """The exact on-axis field behind a circular aperture."""
    wavenumber = 2 * torch.pi / WAVELENGTH
    edge = (APERTURE**2 + distance**2) ** 0.5
    return torch.exp(1j * torch.tensor(wavenumber * distance)) - (
        distance / edge
    ) * torch.exp(1j * torch.tensor(wavenumber * edge))


def _section(
    field: ComplexAmplitude, distances: torch.Tensor, width: int, pitch: float = PITCH
) -> FieldGeometry:
    return FieldGeometry.cross_section(
        field.wavelength, distances, transverse_pitch=pitch, width=width
    )


def test_sampling_a_transverse_plane_is_the_same_as_propagating_to_it() -> None:
    """The identity that keeps the two verbs honest: a plane is a posed grid too."""
    field = _aperture(samples=48)
    distance = 20e-3
    plane = FieldGeometry(
        wavelength=field.wavelength,
        pixel_size=field.pixel_size,
        resolution=field.geometry.resolution,
        origin=torch.tensor([0.0, 0.0, distance], dtype=torch.float64),
        rotation=torch.eye(3, dtype=torch.float64),
    )
    propagator = RayleighSommerfeld(distance)

    propagated = propagator(field).as_tensor()
    sampled = propagator.propagate_to(field, plane).as_tensor()

    assert float((propagated - sampled).abs().max() / propagated.abs().max()) < 1e-10


def test_the_exact_sum_holds_along_a_whole_line_not_just_at_one_point() -> None:
    """The closed form is a function of z, so a section can be checked against all
    of it at once, where a propagation can only reach one plane.
    """
    field = _aperture(samples=240)
    distances = torch.linspace(10e-3, 50e-3, 5, dtype=torch.float64)

    line = RayleighSommerfeld(0.0).propagate_to(field, _section(field, distances, 1))

    got = line.as_tensor().reshape(-1)
    wanted = torch.stack([_on_axis(float(z)) for z in distances])
    # The staircase edge of the pixelated circle is the one approximation, and it is
    # the same one the plane-to-plane test allows 1e-2 for at its three distances.
    assert float(((got - wanted).abs() / wanted.abs()).max()) < 1.5e-2


def test_every_row_of_a_section_is_what_propagating_to_that_plane_gives() -> None:
    """The claim the NUFFT route makes, and the reason a coarse z pitch costs
    nothing in accuracy: the sum is evaluated at the samples, not interpolated
    onto them.
    """
    field = _aperture()
    distances = torch.linspace(18e-3, 22e-3, 5, dtype=torch.float64)
    section = _section(field, distances, field.resolution[1])

    sampled = AngularSpectrumMethod(0.0, padded_resolution=PADDED).propagate_to(
        field, section
    )

    middle = field.resolution[0] // 2
    for index, distance in enumerate(distances):
        plane = AngularSpectrumMethod(
            float(distance), padded_resolution=PADDED
        )(field).as_tensor()[middle]
        error = (sampled.as_tensor()[index] - plane).abs().max() / plane.abs().max()
        assert float(error) < 1e-5


def test_the_two_routes_agree_where_both_are_valid() -> None:
    """Which is why the exact one came first. The floor here is the reference's own
    quadrature, not the transform: at 2 um sampling the spherical kernel is only
    just resolved.
    """
    field = _aperture()
    distances = torch.linspace(18e-3, 22e-3, 5, dtype=torch.float64)
    section = _section(field, distances, field.resolution[1])

    spectrum = AngularSpectrumMethod(0.0, padded_resolution=PADDED).propagate_to(
        field, section
    )
    direct = RayleighSommerfeld(0.0, convolution=False).propagate_to(field, section)

    middle = slice(48, 80)
    a = spectrum.as_tensor()[:, middle]
    b = direct.as_tensor()[:, middle]
    assert float((a - b).abs().max() / b.abs().max()) < 0.06


def test_a_zoomed_section_lands_where_it_says_it_does() -> None:
    """The output pitch is free, which a plane-by-plane route cannot offer without
    a resampling transform per plane.
    """
    field = _aperture()
    distances = torch.linspace(18e-3, 22e-3, 5, dtype=torch.float64)
    section = _section(field, distances, 64, pitch=PITCH / 4)

    spectrum = AngularSpectrumMethod(0.0, padded_resolution=PADDED).propagate_to(
        field, section
    )
    direct = RayleighSommerfeld(0.0, convolution=False).propagate_to(field, section)

    assert tuple(spectrum.shape) == (5, 64)
    a, b = spectrum.as_tensor(), direct.as_tensor()
    assert float((a - b).abs().max() / b.abs().max()) < 0.06


def _focusing_beam() -> ComplexAmplitude:
    """A beam that fills the grid's band, so its section really does need a fine z."""
    geometry = _geometry(1e-6, 256)
    x, y = geometry.get_spatial_grid()
    lens = torch.exp(-1j * torch.pi / (WAVELENGTH * 200e-6) * (x**2 + y**2))
    return ComplexAmplitude.from_geometry(
        geometry,
        data=torch.exp(-(x**2 + y**2) / (60e-6) ** 2) * lens,
        dtype=torch.complex128,
    )


def _weighted_quantile(
    step: torch.Tensor, energy: torch.Tensor, enclosed_energy: float
) -> torch.Tensor:
    """The same quantity by sorting, which the buckets approximate."""
    order = step.argsort(dim=-1, descending=True)
    tail = energy.gather(-1, order).cumsum(-1)
    budget = (1.0 - enclosed_energy) * energy.sum(-1, keepdim=True)
    first = (tail > budget).to(torch.uint8).argmax(dim=-1)
    return step.gather(-1, order).gather(-1, first.unsqueeze(-1)).squeeze(-1)


@pytest.mark.parametrize("enclosed_energy", (0.999, 0.99, 0.5))
def test_the_bucketed_quantile_bounds_the_exact_one(enclosed_energy: float) -> None:
    """The ratio is compared against 1, so the quantile behind it has to err upwards.

    Reading below the exact quantile would call a section resolved that is not, and
    the bucket edges are the place that can go wrong: the index has to convert back
    to the edge of its own bucket.
    """
    generator = torch.Generator().manual_seed(0)
    step = torch.rand(3, 200_000, generator=generator, dtype=torch.float64)
    energy = torch.rand(3, 200_000, generator=generator, dtype=torch.float64)

    bucketed = _phase_step_quantile(step, energy, enclosed_energy)
    exact = _weighted_quantile(step, energy, enclosed_energy)
    width = step.amax(dim=-1) / QUANTILE_BUCKETS

    assert torch.all(bucketed >= exact)
    assert torch.all(bucketed - exact <= width)


def test_a_section_too_coarse_to_show_the_focus_says_so() -> None:
    """A picture that steps over the depth of focus is a moire of one, and looks
    every bit as convincing.
    """
    beam = _focusing_beam()
    propagator = AngularSpectrumMethod(0.0, padded_resolution=PADDED)
    around_focus = torch.linspace(150e-6, 250e-6, 21, dtype=torch.float64)
    coarse = _section(beam, around_focus, 256, pitch=1e-6)

    assert propagator.nyquist_ratio(beam, coarse)[0] > 1.0
    with pytest.warns(RuntimeWarning, match="does not resolve"):
        propagator.propagate_to(beam, coarse)


def test_a_section_fine_enough_to_show_the_focus_keeps_quiet() -> None:
    beam = _focusing_beam()
    propagator = AngularSpectrumMethod(0.0, padded_resolution=PADDED)
    around_focus = torch.linspace(150e-6, 250e-6, 401, dtype=torch.float64)
    fine = _section(beam, around_focus, 256, pitch=1e-6)

    assert propagator.nyquist_ratio(beam, fine)[0] < 1.0
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        propagator.propagate_to(beam, fine)


def test_a_gradient_reaches_the_field_the_section_was_taken_of() -> None:
    """An analysis tool that drops the graph cannot be optimized through, and a
    depth of focus is a perfectly good thing to optimize.
    """
    geometry = _geometry(1e-6, 64)
    x, y = geometry.get_spatial_grid()
    values = torch.exp(-(x**2 + y**2) / (20e-6) ** 2).to(torch.complex128)
    values.requires_grad_(True)
    field = ComplexAmplitude.from_tensor(
        values, geometry.wavelength, geometry.pixel_size
    )
    section = _section(
        field, torch.linspace(40e-6, 80e-6, 9, dtype=torch.float64), 64, pitch=1e-6
    )

    sampled = AngularSpectrumMethod(0.0, padded_resolution=(128, 128)).propagate_to(
        field, section
    )
    (gradient,) = torch.autograd.grad(sampled.intensity.sum(), values)

    assert torch.isfinite(gradient).all()
    assert float(gradient.abs().max()) > 0.0


def test_a_gradient_reaches_where_the_section_was_taken() -> None:
    """Moving the section along z is differentiable too, so a focus can be found
    by optimisation.
    """
    geometry = _geometry(1e-6, 64)
    x, y = geometry.get_spatial_grid()
    field = ComplexAmplitude.from_geometry(
        geometry,
        data=torch.exp(-(x**2 + y**2) / (20e-6) ** 2).to(torch.complex128),
        dtype=torch.complex128,
    )
    propagator = AngularSpectrumMethod(0.0, padded_resolution=(128, 128))
    template = _section(
        field, torch.linspace(40e-6, 80e-6, 9, dtype=torch.float64), 64, pitch=1e-6
    )

    def brightness(shift: torch.Tensor) -> torch.Tensor:
        moved = FieldGeometry(
            wavelength=template.wavelength,
            pixel_size=template.pixel_size,
            resolution=template.resolution,
            origin=template.origin + shift * torch.tensor([0.0, 0.0, 1.0]),
            rotation=template.rotation,
        )
        return propagator.propagate_to(field, moved).intensity.sum()

    shift = torch.zeros((), dtype=torch.float64, requires_grad=True)
    (gradient,) = torch.autograd.grad(brightness(shift), shift)

    # Small enough that the second-order term of the difference is below the
    # agreement asserted, which at 1e-7 it is not.
    step = 1e-8
    with torch.no_grad():
        difference = (
            brightness(torch.tensor(step, dtype=torch.float64))
            - brightness(torch.tensor(-step, dtype=torch.float64))
        ) / (2 * step)

    assert float(gradient) == pytest.approx(float(difference), rel=1e-4)
