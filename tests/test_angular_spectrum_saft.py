"""Tests for angular spectrum propagation onto a freely chosen output pitch.

The forward leg is a plain FFT and the inverse leg is the semi-analytical transform,
whose sample spacing is set by the quadratic phase it carries. Since that phase is the
transfer function's own, extracting it both fixes the output pitch and takes the
dominant term out of what still has to be sampled.

The strongest statement available is against :class:`AngularSpectrumMethod`: this is
the same method on a different output grid, so at unit zoom the two must agree to
floating point. Against :class:`RayleighSommerfeld` the agreement is far looser, and
deliberately so -- the angular spectrum samples the transfer function in frequency
while the direct integral samples the kernel in space, and those are genuinely
different discretisations. Measured here: plain ASM and this propagator differ from the
integral by the *same* 2.1e-1, and from each other by 1e-15.
"""

from __future__ import annotations

import warnings

import pytest
import torch

from hologradpy.optics.complex_amplitude import ComplexAmplitude, FieldGeometry
from hologradpy.optics.modules.propagators import (
    AngularSpectrumMethod,
    AngularSpectrumSAFT,
    RayleighSommerfeld,
)

RESOLUTION = (64, 64)
PITCH = 2e-6
WAVELENGTH = 633e-9
DISTANCE = 2e-4


def _aperture(radius: float = 20e-6) -> ComplexAmplitude:
    geometry = FieldGeometry(
        resolution=RESOLUTION,
        pixel_size=torch.tensor([PITCH, PITCH], dtype=torch.float64),
        wavelength=torch.tensor(WAVELENGTH, dtype=torch.float64),
    )
    x, y = geometry.get_spatial_grid()
    lit = (torch.sqrt(x**2 + y**2) <= radius).to(torch.complex128)
    return ComplexAmplitude.from_geometry(geometry, data=lit, dtype=torch.complex128)


def _plain(field) -> torch.Tensor:
    """The values without the wrapper, which checks geometry on every operation."""
    return field.as_tensor() if hasattr(field, "as_tensor") else field


def _rms(a, b) -> float:
    a, b = _plain(a), _plain(b)
    return float(((a - b).abs() ** 2).mean().sqrt() / b.abs().max())


def test_at_unit_zoom_it_is_the_angular_spectrum_method() -> None:
    """The load-bearing test: same method, same grid, so the same numbers.

    This pins the transfer function, the normalisation and the sample alignment all
    at once. The transform returns an unnormalised sum, so a missing division by the
    transform size would show here as a factor of tens of thousands.

    The floor is the reference's, not this propagator's: AngularSpectrumMethod builds
    its transfer function on FastFourierTransform.frequencies, which comes from
    ``torch.arange`` at the default dtype, so on a float32 default the two agree to
    4e-8 rather than the 4e-15 they reach when the grids are float64 throughout.
    """
    field = _aperture()

    zoomed = AngularSpectrumSAFT(
        DISTANCE, pixel_size_out=(PITCH, PITCH), resolution_out=RESOLUTION
    )(field)
    plain = AngularSpectrumMethod(
        DISTANCE, padded_resolution=(128, 128)
    )(field)

    assert _rms(zoomed, plain) < 1e-6


def test_the_output_pitch_defaults_to_the_input_pitch() -> None:
    """Always representable, where a coarser one would tile the plane."""
    field = _aperture()
    propagator = AngularSpectrumSAFT(DISTANCE)

    propagator(field)

    pitch = propagator.pixel_size_out.reshape(-1, 2)[0]
    assert float(pitch[0]) == pytest.approx(PITCH)
    assert float(pitch[1]) == pytest.approx(PITCH)


@pytest.mark.parametrize("zoom", [2.0, 4.0])
def test_the_zoom_is_real(zoom) -> None:
    """A finer pitch spreads the same feature over proportionally more samples.

    Measured on the central lobe rather than a second moment: the window holds a
    fixed number of samples, so a finer pitch sees less of the plane and anything
    reaching the edge would be clipped rather than magnified.
    """
    field = _aperture()

    def lobe_width(pitch_out):
        out = AngularSpectrumSAFT(
            DISTANCE,
            pixel_size_out=(pitch_out, pitch_out),
            resolution_out=RESOLUTION,
        )(field)
        row = (_plain(out).abs() ** 2)[RESOLUTION[0] // 2]
        return float((row > 0.5 * row.max()).sum())

    assert lobe_width(PITCH / zoom) == pytest.approx(
        zoom * lobe_width(PITCH), rel=0.25
    )


def test_it_tracks_the_direct_integral_where_they_are_comparable() -> None:
    """Against the ground truth, at the tolerance the two methods actually allow.

    A tight bound here would be wrong: the angular spectrum and the Rayleigh-Sommerfeld
    sum discretise differently, and the gap is the method's, not this propagator's.
    """
    field = _aperture()
    pitch_out = PITCH / 4

    zoomed = AngularSpectrumSAFT(
        DISTANCE,
        pixel_size_out=(pitch_out, pitch_out),
        resolution_out=RESOLUTION,
    )(field)
    reference = RayleighSommerfeld(
        DISTANCE,
        pixel_size_out=(pitch_out, pitch_out),
        resolution_out=RESOLUTION,
    )(field)

    assert _rms(zoomed, reference) < 0.05

    # The same total power, which is the part that is not a matter of discretisation.
    power = float((_plain(zoomed).abs() ** 2).sum())
    wanted = float((_plain(reference).abs() ** 2).sum())
    assert power == pytest.approx(wanted, rel=0.02)


def test_the_margin_is_comfortable_when_zooming_in() -> None:
    """Finer output pitches are the direction this is for, and cost margin."""
    field = _aperture()
    margins = []
    for zoom in (1.0, 2.0, 4.0):
        propagator = AngularSpectrumSAFT(
            DISTANCE,
            pixel_size_out=(PITCH / zoom, PITCH / zoom),
            resolution_out=RESOLUTION,
        )
        propagator(field)
        margins.append(max(propagator.sampling_margin()))

    assert margins == sorted(margins, reverse=True)
    assert margins[-1] < 1.0


def test_a_coarser_pitch_is_flagged_rather_than_silently_tiled() -> None:
    """The failure that is invisible in the numbers and obvious in a picture.

    A pitch above the input's pushes the sample lattice past one period, and the plane
    comes back tiled with copies of itself. Nothing about the amplitudes says so.
    """
    field = _aperture()
    propagator = AngularSpectrumSAFT(
        DISTANCE, pixel_size_out=(4 * PITCH, 4 * PITCH), resolution_out=RESOLUTION
    )

    with pytest.warns(RuntimeWarning, match="tiled with copies"):
        propagator(field)

    assert max(propagator.sampling_margin()) > 1.0


def test_a_workable_grid_says_nothing() -> None:
    """The warning has to stay quiet when it should, or it will be ignored."""
    field = _aperture()
    propagator = AngularSpectrumSAFT(
        DISTANCE, pixel_size_out=(PITCH / 4, PITCH / 4), resolution_out=RESOLUTION
    )

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        propagator(field)

