"""Gauss's law on a field vector, and what propagation does to it.

The component axis is only worth having if the components stay physical while they
travel. A field vector in free space satisfies ``k . E = 0`` for every plane wave in its
spectrum, and every module in the package treats the components independently, so the
question these answer is whether independent treatment preserves that.
"""

from __future__ import annotations

import pytest
import torch

from hologradpy.optics.complex_amplitude import ComplexAmplitude, FieldGeometry
from hologradpy.optics.modules.propagators import AngularSpectrumMethod
from hologradpy.optics.polarization import gauss_law_residual, longitudinal_component

pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")

RESOLUTION = (64, 64)
PITCH = 1e-6
WAVELENGTH = 800e-9
DISTANCE = 20e-6


def _geometry() -> FieldGeometry:
    return FieldGeometry(
        wavelength=torch.tensor(WAVELENGTH, dtype=torch.float64),
        pixel_size=torch.tensor([PITCH, PITCH], dtype=torch.float64),
        resolution=RESOLUTION,
    )


def _converging_beam(tilt: float = 0.0) -> ComplexAmplitude:
    """A Gaussian carrying a quadratic phase, so its spectrum spans real angles."""
    geometry = _geometry()
    grid_x, grid_y = geometry.get_spatial_grid()
    waist = 8 * PITCH
    envelope = torch.exp(-(grid_x**2 + grid_y**2) / waist**2)
    wavenumber = 2 * torch.pi / WAVELENGTH
    curvature = torch.exp(-1j * wavenumber * (grid_x**2 + grid_y**2) / (2 * 60e-6))
    ramp = torch.exp(1j * wavenumber * tilt * grid_x)
    return ComplexAmplitude.from_geometry(
        geometry, data=(envelope * curvature * ramp).to(torch.complex128)
    )


def _physical_vector(tilt: float = 0.0) -> ComplexAmplitude:
    """A field vector Gauss's law allows, polarized along x."""
    transverse_x = _converging_beam(tilt)
    transverse_y = transverse_x * 0.0
    return ComplexAmplitude.from_components(
        transverse_x,
        transverse_y,
        longitudinal_component(transverse_x, transverse_y),
    )


def test_the_longitudinal_component_makes_the_field_source_free() -> None:
    vector = _physical_vector()

    residual = gauss_law_residual(vector)

    assert float(residual.abs().max()) < 1e-9


def test_dropping_the_longitudinal_component_breaks_the_law() -> None:
    """The check has to be able to fail, or it measures nothing."""
    transverse = _converging_beam()
    flat = ComplexAmplitude.from_components(
        transverse, transverse * 0.0, transverse * 0.0
    )

    residual = gauss_law_residual(flat)

    assert float(residual.abs().max()) > 1e-3


def test_the_longitudinal_component_grows_with_the_angles_in_the_spectrum() -> None:
    """It is the term a high numerical aperture makes matter, so tilting the beam has
    to raise it.
    """
    straight = _physical_vector(tilt=0.0)
    tilted = _physical_vector(tilt=0.3)

    assert float(tilted.component(2).amplitude.max()) > 3 * float(
        straight.component(2).amplitude.max()
    )


def test_propagating_the_components_independently_keeps_the_field_physical() -> None:
    """The claim the component axis rests on.

    Every module treats the components on their own, and the angular spectrum multiplies
    each plane wave by a phase that depends on the wavevector alone. The longitudinal
    component the propagated transverse field demands is therefore the propagated
    longitudinal component.

    The propagation runs on the sampled grid, so that what is measured is the commuting
    alone. Padding the canvas and cropping it back truncates the field at the window
    edge, which moves the spectrum by an amount that has nothing to do with components.
    """
    vector = _physical_vector()
    propagator = AngularSpectrumMethod(DISTANCE, padded_resolution=RESOLUTION)

    propagated = propagator(vector)

    assert propagated.is_vector
    assert float(gauss_law_residual(propagated).abs().max()) < 1e-9
    torch.testing.assert_close(
        propagated.component(2).as_tensor(),
        longitudinal_component(
            propagated.component(0), propagated.component(1)
        ).as_tensor(),
        rtol=1e-6,
        atol=1e-12,
    )


def test_a_transverse_component_carrying_a_vector_is_refused() -> None:
    vector = _physical_vector()

    with pytest.raises(ValueError, match="each transverse component"):
        longitudinal_component(vector, vector)


def test_the_residual_needs_all_three_components() -> None:
    with pytest.raises(ValueError, match="all three components"):
        gauss_law_residual(_converging_beam())
