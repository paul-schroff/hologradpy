"""Optical power tracking: ``ComplexAmplitude.power`` / ``with_power`` and the
power-conserving (``power_normalized``) FFT propagator.

Power is the integral of intensity over area, ``sum(|E|^2) * pixel_area``, so it
is conserved by a lossless Fourier lens (Parseval) only once the physical
``(du*dv)/(lambda*f)`` prefactor is applied -- which is what ``power_normalized``
does. The reduction is in float64 for precision.
"""

from __future__ import annotations

import pytest
import torch

from hologradpy.optics.complex_amplitude import ComplexAmplitude
from hologradpy.optics.modules.propagators import FourierLensFFT


pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")

WAVELENGTH = torch.tensor([0.63e-6])
PIXEL_SIZE = torch.tensor([[12.5e-6, 12.5e-6]])
PIXEL_AREA = 12.5e-6 * 12.5e-6


def test_power_value_matches_integral() -> None:
    # Uniform amplitude 2 over a 4x5 field: power = sum(|E|^2) * pixel_area.
    field = ComplexAmplitude(2.0 * torch.ones(4, 5, dtype=torch.complex64),
                             WAVELENGTH, PIXEL_SIZE)
    expected = 4 * 4 * 5 * PIXEL_AREA
    assert float(field.power()) == pytest.approx(expected, rel=1e-5)


def test_power_shape_per_batch_and_wavelength() -> None:
    wavelengths = torch.tensor([0.6e-6, 0.7e-6, 0.8e-6])
    pixel_size = torch.tensor([[10e-6, 10e-6]] * 3)
    field = ComplexAmplitude(
        torch.ones(2, 3, 4, 5, dtype=torch.complex64), wavelengths, pixel_size
    )
    # One value per (*batch, wavelength).
    assert tuple(field.power().shape) == (2, 3)


def test_with_power_sets_power_and_preserves_phase() -> None:
    data = (torch.randn(8, 8) + 1j * torch.randn(8, 8)).to(torch.complex64)
    field = ComplexAmplitude(data, WAVELENGTH, PIXEL_SIZE)
    scaled = field.with_power(1e-3)
    assert abs(float(scaled.power()) - 1e-3) < 1e-9
    torch.testing.assert_close(scaled.phase, field.phase)


def test_with_power_preserves_autograd_graph() -> None:
    data = torch.ones(4, 5, dtype=torch.complex64, requires_grad=True)
    field = ComplexAmplitude(data, WAVELENGTH, PIXEL_SIZE)
    field.with_power(1e-3).intensity.sum().backward()
    assert data.grad is not None


def test_power_argument_in_constructor() -> None:
    data = (torch.randn(8, 8) + 1j * torch.randn(8, 8)).to(torch.complex64)
    field = ComplexAmplitude(data, WAVELENGTH, PIXEL_SIZE, power=2e-3)
    assert abs(float(field.power()) - 2e-3) < 1e-9


def test_power_argument_in_from_geometry() -> None:
    from hologradpy.optics.complex_amplitude import FieldGeometry

    geometry = FieldGeometry(WAVELENGTH, PIXEL_SIZE, (4, 5))
    field = ComplexAmplitude.from_geometry(geometry, power=5e-3)
    assert abs(float(field.power()) - 5e-3) < 1e-9


@pytest.mark.parametrize("padded_resolution", [(256, 256), (512, 512)])
def test_fft_conserves_power_when_normalized(padded_resolution) -> None:
    torch.manual_seed(0)
    data = (torch.randn(128, 128) + 1j * torch.randn(128, 128)).to(torch.complex64)
    field = ComplexAmplitude(data, WAVELENGTH, PIXEL_SIZE, power=1e-3)

    lens = FourierLensFFT(
        focal_length=0.25,
        padded_resolution=padded_resolution,
        power_normalized=True,
    )
    output = lens(field)
    # Full Fourier-plane power equals the input power (Parseval, physical units).
    assert abs(float(output.power()) / float(field.power()) - 1.0) < 1e-4


def test_fft_unnormalized_does_not_conserve_power() -> None:
    data = (torch.randn(128, 128) + 1j * torch.randn(128, 128)).to(torch.complex64)
    field = ComplexAmplitude(data, WAVELENGTH, PIXEL_SIZE, power=1e-3)

    # power_normalized=True is the default; the raw norm="backward" DFT (opt-out)
    # keeps the legacy arbitrary-unit scale -- a large factor-N offset.
    lens = FourierLensFFT(
        focal_length=0.25, padded_resolution=(512, 512), power_normalized=False
    )
    output = lens(field)
    assert float(output.power()) / float(field.power()) > 100.0
