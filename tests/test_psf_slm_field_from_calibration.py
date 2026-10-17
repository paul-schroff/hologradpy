"""PSFSLMField seeded from a measured SLM-plane field, and the transmission held while
the kernel is frozen.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from hologradpy.optics.complex_amplitude import ComplexAmplitude, FieldGeometry
from hologradpy.optics.modules.slm_fields import PSFSLMField

SLM_RESOLUTION = (64, 64)
SLM_PIXEL_SIZE = 12.5e-6
WAVELENGTH = 0.670e-6
FOCAL_LENGTH = 0.1
# The kernel's pitch is lambda f / (2 L), 41.9 um, so this extent takes 21 samples.
PSF_EXTENT = 860e-6
KERNEL_SIZE = 21


def _geometry() -> FieldGeometry:
    return FieldGeometry(
        resolution=SLM_RESOLUTION,
        pixel_size=torch.tensor([SLM_PIXEL_SIZE, SLM_PIXEL_SIZE]),
        wavelength=torch.tensor(WAVELENGTH),
    )


def _calibration(wavefront: torch.Tensor) -> SimpleNamespace:
    """A stand-in for a wavefront calibration record, holding only its field."""
    geometry = _geometry()
    return SimpleNamespace(
        complex_amplitude=ComplexAmplitude(
            wavefront, wavelength=geometry.wavelength, pixel_size=geometry.pixel_size
        )
    )


def _kernel_field() -> torch.Tensor:
    """The SLM-plane field of a random complex kernel, in the span of the kernel."""
    source = PSFSLMField(FOCAL_LENGTH, PSF_EXTENT)
    source.initialize_from_geometry(_geometry())
    assert source.psf_kernel_size == (KERNEL_SIZE, KERNEL_SIZE)
    generator = torch.Generator().manual_seed(0)
    kernel = torch.complex(
        torch.randn(KERNEL_SIZE, KERNEL_SIZE, generator=generator),
        torch.randn(KERNEL_SIZE, KERNEL_SIZE, generator=generator),
    )
    with torch.no_grad():
        source.psf_kernel.copy_(kernel)
    return source.get_wavefront().detach()


def _fitted(wavefront: torch.Tensor) -> PSFSLMField:
    field = PSFSLMField.from_calibration_data(
        _calibration(wavefront), FOCAL_LENGTH, PSF_EXTENT
    )
    field.initialize_from_geometry(_geometry())
    return field


def test_reproduces_a_field_the_kernel_can_make() -> None:
    """The field comes back up to its scale, since the kernel is set to unit peak."""
    wavefront = _kernel_field()

    field = _fitted(wavefront)
    fitted = field.get_wavefront().detach()
    scale = (fitted.conj() * wavefront).sum() / (fitted.abs() ** 2).sum()

    assert field.get_psf_kernel().is_complex()
    assert field.wavefront_fit_residual < 1e-3
    error = (scale * fitted - wavefront).abs().max()
    assert float(error / wavefront.abs().max()) < 1e-3


def test_the_kernel_has_unit_peak_whatever_the_power() -> None:
    """The learning rate of a refit then means the same step for every field."""
    wavefront = _kernel_field()

    for power in (1e-6, 1.0, 1e3):
        field = _fitted(wavefront * power)
        peak = field.get_psf_kernel().detach().abs().max()
        assert float(peak) == pytest.approx(1.0)


def test_reports_what_a_rough_field_loses() -> None:
    """A field with an independent phase per pixel has no compact focal spot, so most
    of it lies outside the kernel's span.
    """
    generator = torch.Generator().manual_seed(1)
    phase = 2 * torch.pi * torch.rand(SLM_RESOLUTION, generator=generator)

    field = _fitted(torch.exp(1j * phase).to(torch.complex64))

    assert field.wavefront_fit_residual > 0.5


def test_holds_the_transmission_while_the_kernel_is_frozen() -> None:
    field = _fitted(_kernel_field())
    field.psf_kernel.requires_grad_(False)

    assert field.get_transmission() is field.get_transmission()


def test_transforms_again_after_the_kernel_changes() -> None:
    field = _fitted(_kernel_field())
    field.psf_kernel.requires_grad_(False)
    before = field.get_transmission()

    field.psf_kernel.data.mul_(2.0)
    after = field.get_transmission()

    assert torch.allclose(after, 2.0 * before, rtol=1e-4, atol=1e-6)


def test_a_learnable_kernel_carries_its_gradient() -> None:
    field = _fitted(_kernel_field())

    field.get_transmission().abs().sum().backward()

    assert field.psf_kernel.grad is not None
    assert float(field.psf_kernel.grad.abs().sum()) > 0


def test_a_field_of_the_wrong_shape_is_refused() -> None:
    wavefront = torch.ones((32, 32), dtype=torch.complex64)
    field = PSFSLMField.from_calibration_data(
        _calibration(wavefront), FOCAL_LENGTH, PSF_EXTENT
    )
    with pytest.raises(ValueError, match="wavefront to fit"):
        field.initialize_from_geometry(_geometry())
