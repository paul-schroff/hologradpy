"""PSFSLMField seeded from a measured SLM-plane field or a captured focal spot, and the
transmission held while the kernel is frozen.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from hologradpy.grids import odd_sample_count, plane_center
from hologradpy.optics.complex_amplitude import ComplexAmplitude, FieldGeometry
from hologradpy.optics.modules.slm_fields import PSFSLMField
from hologradpy.optics.modules.virtual_slms import VirtualSLM
from hologradpy.optics.systems import SLMCZT
from hologradpy.utils import as_image

SLM_RESOLUTION = (64, 64)
SLM_PIXEL_SIZE = 12.5e-6
WAVELENGTH = 0.670e-6
FOCAL_LENGTH = 0.1
# The kernel's pitch is lambda f / (2 L), 41.9 um, so this extent takes 21 samples.
PSF_EXTENT = 860e-6
KERNEL_SIZE = 21
# The pitch of a captured spot, finer than the kernel's.
CAMERA_PITCH = 20e-6


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


def _spot_with_a_side_lobe(
    lobe_offset: tuple[int, int], waist: float = 80e-6
) -> torch.Tensor:
    """The amplitude of a captured focal spot at :data:`CAMERA_PITCH`, on a crop of the
    kernel's extent. A second lobe of 0.6 times the peak amplitude sits
    ``lobe_offset``, ``(columns, rows)``, from the centre pixel of the crop.
    """
    side = odd_sample_count(PSF_EXTENT, CAMERA_PITCH)
    # The row and the column of each pixel, counted from the centre pixel.
    offsets = torch.arange(side, dtype=torch.float64) - side // 2
    rows, columns = torch.meshgrid(offsets, offsets, indexing="ij")
    lobe_columns, lobe_rows = lobe_offset
    waist_pixels = waist / CAMERA_PITCH
    spot = torch.exp(-(columns**2 + rows**2) / waist_pixels**2) + 0.6 * torch.exp(
        -((columns - lobe_columns) ** 2 + (rows - lobe_rows) ** 2) / waist_pixels**2
    )
    return spot.float()


def test_the_seeded_model_images_a_captured_spot_upright() -> None:
    """A captured spot seeds the kernel, and the model images the spot the way the
    camera captured it.

    The spot's second lobe sits 10 columns right of its centre and 6 rows above it.
    The model's camera is sampled at the spot's pitch, so the image of a flat phase
    holds the lobe 10 columns right of the camera's centre and 6 rows above it, at
    0.6 squared of the peak intensity. The point reflection of that pixel about the
    centre stays dark.
    """
    lobe_columns, lobe_rows = 10, -6
    camera_resolution = (64, 64)
    model = SLMCZT(
        input_geometry=_geometry(),
        virtual_slm=VirtualSLM(full_scale_cycles=1.0),
        camera_resolution=camera_resolution,
        camera_pixel_size=(CAMERA_PITCH, CAMERA_PITCH),
        focal_length=FOCAL_LENGTH,
        slm_field=PSFSLMField(
            FOCAL_LENGTH,
            PSF_EXTENT,
            init_psf_kernel=_spot_with_a_side_lobe((lobe_columns, lobe_rows)),
            init_psf_kernel_pixel_size=(CAMERA_PITCH, CAMERA_PITCH),
        ),
    )

    with torch.no_grad():
        intensity = as_image(model().intensity)

    center_column, center_row = plane_center(camera_resolution)
    peak = float(intensity.max())
    assert divmod(int(intensity.argmax()), camera_resolution[1]) == (
        center_row,
        center_column,
    )
    lobe = float(intensity[center_row + lobe_rows, center_column + lobe_columns])
    reflected = float(intensity[center_row - lobe_rows, center_column - lobe_columns])
    assert lobe / peak == pytest.approx(0.6**2, rel=0.1)
    assert reflected / peak < 1e-2
