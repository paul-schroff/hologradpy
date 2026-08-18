from ..modules.propagators import FourierLensFFT
from ..modules.slm_fields import SLMField
from ..modules.geometric_transforms import GeometricWarp
from ..modules.virtual_slms.abstract import VirtualSLM
from ..complex_amplitude import FieldGeometry

from .abstract import (
    SLMFourierLensModel,
    camera_shift_pixels,
    slm_stages,
    upscaled_padding,
)
from ..modules.abstract import capture_init


class SLMFFTAffine(SLMFourierLensModel):
    """SLM -> padded-FFT Fourier lens -> affine camera registration.

    A :class:`FourierLensFFT` maps onto the padded focal plane and a learnable
    :class:`GeometricWarp` registers it onto the camera. ``camera_angle`` (degrees)
    / ``camera_shift`` (``(x, y)`` metres) seed the warp.
    """

    virtual_slm: VirtualSLM
    slm_field: SLMField
    fourier_lens: FourierLensFFT
    affine_transform: GeometricWarp

    @capture_init
    def __init__(
        self,
        input_geometry: FieldGeometry,
        virtual_slm: VirtualSLM,
        camera_resolution: tuple[int, int],
        camera_pixel_size: tuple[float, float],
        focal_length: float,
        slm_field: SLMField,
        padded_resolution: tuple[int, int] = (2048, 2048),
        camera_angle: float = 0.0,
        camera_shift: tuple[float, float] = (0.0, 0.0),
        power_normalized: bool = True,
        pointing_focal_shift_std: float | tuple[float, float] | None = None,
        pointing_seed: int | None = None,
        grid_cache: bool = False,
    ) -> None:
        super().__init__(
            input_geometry=input_geometry,
            focal_length=focal_length,
            pointing_focal_shift_std=pointing_focal_shift_std,
            pointing_seed=pointing_seed,
            **slm_stages(virtual_slm, slm_field, grid_cache),
            fourier_lens=FourierLensFFT(
                focal_length,
                padded_resolution=upscaled_padding(padded_resolution, virtual_slm),
                power_normalized=power_normalized,
            ),
            affine_transform=GeometricWarp(
                resolution_out=camera_resolution,
                pixel_size_out=camera_pixel_size,
                shift=camera_shift_pixels(camera_shift, camera_pixel_size),
                angle=camera_angle,
            ),
        )

    def affine_module(self) -> GeometricWarp:
        return self.affine_transform
