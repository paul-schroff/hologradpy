from __future__ import annotations

from ..modules.propagators import FourierLensNUFFT
from ..modules.slm_fields import SLMField
from ..modules.virtual_slms.abstract import VirtualSLM
from ..complex_amplitude import FieldGeometry

from .abstract import (
    SLMFourierLensModel,
    camera_shift_pixels,
    slm_stages,
    upscaled_padding,
)
from ..modules.abstract import capture_init


class SLMNUFFT(SLMFourierLensModel):
    """SLM -> NUFFT Fourier lens with learnable focal-plane geometry."""

    virtual_slm: VirtualSLM
    slm_field: SLMField
    fourier_lens: FourierLensNUFFT

    @capture_init
    def __init__(
        self,
        input_geometry: FieldGeometry,
        virtual_slm: VirtualSLM,
        camera_resolution: tuple[int, int],
        camera_pixel_size: tuple[float, float],
        focal_length: float,
        slm_field: SLMField,
        camera_angle: float = 0.0,
        camera_shift: tuple[float, float] = (0.0, 0.0),
        padded_resolution: tuple[int, int] | None = None,
        power_normalized: bool = True,
        nufft_kwargs: dict | None = None,
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
            fourier_lens=FourierLensNUFFT(
                focal_length,
                resolution_out=camera_resolution,
                pixel_size_out=camera_pixel_size,
                shift=camera_shift_pixels(camera_shift, camera_pixel_size),
                angle=camera_angle,
                learnable=True,
                power_normalized=power_normalized,
                padded_resolution=upscaled_padding(padded_resolution, virtual_slm),
                nufft_kwargs={} if nufft_kwargs is None else nufft_kwargs,
            ),
        )

    def affine_module(self) -> FourierLensNUFFT:
        return self.fourier_lens
