from ..modules.propagators import FourierLensNUFFT
from ..modules.slm_fields import SLMField
from ..modules.geometric_transforms import GeometricWarp
from ..modules.virtual_slms.abstract import VirtualSLM
from ..complex_amplitude import FieldGeometry

from .abstract import (
    SLMFourierLensModel,
    camera_shift_pixels,
    slm_stages,
)
from ..modules.abstract import capture_init


class SLMNUFFTAffine(SLMFourierLensModel):
    """SLM -> Fourier lens (NUFFT) -> affine camera registration.

    The ``FourierLensNUFFT`` carries a fixed coarse geometric transform
    (``camera_shift`` ``(x, y)`` metres / ``camera_angle`` degrees, applied
    via its k-space trajectory) and maps onto a slightly oversized grid at the
    camera pixel size. The
    ``GeometricWarp`` then performs the *learnable* fine registration
    onto the camera resolution, because the NUFFT cannot learn those geometric
    parameters efficiently.
    """

    virtual_slm: VirtualSLM
    slm_field: SLMField
    fourier_lens: FourierLensNUFFT
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
        nufft_resolution: tuple[int, int] | None = None,
        camera_angle: float = 0.0,
        camera_shift: tuple[float, float] = (0.0, 0.0),
        power_normalized: bool = True,
        pointing_focal_shift_std: float | tuple[float, float] | None = None,
        pointing_seed: int | None = None,
        grid_cache: bool = False,
    ) -> None:
        # The NUFFT outputs a slightly oversized grid so the learnable affine
        # has margin to shift/rotate without cropping signal.
        if nufft_resolution is None:
            nufft_resolution = tuple(
                int(camera_resolution[i] * 1.1) for i in range(2)
            )

        super().__init__(
            input_geometry=input_geometry,
            focal_length=focal_length,
            pointing_focal_shift_std=pointing_focal_shift_std,
            pointing_seed=pointing_seed,
            **slm_stages(virtual_slm, slm_field, grid_cache),
            fourier_lens=FourierLensNUFFT(
                focal_length,
                resolution_out=nufft_resolution,
                pixel_size_out=camera_pixel_size,
                shift=camera_shift_pixels(camera_shift, camera_pixel_size),
                angle=camera_angle,
                power_normalized=power_normalized,
            ),
            affine_transform=GeometricWarp(
                resolution_out=camera_resolution,
                pixel_size_out=camera_pixel_size,
            ),
        )

    def affine_module(self) -> GeometricWarp:
        return self.affine_transform
