from ..propagators import FourierLensNUFFT
from ..diagonal_elements import StaticSLMField
from ..geometric_transforms import PartialAffineTransform
from ..virtual_slms.abstract import VirtualSLM
from ..complex_amplitude import FieldGeometry

from .abstract import SLMFourierLensModel, capture_init


class SLMNUFFTAffine(SLMFourierLensModel):
    """SLM -> Fourier lens (NUFFT) -> affine camera registration.

    The ``FourierLensNUFFT`` carries a fixed coarse geometric transform
    (``camera_shift`` / ``camera_angle``, applied via its k-space trajectory)
    and maps onto a slightly oversized grid at the camera pixel size. The
    ``PartialAffineTransform`` then performs the *learnable* fine registration
    onto the camera resolution, because the NUFFT cannot learn those geometric
    parameters efficiently.
    """

    virtual_slm: VirtualSLM
    static_slm_field: StaticSLMField
    fourier_lens: FourierLensNUFFT
    affine_transform: PartialAffineTransform

    @capture_init
    def __init__(
        self,
        input_geometry: FieldGeometry,
        virtual_slm: VirtualSLM,
        camera_resolution: tuple[int, int],
        camera_pixel_size: tuple[float, float],
        focal_length: float,
        static_slm_field: StaticSLMField,
        nufft_resolution: tuple[int, int] | None = None,
        camera_angle: float = 0.0,
        camera_shift: tuple[float, float] = (0.0, 0.0),
        power_normalized: bool = True,
        pointing_focal_shift_std: float | tuple[float, float] | None = None,
        pointing_seed: int | None = None,
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
            virtual_slm=virtual_slm,
            static_slm_field=static_slm_field,
            fourier_lens=FourierLensNUFFT(
                focal_length,
                resolution_out=nufft_resolution,
                pixel_size_out=camera_pixel_size,
                shift=camera_shift,
                angle=camera_angle,
                power_normalized=power_normalized,
            ),
            affine_transform=PartialAffineTransform(
                resolution_out=camera_resolution,
                pixel_size_out=camera_pixel_size,
            ),
        )
