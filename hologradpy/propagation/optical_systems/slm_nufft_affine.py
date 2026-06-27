import torch

from ..propagators import FourierLensNUFFT
from ..diagonal_elements import StaticSLMField
from ..geometric_transforms import PartialAffineTransform
from ..virtual_slms.abstract import VirtualSLM
from ..complex_amplitude import FieldGeometry

from .abstract import SLMFourierLensModel


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
    constant_field: StaticSLMField
    fourier_lens: FourierLensNUFFT
    affine_transform: PartialAffineTransform

    def __init__(
        self,
        input_geometry: FieldGeometry,
        virtual_slm: VirtualSLM,
        camera_resolution: tuple[int, int],
        camera_pixel_size: tuple[float, float],
        focal_length: float,
        constant_field_slm: torch.Tensor,
        nufft_resolution: tuple[int, int] | None = None,
        camera_angle: float = 0.0,
        camera_shift: tuple[float, float] = (0.0, 0.0),
    ) -> None:
        # The NUFFT outputs a slightly oversized grid so the learnable affine
        # has margin to shift/rotate without cropping signal.
        if nufft_resolution is None:
            nufft_resolution = tuple(
                int(camera_resolution[i] * 1.1) for i in range(2)
            )

        super().__init__(
            input_geometry=input_geometry,
            virtual_slm=virtual_slm,
            constant_field=StaticSLMField(constant_field_slm),
            fourier_lens=FourierLensNUFFT(
                focal_length,
                resolution_out=nufft_resolution,
                pixel_size_out=camera_pixel_size,
                shift=camera_shift,
                angle=camera_angle,
            ),
            affine_transform=PartialAffineTransform(
                resolution_out=camera_resolution,
                pixel_size_out=camera_pixel_size,
            ),
        )

    def get_checkpoint_spec(self) -> dict[str, object]:
        camera_pixel_size = tuple(
            float(value)
            for value in self.affine_transform.pixel_size_out.detach()
            .cpu()
            .reshape(-1)[:2]
            .tolist()
        )

        return {
            "input_geometry": self.input_geometry,
            "virtual_slm": self.virtual_slm,
            "camera_resolution": tuple(self.affine_transform.resolution_out),
            "camera_pixel_size": camera_pixel_size,
            "focal_length": float(self.fourier_lens.focal_length),
            "constant_field_slm": self.constant_field.init_field,
            "nufft_resolution": tuple(self.fourier_lens.resolution_out),
            "camera_angle": float(self.fourier_lens.angle_init),
            "camera_shift": tuple(self.fourier_lens.shift_init),
        }
