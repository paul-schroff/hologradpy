from collections import OrderedDict
import torch

from ..propagators import FourierLensFFT
from ..elements import ConstantSLMField, PartialAffineTransform
from ..virtual_slms.abstract import VirtualSLM

from ...hardware.utils import CameraData

from .abstract import SLMCameraModel


class SLMFFTAffine(SLMCameraModel):
    def __init__(
        self,
        virtual_slm: VirtualSLM,
        camera_data: CameraData,
        focal_length: float,
        constant_field_slm: torch.Tensor,
        padded_resolution: tuple[int, int] = (2048, 2048),
        camera_angle: float = 0.0,
        camera_shift: tuple[float, float] = (0.0, 0.0),
        device: str = "cpu",
        verbose: bool = False,
    ) -> None:
        camera_resolution = tuple(camera_data.shape[i] for i in range(2))
        camera_pixel_size = tuple(
            camera_data.pitch_um[i] * 1e-6 for i in range(2)
        )

        # Create constant field module
        constant_field = ConstantSLMField(
            init_field=constant_field_slm,
            pixel_pitch=virtual_slm.pixel_size_in[0],
            device=device,
        )

        # Create the Fourier lens module
        fourier_lens = FourierLensFFT(
            focal_length=focal_length,
            wavelength=virtual_slm.wavelength,
            resolution_in=virtual_slm.resolution_in,
            pixel_pitch_in=virtual_slm.pixel_size_in[0],
            padded_resolution=padded_resolution,
            device=device,
            fft_kwargs={"norm": "ortho"},
        )

        # Create the affine transform module
        affine_transform = PartialAffineTransform(
            resolution_in=fourier_lens.padded_resolution,
            resolution_out=camera_resolution,
            pixel_size_in=fourier_lens.pixel_size_out,
            pixel_size_out=camera_pixel_size,
            shift=camera_shift,
            angle=camera_angle,
            device=device,
            verbose=verbose,
        )

        # Adding modules to the nn.Sequential container in the desired order.
        super().__init__(
            OrderedDict([
                ("virtual_slm", virtual_slm),
                ("constant_field", constant_field),
                ("fourier_lens", fourier_lens),
                ("affine_transform", affine_transform),
            ])
        )
