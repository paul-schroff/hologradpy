from __future__ import annotations

from typing import TYPE_CHECKING

from collections import OrderedDict
import torch

from ..propagators import FourierLensNUFFT
from ..diagonal_elements import StaticSLMField
from ..geometric_transforms import PartialAffineTransform
from ..virtual_slms.abstract import VirtualSLM

from .abstract import SLMFourierLensModel

if TYPE_CHECKING:
    from ...hardware.camera_data import CameraData


class SLMNUFFTAffine(SLMFourierLensModel):
    def __init__(
        self,
        virtual_slm: VirtualSLM,
        camera_data: CameraData,
        focal_length: float,
        constant_field_slm: torch.Tensor,
        camera_angle: float = 0.0,
        camera_shift: tuple[float, float] = (0.0, 0.0),
        device: str = "cpu",
        verbose: bool = False,
    ) -> None:
        self.camera_data = camera_data
        self.device_name = device
        self.verbose = verbose
        self.camera_angle = camera_angle
        self.camera_shift = camera_shift
        self.focal_length = focal_length

        camera_resolution = tuple(camera_data.shape[i] for i in range(2))
        camera_pixel_size = tuple(camera_data.pitch_um[i] * 1e-6 for i in range(2))

        # Create constant field module
        constant_field = StaticSLMField(
            init_field=constant_field_slm,
            pixel_pitch=virtual_slm.pixel_size_in[0],
            device=device,
        )

        # Create the Fourier lens module
        fourier_lens = FourierLensNUFFT(
            focal_length=focal_length,
            wavelength=virtual_slm.wavelength,
            resolution_in=virtual_slm.resolution_in,
            pixel_size_in=virtual_slm.pixel_size_in,
            resolution_out=tuple(int(camera_resolution[i] * 1.1) for i in range(2)),
            pixel_size_out=camera_pixel_size,
            scale_factor=(1, 1),
            shift=camera_shift,
            angle=camera_angle,
            device=device,
        )

        # Create the affine transform module
        affine_transform = PartialAffineTransform(
            resolution_in=fourier_lens.resolution_out,
            resolution_out=camera_resolution,
            pixel_size_in=fourier_lens.pixel_size_out,
            pixel_size_out=camera_pixel_size,
            device=device,
            verbose=verbose,
        )

        # Adding modules to the nn.Sequential container in the desired order.
        super().__init__(
            OrderedDict(
                [
                    ("virtual_slm", virtual_slm),
                    ("constant_field", constant_field),
                    ("fourier_lens", fourier_lens),
                    ("affine_transform", affine_transform),
                ]
            )
        )

    def get_checkpoint_spec(self) -> dict[str, object]:
        return {
            "virtual_slm": self.virtual_slm,
            "camera_data": self.camera_data,
            "focal_length": self.focal_length,
            "constant_field_slm": self.constant_field.init_field,
            "camera_angle": self.camera_angle,
            "camera_shift": self.camera_shift,
            "device": self.device_name,
            "verbose": self.verbose,
        }
