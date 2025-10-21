from collections import OrderedDict
import torch

from ..propagators import FourierLensFFT
from ..elements import ConstantSLMField, PartialAffineTransform
from ..virtual_slms.abstract import VirtualSLM

from slmsuite.hardware.cameras.camera import Camera

from .abstract import SLMCameraModel


class SLMFFTAffine(SLMCameraModel):
    def __init__(
        self,
        virtual_slm: VirtualSLM,
        camera: Camera,
        focal_length: float,
        constant_field_slm: torch.Tensor,
        padded_resolution: tuple[int, int] = (2048, 2048),
        device: str = "cpu",
    ) -> None:
        # Create constant field module
        constant_field = ConstantSLMField(
            init_field=constant_field_slm,
            pixel_pitch=virtual_slm.slm.pitch_um * 1e-6,
            device=device,
        )

        # Create the Fourier lens module
        fourier_lens = FourierLensFFT(
            focal_length=focal_length,
            wavelength=virtual_slm.slm.wav_um * 1e-6,
            resolution_in=virtual_slm.slm.shape,
            pixel_pitch_in=virtual_slm.slm.pitch_um[0] * 1e-6,
            padded_resolution=padded_resolution,
            device=device,
            fft_kwargs={"norm": "ortho"},
        )

        # Calculate scaling factor and shift for the affine transformation
        scale = tuple(
            fourier_lens.pixel_size_out[i] / (camera.pitch_um[i] * 1e-6)
            for i in range(2)
        )[::-1]

        shift = tuple(
            (camera.shape[i] - fourier_lens.resolution_out[i] * scale[i]) / 2
            for i in range(2)
        )[::-1]

        # Create the affine transform module
        affine_transform = PartialAffineTransform(
            resolution_in=fourier_lens.padded_resolution,
            pixel_pitch_in=fourier_lens.pixel_size_out[0],
            resolution_out=camera.shape,
            scale=scale,
            shift=shift,
            angle=0,
            device=device,
            verbose=False,
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
