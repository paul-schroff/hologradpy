from collections import OrderedDict
import torch

from ..propagators import FourierLensFFT
from ..elements import ConstantSLMField
from ..virtual_slms.abstract import VirtualSLM

from .abstract import SLMCameraModel


class SLMFFT(SLMCameraModel):
    def __init__(
        self,
        virtual_slm: VirtualSLM,
        focal_length: float,
        constant_field_slm: torch.Tensor,
        padded_resolution: tuple[int, int] = (2048, 2048),
        device: str = "cpu",
    ) -> None:
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

        # Adding modules to the nn.Sequential container in the desired order.
        super().__init__(
            OrderedDict([
                ("virtual_slm", virtual_slm),
                ("constant_field", constant_field),
                ("fourier_lens", fourier_lens),
            ])
        )
