import torch

from ..complex_amplitude import FieldGeometry

from ..propagators import FourierLensFFT
from ..diagonal_elements import StaticSLMField
from ..virtual_slms.abstract import VirtualSLM

from .abstract import SLMFourierLensModel


class SLMFFT(SLMFourierLensModel):
    virtual_slm: VirtualSLM
    static_slm_field: StaticSLMField
    fourier_lens: FourierLensFFT

    def __init__(
        self,
        input_geometry: FieldGeometry,
        virtual_slm: VirtualSLM,
        static_slm_field: StaticSLMField,
        focal_length: float,
        padded_resolution: tuple[int, int] = (2048, 2048),
        pointing_focal_shift_std: float | tuple[float, float] | None = None,
        pointing_generator: torch.Generator | None = None,
    ) -> None:
        super().__init__(
            input_geometry=input_geometry,
            focal_length=focal_length,
            pointing_focal_shift_std=pointing_focal_shift_std,
            pointing_generator=pointing_generator,
            virtual_slm=virtual_slm,
            static_slm_field=static_slm_field,
            fourier_lens=FourierLensFFT(
                focal_length, padded_resolution=padded_resolution
            ),
        )

    def get_checkpoint_spec(self) -> dict[str, object]:
        return {
            "input_geometry": self.input_geometry,
            "virtual_slm": self.virtual_slm,
            "static_slm_field": self.static_slm_field,
            "focal_length": float(self.fourier_lens.focal_length.item()),
            "padded_resolution": self.fourier_lens._padded_resolution_init,
            "pointing_focal_shift_std": self.pointing_focal_shift_std,
        }
