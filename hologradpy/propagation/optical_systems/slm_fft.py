import torch

from ..complex_amplitude import FieldGeometry, ComplexAmplitude

from ..propagators import FourierLensFFT
from ..diagonal_elements import StaticSLMField
from ..virtual_slms.abstract import VirtualSLM

from .abstract import OpticalSystem


class SLMFFT(OpticalSystem):
    virtual_slm: VirtualSLM
    constant_field: StaticSLMField
    fourier_lens: FourierLensFFT

    def __init__(
        self,
        input_geometry: FieldGeometry,
        focal_length: float,
        constant_field_slm: ComplexAmplitude,
        init_phase: torch.Tensor | None = None,
        padded_resolution: tuple[int, int] = (2048, 2048),
    ) -> None:
        super().__init__(
            input_geometry=input_geometry,
            virtual_slm=VirtualSLM(phase_scaling=1.0, init_phase=init_phase),
            constant_field=StaticSLMField(constant_field_slm),
            fourier_lens=FourierLensFFT(
                focal_length, padded_resolution=padded_resolution
            ),
        )

    def get_checkpoint_spec(self) -> dict[str, object]:
        return {
            "input_geometry": self.input_geometry,
            "focal_length": float(self.fourier_lens.focal_length.item()),
            "constant_field_slm": self.constant_field.init_field,
            "init_phase": self.virtual_slm.init_phase,
            "padded_resolution": self.fourier_lens._padded_resolution_init,
        }
