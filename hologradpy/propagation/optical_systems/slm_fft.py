from ..complex_amplitude import FieldGeometry

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
        virtual_slm: VirtualSLM,
        static_slm_field: StaticSLMField,
        focal_length: float,
        padded_resolution: tuple[int, int] = (2048, 2048),
    ) -> None:
        super().__init__(
            input_geometry=input_geometry,
            virtual_slm=virtual_slm,
            constant_field=static_slm_field,
            fourier_lens=FourierLensFFT(
                focal_length, padded_resolution=padded_resolution
            ),
        )

    def get_checkpoint_spec(self) -> dict[str, object]:
        return {
            "input_geometry": self.input_geometry,
            "virtual_slm": self.virtual_slm,
            "static_slm_field": self.constant_field,
            "focal_length": float(self.fourier_lens.focal_length.item()),
            "padded_resolution": self.fourier_lens._padded_resolution_init,
        }
