from ..complex_amplitude import FieldGeometry

from ..propagators import FourierLensFFT
from ..diagonal_elements import StaticSLMField
from ..virtual_slms.abstract import VirtualSLM

from .abstract import SLMFourierLensModel, capture_init


class SLMFFT(SLMFourierLensModel):
    virtual_slm: VirtualSLM
    static_slm_field: StaticSLMField
    fourier_lens: FourierLensFFT

    @capture_init
    def __init__(
        self,
        input_geometry: FieldGeometry,
        virtual_slm: VirtualSLM,
        static_slm_field: StaticSLMField,
        focal_length: float,
        padded_resolution: tuple[int, int] = (2048, 2048),
        pointing_focal_shift_std: float | tuple[float, float] | None = None,
        pointing_seed: int | None = None,
    ) -> None:
        super().__init__(
            input_geometry=input_geometry,
            focal_length=focal_length,
            pointing_focal_shift_std=pointing_focal_shift_std,
            pointing_seed=pointing_seed,
            virtual_slm=virtual_slm,
            static_slm_field=static_slm_field,
            fourier_lens=FourierLensFFT(
                focal_length, padded_resolution=padded_resolution
            ),
        )
