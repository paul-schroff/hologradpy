from ..complex_amplitude import FieldGeometry

from ..modules.propagators import FourierLensFFT
from ..modules.slm_fields import SLMField
from ..modules.virtual_slms.abstract import VirtualSLM

from .abstract import SLMFourierLensModel, slm_stages, upscaled_padding
from ..modules.abstract import capture_init


class SLMFFT(SLMFourierLensModel):
    virtual_slm: VirtualSLM
    slm_field: SLMField
    fourier_lens: FourierLensFFT

    @capture_init
    def __init__(
        self,
        input_geometry: FieldGeometry,
        virtual_slm: VirtualSLM,
        slm_field: SLMField,
        focal_length: float,
        padded_resolution: tuple[int, int] = (2048, 2048),
        pointing_focal_shift_std: float | tuple[float, float] | None = None,
        pointing_seed: int | None = None,
        grid_cache: bool = False,
    ) -> None:
        super().__init__(
            input_geometry=input_geometry,
            focal_length=focal_length,
            pointing_focal_shift_std=pointing_focal_shift_std,
            pointing_seed=pointing_seed,
            **slm_stages(virtual_slm, slm_field, grid_cache),
            fourier_lens=FourierLensFFT(
                focal_length,
                padded_resolution=upscaled_padding(padded_resolution, virtual_slm),
            ),
        )
