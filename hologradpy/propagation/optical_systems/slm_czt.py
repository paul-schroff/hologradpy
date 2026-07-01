import torch

from ..propagators import FourierLensCZT
from ..diagonal_elements import StaticSLMField
from ..virtual_slms.abstract import VirtualSLM
from ..complex_amplitude import FieldGeometry

from .abstract import SLMFourierLensModel


class SLMCZT(SLMFourierLensModel):
    """SLM -> exact chirp-z Fourier lens with learnable focal-plane geometry.

    The chirp-z counterpart of :class:`SLMNUFFTAffine`, but *without* a separate
    affine registration stage: :class:`FourierLensCZT` carries learnable
    ``scale_factor`` / ``shift`` / ``angle``, so the focal-plane affine map is
    learned inside the (exact, power-correct) lens itself and maps directly onto
    the camera resolution at the camera pixel size. ``camera_angle`` (degrees) /
    ``camera_shift`` (output pixels) seed those learnable parameters.
    """

    virtual_slm: VirtualSLM
    static_slm_field: StaticSLMField
    fourier_lens: FourierLensCZT

    def __init__(
        self,
        input_geometry: FieldGeometry,
        virtual_slm: VirtualSLM,
        camera_resolution: tuple[int, int],
        camera_pixel_size: tuple[float, float],
        focal_length: float,
        static_slm_field: StaticSLMField,
        camera_angle: float = 0.0,
        camera_shift: tuple[float, float] = (0.0, 0.0),
        power_normalized: bool = True,
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
            fourier_lens=FourierLensCZT(
                focal_length,
                resolution_out=camera_resolution,
                pixel_size_out=camera_pixel_size,
                shift=camera_shift,
                angle=camera_angle,
                learnable=True,
                power_normalized=power_normalized,
            ),
        )

    def get_checkpoint_spec(self) -> dict[str, object]:
        camera_pixel_size = tuple(
            float(value)
            for value in self.fourier_lens.pixel_size_out.detach()
            .cpu()
            .reshape(-1)[:2]
            .tolist()
        )

        return {
            "input_geometry": self.input_geometry,
            "virtual_slm": self.virtual_slm,
            "camera_resolution": tuple(self.fourier_lens.resolution_out),
            "camera_pixel_size": camera_pixel_size,
            "focal_length": float(self.fourier_lens.focal_length),
            "static_slm_field": self.static_slm_field,
            "camera_angle": float(self.fourier_lens.angle_init),
            "camera_shift": tuple(self.fourier_lens.shift_init),
            "power_normalized": self.fourier_lens.power_normalized,
            "pointing_focal_shift_std": self.pointing_focal_shift_std,
        }
