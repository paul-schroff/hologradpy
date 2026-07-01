import torch

from ..propagators import FourierLensFFT
from ..diagonal_elements import StaticSLMField
from ..geometric_transforms import PartialAffineTransform
from ..virtual_slms.abstract import VirtualSLM
from ..complex_amplitude import FieldGeometry

from .abstract import SLMFourierLensModel


class SLMFFTAffine(SLMFourierLensModel):
    virtual_slm: VirtualSLM
    static_slm_field: StaticSLMField
    fourier_lens: FourierLensFFT
    affine_transform: PartialAffineTransform

    def __init__(
        self,
        input_geometry: FieldGeometry,
        virtual_slm: VirtualSLM,
        camera_resolution: tuple[int, int],
        camera_pixel_size: tuple[float, float],
        focal_length: float,
        static_slm_field: StaticSLMField,
        padded_resolution: tuple[int, int] = (2048, 2048),
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
            fourier_lens=FourierLensFFT(
                focal_length,
                padded_resolution=padded_resolution,
                power_normalized=power_normalized,
            ),
            affine_transform=PartialAffineTransform(
                resolution_out=camera_resolution,
                pixel_size_out=camera_pixel_size,
                shift=camera_shift,
                angle=camera_angle,
            ),
        )

    def get_checkpoint_spec(self) -> dict[str, object]:
        camera_pixel_size = tuple(
            float(value)
            for value in self.affine_transform.pixel_size_out.detach()
            .cpu()
            .reshape(-1)[:2]
            .tolist()
        )

        return {
            "input_geometry": self.input_geometry,
            "virtual_slm": self.virtual_slm,
            "camera_resolution": tuple(self.affine_transform.resolution_out),
            "camera_pixel_size": camera_pixel_size,
            "focal_length": float(self.fourier_lens.focal_length.item()),
            "static_slm_field": self.static_slm_field,
            "padded_resolution": self.fourier_lens._padded_resolution_init,
            "camera_angle": float(self.affine_transform.init_angle),
            "camera_shift": tuple(self.affine_transform.init_shift),
            "power_normalized": self.fourier_lens.power_normalized,
            "pointing_focal_shift_std": self.pointing_focal_shift_std,
        }
