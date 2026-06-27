from ..propagators import FourierLensFFT
from ..diagonal_elements import StaticSLMField
from ..geometric_transforms import PartialAffineTransform
from ..virtual_slms.abstract import VirtualSLM
from ..complex_amplitude import FieldGeometry

from .abstract import SLMFourierLensModel


class SLMFFTAffine(SLMFourierLensModel):
    virtual_slm: VirtualSLM
    constant_field: StaticSLMField
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
    ) -> None:
        super().__init__(
            input_geometry=input_geometry,
            virtual_slm=virtual_slm,
            constant_field=static_slm_field,
            fourier_lens=FourierLensFFT(
                focal_length, padded_resolution=padded_resolution
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
            "static_slm_field": self.constant_field,
            "padded_resolution": self.fourier_lens._padded_resolution_init,
            "camera_angle": float(self.affine_transform.init_angle),
            "camera_shift": tuple(self.affine_transform.init_shift),
        }
