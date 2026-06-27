"""Modules that modify the electric field in a 2D plane"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import Tensor
from torch.nn import Parameter

from kornia.geometry.transform import get_affine_matrix2d
from kornia.geometry import warp_perspective

from .utils.optics_utils import lens_phase, circular_mask
from .utils.fourier_utils import get_spatial_grid
from .utils.tensor_utils import unsqueeze_to
from .optics_module import OpticsModule, SaveDict
from .complex_amplitude import (
    ComplexAmplitude,
    broadcast_wavelength_operand,
)

if TYPE_CHECKING:
    from ..calibration.wavefront.abstract import WavefrontCalibrationData

class ConstantSLMField(OpticsModule):
    def __init__(
        self: ConstantSLMField,
        init_field: ComplexAmplitude | None = None,
    ) -> None:
        super().__init__()
        self.init_field: ComplexAmplitude | None = init_field
    
    def lazy_init(
        self: ConstantSLMField, complex_amplitude: ComplexAmplitude
    ) -> None:
        super().lazy_init(complex_amplitude)

        if self.init_field is None:
            number_of_wavelengths = complex_amplitude.number_of_wavelengths
            # A uniform default field is wavelength-independent, but the
            # ComplexAmplitude layout requires an explicit wavelength axis when
            # more than one wavelength is present.
            default_shape = (
                self.resolution_in
                if number_of_wavelengths == 1
                else (number_of_wavelengths, *self.resolution_in)
            )
            self.init_field = ComplexAmplitude(
                data=torch.ones(
                    default_shape,
                    dtype=complex_amplitude.dtype,
                    device=complex_amplitude.device
                ),
                wavelength=complex_amplitude.wavelength,
                pixel_size=complex_amplitude.pixel_size,
            )

        self.phase = Parameter(
            torch.tensor(
                self.init_field.phase, 
                dtype=complex_amplitude.dtype_r,
                device=complex_amplitude.device
            ),
            requires_grad=False,
        )

        self.amplitude = Parameter(
            torch.tensor(
                self.init_field.amplitude, 
                dtype=complex_amplitude.dtype_r,
                device=complex_amplitude.device
            ),
            requires_grad=False,
        )

    @classmethod
    def from_file(
        cls, path: str, device: torch.device = "cpu"
    ) -> ConstantSLMField:
        state: SaveDict = torch.load(
            path, map_location=device, weights_only=False
        )
        state_dict = state["state_dict"]
        geometry = state["input_geometry"]

        init_field_data: Tensor = (
            state_dict["amplitude"] * torch.exp(1j * state_dict["phase"])
        )

        init_field = ComplexAmplitude(
            data=init_field_data.to(device),
            wavelength=geometry.wavelength.to(device),
            pixel_size=geometry.pixel_size.to(device),
        )
        return cls(init_field=init_field)

    @classmethod
    def from_calibration_data(
        cls, calibration_data: WavefrontCalibrationData
    ) -> ConstantSLMField:
        return cls(init_field=calibration_data.complex_amplitude)

    def forward(
        self: ConstantSLMField, complex_amplitude: ComplexAmplitude
    ) -> ComplexAmplitude:
        ndim = complex_amplitude.ndim

        input_field = unsqueeze_to(complex_amplitude, ndim)
        amplitude = unsqueeze_to(self.amplitude, ndim)
        phase = unsqueeze_to(self.phase, ndim)

        modified_field = input_field * amplitude * torch.exp(1j * phase)

        return modified_field.with_geometry(
            wavelength=complex_amplitude.wavelength,
            pixel_size=self.pixel_size_out,
        )


class PartialAffineTransform(OpticsModule):
    def __init__(
        self: PartialAffineTransform,
        resolution_out: tuple[int, int],
        pixel_size_out: tuple[float, float],
        scale_factor: tuple[float, float] = (1, 1),
        shift: tuple[float, float] = (0, 0),
        angle: float = 0.0,
        rotation_center_shift: tuple[float, float] = (0, 0),
        verbose: bool = False,
    ) -> None:
        super().__init__(pixel_size_out, resolution_out)

        self.verbose = verbose

        self.init_scale_factor = scale_factor
        self.init_shift = shift
        self.init_angle = angle
        self.init_rotation_center_shift = rotation_center_shift

        self.register_parameter("scale_factor", None)
        self.register_parameter("shift", None)
        self.register_parameter("angle", None)
        self.register_parameter("rotation_center_shift", None)

        self.scale_factor: Parameter | None
        self.shift: Parameter | None
        self.angle: Parameter | None
        self.rotation_center_shift: Parameter | None

    def lazy_init(self, complex_amplitude: ComplexAmplitude) -> None:
        super().lazy_init(complex_amplitude)

        number_of_wavelengths = complex_amplitude.wavelength.numel()

        # Scaling factor
        self.scale_factor = Parameter(
            torch.tensor(
                self.init_scale_factor,
                dtype=complex_amplitude.dtype_r,
                device=complex_amplitude.device,
            ),
            requires_grad=False
        )

        # Shift from the center in pixels
        self.shift = Parameter(
            torch.tensor(
                self.init_shift, 
                dtype=complex_amplitude.dtype_r, 
                device=complex_amplitude.device
            ),
            requires_grad=False
        )

        # Rotation angle in degrees
        self.angle = Parameter(
            torch.tensor(
                [self.init_angle] * number_of_wavelengths, 
                dtype=complex_amplitude.dtype_r, 
                device=complex_amplitude.device
            ),
            requires_grad=False
        )

        # Shift of the rotation center relative to the shift_center
        self.rotation_center_shift = Parameter(
            torch.tensor(
                self.init_rotation_center_shift, 
                dtype=complex_amplitude.dtype_r, 
                device=complex_amplitude.device
            ).repeat(number_of_wavelengths, 1),
            requires_grad=False
        )

        # Setting scaling to pixel size ratios
        self.scale = (self.pixel_size_in / self.pixel_size_out).fliplr()

        # Setting the rotation center to the center of the input image
        rotation_center = tuple(
            self.resolution_in[i] // 2 for i in range(2)
        )[::-1]
        self.rotation_center = torch.tensor(
            rotation_center, 
            dtype=complex_amplitude.dtype_r, 
            device=complex_amplitude.device
        ).repeat(number_of_wavelengths, 1)  # Repeat for batch dimension if needed

        # Shift moving the centre of the input image to the center of the 
        # output image
        resolution_out = torch.tensor(
            self.resolution_out, 
            dtype=complex_amplitude.dtype_r, 
            device=complex_amplitude.device
        )
        resolution_in = torch.tensor(
            self.resolution_in, 
            dtype=complex_amplitude.dtype_r, 
            device=complex_amplitude.device
        )

        self.shift_center = (
            self.rotation_center * (self.scale - 1)
            + (resolution_out - resolution_in * self.scale) / 2
        ).fliplr()

        self.affine_matrix = self.get_affine_matrix()

    @classmethod
    def from_file(
        cls, path: str, device: torch.device = "cpu"
    ) -> PartialAffineTransform:
        state: SaveDict = torch.load(
            path, map_location=device, weights_only=False
        )
        state_dict = state["state_dict"]
        return cls(
            resolution_out=state["resolution_out"],
            pixel_size_out=tuple(state["pixel_size_out"][0].tolist()),
            scale_factor=tuple(state_dict["scale_factor"].tolist()),
            shift=tuple(state_dict["shift"].tolist()),
            angle=state_dict["angle"][0].item(),
            rotation_center_shift=tuple(
                state_dict["rotation_center_shift"][0].tolist()
            ),
        )

    def get_affine_matrix(self) -> torch.Tensor:
        return get_affine_matrix2d(
            unsqueeze_to((self.shift_center + self.shift), 2),
            unsqueeze_to(self.rotation_center + self.rotation_center_shift, 2),
            unsqueeze_to(self.scale * self.scale_factor, 2),
            unsqueeze_to(self.angle, 1),
        )

    def forward(
        self: PartialAffineTransform, complex_amplitude: ComplexAmplitude
    ) -> ComplexAmplitude:
        """Applies a partial affine transformation to a field of arbitrary
        batch rank ``(*batch, n_wl, H, W)``.

        All leading batch dimensions are collapsed onto kornia's batch axis,
        the per-wavelength affine matrix is tiled to match, and the original
        rank is restored on output.
        """
        if self.verbose:
            print("Scale:", self.scale.data)
            print("Shift:", self.shift.data)
            print("Angle:", self.angle.data)

        number_of_wavelengths = complex_amplitude.number_of_wavelengths

        # Collapse all batch dimensions into a single leading axis and merge
        # (image, wavelength) onto kornia's batch axis with a single channel.
        flat_field, batch_spec = complex_amplitude.flatten_batch()
        number_of_images = flat_field.shape[0]
        field = flat_field.reshape(
            number_of_images * number_of_wavelengths,
            1,
            *complex_amplitude.resolution,
        )

        # The affine matrix is per-wavelength: (n_wl, 3, 3). Tile it across the
        # batch images, keeping wavelength alignment with the row-major
        # (image, wavelength) flattening of ``field`` above.
        self.affine_matrix = self.get_affine_matrix()
        affine_matrix = (
            self.affine_matrix.unsqueeze(0)
            .expand(number_of_images, -1, -1, -1)
            .reshape(number_of_images * number_of_wavelengths, 3, 3)
        )

        # Kornia does not support complex numbers in warp_perspective(),
        # so we need to split the real and imaginary parts and then
        # recombine them.
        output_real = warp_perspective(
            field.real, affine_matrix, self.resolution_out
        )
        output_imag = warp_perspective(
            field.imag, affine_matrix, self.resolution_out
        )

        transformed_field = output_real + 1j * output_imag

        # Normalize to conserve optical power
        transformed_field = transformed_field / self.scale.prod().sqrt()

        # Restore canonical (N, n_wavelengths, H_out, W_out) layout.
        transformed_field = transformed_field.reshape(
            number_of_images, number_of_wavelengths, *self.resolution_out
        )

        return ComplexAmplitude.unflatten_batch(
            transformed_field,
            batch_spec,
            complex_amplitude.wavelength,
            self.pixel_size_out,
        )


class SimpleLens(OpticsModule):
    """Ideal thin lens: an apertured quadratic phase mask.

    A diagonal (per-pixel) operator that preserves sampling, so the output
    pixel size and resolution match the input. The lens phase is built
    per-wavelength at lazy initialisation.
    """

    def __init__(
        self: SimpleLens,
        focal_length: float,
        aperture_radius: float,
    ) -> None:
        super().__init__()

        self.focal_length: float = focal_length
        self.aperture_radius: float = aperture_radius

    def lazy_init(
        self: SimpleLens, complex_amplitude: ComplexAmplitude
    ) -> None:
        super().lazy_init(complex_amplitude)

        grid_x, grid_y = get_spatial_grid(
            self.resolution_in,
            tuple(self.pixel_size_in[0].tolist()),
            complex_amplitude.device,
        )

        # Per-wavelength lens phase (n_wavelengths, H, W).
        wavenumber = complex_amplitude.wavenumber.reshape(-1, 1, 1)
        phase = lens_phase(
            grid_x.unsqueeze(0), grid_y.unsqueeze(0),
            self.focal_length, wavenumber,
        )

        # Aperture is wavelength-independent; centred on the grid.
        aperture = circular_mask(grid_x, grid_y, self.aperture_radius)

        # Complex transmission of the apertured lens.
        self.register_buffer(
            "transmission", aperture * torch.exp(1j * phase)
        )

    def forward(
        self: SimpleLens, complex_amplitude: ComplexAmplitude
    ) -> ComplexAmplitude:
        transmission = broadcast_wavelength_operand(
            self.transmission, complex_amplitude.ndim
        )
        modulated = complex_amplitude * transmission
        return modulated.with_geometry(
            wavelength=complex_amplitude.wavelength,
            pixel_size=self.pixel_size_out,
        )

    def adjoint(
        self: SimpleLens, complex_amplitude: ComplexAmplitude
    ) -> ComplexAmplitude:
        """Conjugate transpose of :meth:`forward` (a diagonal operator, so the
        adjoint multiplies by the conjugate transmission)."""
        self._ensure_initialized()
        transmission = broadcast_wavelength_operand(
            self.transmission.conj(), complex_amplitude.ndim
        )
        modulated = complex_amplitude * transmission
        return modulated.with_geometry(
            wavelength=complex_amplitude.wavelength,
            pixel_size=self.pixel_size_out,
        )


# TODO: Implement doublet lens module
