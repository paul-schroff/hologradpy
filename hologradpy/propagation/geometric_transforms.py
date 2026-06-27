"""Geometric (resampling) transforms of the electric field."""

from __future__ import annotations

import torch
from torch.nn import Parameter

from kornia.geometry.transform import get_affine_matrix2d
from kornia.geometry import warp_perspective

from ..utils import unsqueeze_to
from .optics_module import OpticsModule, SaveDict
from .complex_amplitude import ComplexAmplitude


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
            requires_grad=False,
        )

        # Shift from the center in pixels
        self.shift = Parameter(
            torch.tensor(
                self.init_shift,
                dtype=complex_amplitude.dtype_r,
                device=complex_amplitude.device,
            ),
            requires_grad=False,
        )

        # Rotation angle in degrees
        self.angle = Parameter(
            torch.tensor(
                [self.init_angle] * number_of_wavelengths,
                dtype=complex_amplitude.dtype_r,
                device=complex_amplitude.device,
            ),
            requires_grad=False,
        )

        # Shift of the rotation center relative to the shift_center
        self.rotation_center_shift = Parameter(
            torch.tensor(
                self.init_rotation_center_shift,
                dtype=complex_amplitude.dtype_r,
                device=complex_amplitude.device,
            ).repeat(number_of_wavelengths, 1),
            requires_grad=False,
        )

        # Setting scaling to pixel size ratios
        self.scale = (self.pixel_size_in / self.pixel_size_out).fliplr()

        # Setting the rotation center to the center of the input image
        rotation_center = tuple(self.resolution_in[i] // 2 for i in range(2))[::-1]
        self.rotation_center = torch.tensor(
            rotation_center,
            dtype=complex_amplitude.dtype_r,
            device=complex_amplitude.device,
        ).repeat(number_of_wavelengths, 1)  # Repeat for batch dimension if needed

        # Shift moving the centre of the input image to the center of the
        # output image
        resolution_out = torch.tensor(
            self.resolution_out,
            dtype=complex_amplitude.dtype_r,
            device=complex_amplitude.device,
        )
        resolution_in = torch.tensor(
            self.resolution_in,
            dtype=complex_amplitude.dtype_r,
            device=complex_amplitude.device,
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
        state: SaveDict = torch.load(path, map_location=device, weights_only=False)
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
        output_real = warp_perspective(field.real, affine_matrix, self.resolution_out)
        output_imag = warp_perspective(field.imag, affine_matrix, self.resolution_out)

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
