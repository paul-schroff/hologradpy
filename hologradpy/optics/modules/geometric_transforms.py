"""Geometric (resampling) transforms of the electric field."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from jaxtyping import Float
from torch.nn import Parameter

from ...geometry import homogeneous_matrix, rotation_matrix_from_angle
from .abstract import capture_init, OpticsModule
from .learnable_partial_affine import LearnablePartialAffine
from ..complex_amplitude import ComplexAmplitude


class GeometricWarp(OpticsModule):
    """Differentiable geometric warp of the field (a resampling transform).

    Resamples the field through an affine pixel matrix with bilinear interpolation.
    Output pixels that map outside the input are zero. The map is parameterized by
    :attr:`focal_plane_partial_affine`, which the affine optical systems calibrate.
    ``rotation_center_shift`` moves the centre of the rotation and is not part of it.
    """

    @capture_init
    def __init__(
        self: GeometricWarp,
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

        self.init_rotation_center_shift = rotation_center_shift
        self.focal_plane_partial_affine = LearnablePartialAffine(
            resolution_out,
            scale_factor=scale_factor,
            shift=shift,
            angle=angle,
            learnable=False,
        )

        self.register_parameter("rotation_center_shift", None)
        self.rotation_center_shift: Parameter | None

    def lazy_init(self, complex_amplitude: ComplexAmplitude) -> None:
        number_of_wavelengths = complex_amplitude.wavelength.numel()

        self.focal_plane_partial_affine.to(
            device=complex_amplitude.device, dtype=complex_amplitude.dtype_r
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

        # Shift moving the center of the input image to the center of the
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

    def get_affine_matrix(self) -> Float[torch.Tensor, "n_wavelengths 3 3"]:
        """The matrix mapping input ``(x, y)`` pixels to output pixels, per wavelength.

        The field is scaled by the pixel-size ratio times ``scale_factor`` and rotated
        by ``angle``, both about the rotation centre, then shifted by ``shift_center +
        shift``.
        """
        partial_affine = self.focal_plane_partial_affine
        scale = self.scale * partial_affine.scale_factor
        linear = rotation_matrix_from_angle(partial_affine.angle) * scale[..., None, :]
        return homogeneous_matrix(
            linear,
            self.shift_center + partial_affine.shift,
            self.rotation_center + self.rotation_center_shift,
        )

    def forward(
        self: GeometricWarp, complex_amplitude: ComplexAmplitude
    ) -> ComplexAmplitude:
        """Applies a partial affine transformation to a field of arbitrary
        batch rank ``(*batch, n_wl, H, W)``.

        All leading batch dimensions are collapsed onto the batch axis of the
        resampling, the per-wavelength affine matrix is tiled to match, and the
        original rank is restored on output.
        """
        if self.verbose:
            print("Scale:", self.scale.data)
            print("Shift:", self.focal_plane_partial_affine.shift.data)
            print("Angle:", self.focal_plane_partial_affine.angle.data)

        number_of_wavelengths = complex_amplitude.number_of_wavelengths

        # Collapse all batch dimensions into a single leading axis and merge
        # (image, wavelength) onto the batch axis of the resampling.
        flat_field, batch_spec = complex_amplitude.flatten_batch()
        number_of_images = flat_field.shape[0]
        field = flat_field.reshape(
            number_of_images * number_of_wavelengths,
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

        # grid_sample takes real images, so the real and imaginary parts are resampled
        # as two channels on one grid.
        channels = torch.view_as_real(field).permute(0, 3, 1, 2)
        warped = _warp_affine(channels, affine_matrix, self.resolution_out)
        transformed_field = torch.view_as_complex(
            warped.permute(0, 2, 3, 1).contiguous()
        )

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


def _pixel_to_normalized_matrix(
    height: int, width: int, like: torch.Tensor
) -> Float[torch.Tensor, "3 3"]:
    """The matrix taking ``(x, y)`` pixel coordinates to the normalized coordinates of
    :func:`torch.nn.functional.grid_sample` with ``align_corners=True``.

    The first and last pixel centres of each axis map to -1 and 1. A single-pixel axis
    keeps a finite matrix through a denominator of ``1e-14``.

    Args:
        height: The number of rows.
        width: The number of columns.
        like: A tensor providing dtype and device for the matrix.

    Returns:
        The homogeneous normalization matrix.
    """
    return torch.tensor(
        [
            [2.0 / max(width - 1, 1e-14), 0.0, -1.0],
            [0.0, 2.0 / max(height - 1, 1e-14), -1.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=like.dtype,
        device=like.device,
    )


def _warp_affine(
    image: Float[torch.Tensor, "batch channel H W"],
    pixel_matrix: Float[torch.Tensor, "batch 3 3"],
    resolution_out: tuple[int, int],
) -> Float[torch.Tensor, "batch channel H_out W_out"]:
    """Resample real images through affine pixel matrices, one per batch entry.

    Each output pixel ``p`` takes the bilinear interpolation of the input at
    ``pixel_matrix^-1 p``, and pixels that map outside the input are zero. Pixel
    coordinates are ``(x, y)`` with the first pixel centre at the origin.

    Args:
        image: The images to resample.
        pixel_matrix: The affine maps from input to output pixel coordinates.
        resolution_out: The output ``(H, W)``.

    Returns:
        The resampled images.
    """
    height_out, width_out = resolution_out
    height_in, width_in = image.shape[-2:]
    normalize_in = _pixel_to_normalized_matrix(height_in, width_in, pixel_matrix)
    normalize_out = _pixel_to_normalized_matrix(height_out, width_out, pixel_matrix)
    # The map from output to input in normalized coordinates, applied to the output
    # grid axis by axis, so its gradient is an elementwise product and a sum per matrix.
    theta = torch.linalg.inv(
        normalize_out @ pixel_matrix @ torch.linalg.inv(normalize_in)
    )[:, :2, :, None, None]
    x = torch.linspace(-1.0, 1.0, width_out, dtype=image.dtype, device=image.device)
    y = torch.linspace(-1.0, 1.0, height_out, dtype=image.dtype, device=image.device)
    y = y[:, None]
    grid = torch.stack(
        [
            theta[:, 0, 0] * x + theta[:, 0, 1] * y + theta[:, 0, 2],
            theta[:, 1, 0] * x + theta[:, 1, 1] * y + theta[:, 1, 2],
        ],
        dim=-1,
    )
    return F.grid_sample(
        image, grid, mode="bilinear", padding_mode="zeros", align_corners=True
    )
