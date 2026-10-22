"""Geometric (resampling) transforms of the electric field."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from jaxtyping import Float

from ...geometry import homogeneous_matrix, rotation_matrix_from_angle
from ...grids import pixel_frame_angle, pixel_frame_vector, plane_center
from .abstract import capture_init, OpticsModule
from .learnable_partial_affine import LearnablePartialAffine
from ..complex_amplitude import ComplexAmplitude


class GeometricWarp(OpticsModule):
    """Differentiable bilinear resampling of the field through an affine pixel matrix.

    Output pixels that map outside the input are zero. The map is parameterized by
    :attr:`focal_plane_partial_affine`, which the affine optical systems calibrate.
    The centre of the input plane lands on the centre of the output plane
    (:func:`~hologradpy.grids.plane_center`) when the partial affine holds no shift.
    """

    @capture_init
    def __init__(
        self: GeometricWarp,
        resolution_out: tuple[int, int],
        pixel_size_out: tuple[float, float],
        scale_factor: tuple[float, float] = (1, 1),
        shift: tuple[float, float] = (0, 0),
        angle: float = 0.0,
    ) -> None:
        """
        Args:
            resolution_out: The ``(height, width)`` of the output plane, the camera
                grid.
            pixel_size_out: The ``(y, x)`` pitch of the output plane in metres.
            scale_factor: The zoom of the input plane along the output ``(x, y)``. A
                negative zoom mirrors the plane along that axis.
            shift: The shift of the input plane on the output plane, in output pixels
                along ``(x, y)``. The centre of the input plane lands this far from the
                centre pixel of the output plane.
            angle: The turn of the input plane, in degrees from +x towards +y.
        """
        super().__init__(pixel_size_out, resolution_out)

        self.focal_plane_partial_affine = LearnablePartialAffine(
            resolution_out,
            scale_factor=scale_factor,
            shift=shift,
            angle=angle,
            learnable=False,
        )

    def lazy_init(self, complex_amplitude: ComplexAmplitude) -> None:
        number_of_wavelengths = complex_amplitude.wavelength.numel()

        self.focal_plane_partial_affine.to(
            device=complex_amplitude.device, dtype=complex_amplitude.dtype_r
        )

        options = {
            "dtype": complex_amplitude.dtype_r,
            "device": complex_amplitude.device,
        }
        center_in = torch.tensor(plane_center(self.resolution_in), **options)
        center_out = torch.tensor(plane_center(self.resolution_out), **options)
        self.rotation_center = center_in.repeat(number_of_wavelengths, 1)
        self.shift_center = (center_out - center_in).repeat(number_of_wavelengths, 1)

        self.affine_matrix = self.get_affine_matrix()

    def get_affine_matrix(self) -> Float[torch.Tensor, "n_wavelengths 3 3"]:
        """The matrix mapping input ``(column, row)`` pixels to output pixels, per
        wavelength.

        The field is turned by ``angle`` about the rotation centre, then zoomed by
        ``scale_factor`` along the output x and y, then shifted by ``shift_center +
        shift``. The turn acts in metres. The input pitch takes the input pixels to
        metres, and the output pitch takes the metres to output pixels. The angle and
        the shift are physical, so both are converted to the pixel frame first.
        """
        partial_affine = self.focal_plane_partial_affine
        # The pitches are (y, x), flipped to the (x, y) of the partial affine. The
        # output pitch is (2,) during lazy_init and (n_wavelengths, 2) after it.
        return _pixel_matrix(
            self.pixel_size_in.flip(-1),
            partial_affine.scale_factor / self.pixel_size_out.flip(-1),
            partial_affine.angle,
            partial_affine.shift,
            self.shift_center,
            self.rotation_center,
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
        warped = warp_affine(channels, affine_matrix, self.resolution_out)
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


def warp_image(
    image: torch.Tensor,
    pixel_size_in: tuple[float, float],
    resolution_out: tuple[int, int],
    pixel_size_out: tuple[float, float],
    shift: tuple[float, float] = (0.0, 0.0),
    angle_degrees: float = 0.0,
) -> torch.Tensor:
    """Return a real ``image`` resampled onto an output plane, as :class:`GeometricWarp`
    resamples a field.

    The image is turned counterclockwise by ``angle_degrees`` about its centre pixel,
    then scaled from its pitch to the output pitch along the output x and y. The turn
    acts in metres. Its centre pixel then lands on the output's centre pixel, moved by
    ``shift``. Output pixels that map outside the image are 0.

    Args:
        image: The ``(height, width)`` real image.
        pixel_size_in: The image's ``(y, x)`` pixel size.
        resolution_out: The output's ``(height, width)``.
        pixel_size_out: The output's ``(y, x)`` pixel size.
        shift: ``(x, y)`` in output pixels.
        angle_degrees: The angle in degrees.

    Returns:
        torch.Tensor: The ``(height, width)`` resampled image, in the dtype and on the
        device of ``image``.
    """
    like = {"dtype": image.dtype, "device": image.device}
    rotation_center = torch.tensor(plane_center(tuple(image.shape)), **like)
    shift_center = torch.tensor(plane_center(resolution_out), **like) - rotation_center
    matrix = _pixel_matrix(
        torch.tensor(pixel_size_in, **like).flip(-1),
        1.0 / torch.tensor(pixel_size_out, **like).flip(-1),
        torch.tensor(float(angle_degrees), **like),
        torch.tensor(shift, **like),
        shift_center,
        rotation_center,
    )
    return warp_affine(image[None, None], matrix[None], resolution_out)[0, 0]


def _pixel_matrix(
    metres_per_input_pixel: torch.Tensor,
    output_pixels_per_metre: torch.Tensor,
    angle: torch.Tensor,
    shift: torch.Tensor,
    shift_center: torch.Tensor,
    rotation_center: torch.Tensor,
) -> torch.Tensor:
    """The homogeneous matrix mapping input ``(column, row)`` pixels to output pixels.

    The input is turned by ``angle`` about ``rotation_center``, then zoomed along the
    output x and y, and then moved by ``shift_center + shift``. The linear part is
    ``diag(output_pixels_per_metre) @ R @ diag(metres_per_input_pixel)``, with ``R``
    the turn by the pixel-frame angle, so the turn acts in metres. The angle and the
    ``(x, y)`` shift are physical, so both are converted to the pixel frame first.

    Args:
        metres_per_input_pixel: The ``(x, y)`` pitch of the input, in metres.
        output_pixels_per_metre: The output pixels per metre along the output
            ``(x, y)``, the zoom divided by the output pitch.
        angle: The turn in degrees from +x towards +y.
        shift: The ``(x, y)`` shift in output pixels.
        shift_center: The ``(column, row)`` offset from the rotation centre to the
            output's centre pixel.
        rotation_center: The ``(column, row)`` input pixel that the turn and the zoom
            keep fixed.

    Returns:
        torch.Tensor: The homogeneous matrix, ``(..., 3, 3)``.
    """
    rotation = rotation_matrix_from_angle(pixel_frame_angle(angle))
    linear = (
        output_pixels_per_metre[..., :, None]
        * rotation
        * metres_per_input_pixel[..., None, :]
    )
    shift = torch.stack(pixel_frame_vector(shift))
    return homogeneous_matrix(linear, shift_center + shift, rotation_center)


def _pixel_to_normalized_matrix(
    height: int, width: int, like: torch.Tensor
) -> Float[torch.Tensor, "3 3"]:
    """The matrix taking ``(column, row)`` pixel coordinates to the normalized
    coordinates of :func:`torch.nn.functional.grid_sample` with
    ``align_corners=True``.

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


def warp_affine(
    image: Float[torch.Tensor, "batch channel H W"],
    pixel_matrix: Float[torch.Tensor, "batch 3 3"],
    resolution_out: tuple[int, int],
) -> Float[torch.Tensor, "batch channel H_out W_out"]:
    """Resample real images through affine pixel matrices, one per batch entry.

    Each output pixel ``p`` takes the bilinear interpolation of the input at
    ``pixel_matrix^-1 p``, and pixels that map outside the input are zero. Pixel
    coordinates are ``(column, row)`` with the first pixel centre at the origin.

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
