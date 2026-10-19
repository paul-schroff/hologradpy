"""Fourier lenses that sample the window of the focal plane a partial affine sets."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import Tensor

from ....fourier_optics import (
    fourier_lens_magnification,
    fourier_lens_power_prefactor,
)
from ....fourier_transforms import window_offset_from_pixels
from ....grids import pixel_frame_angle, pixel_frame_vector
from ..abstract import OpticsModule
from ..learnable_partial_affine import LearnablePartialAffine
from ...complex_amplitude import pixel_area

if TYPE_CHECKING:
    from ....roi import ROI


class PartialAffineFourierLens(OpticsModule):
    """A Fourier lens that samples the window of the focal plane its learnable partial
    affine sets.

    The partial affine is :attr:`focal_plane_partial_affine`, a
    :class:`~hologradpy.optics.modules.learnable_partial_affine.LearnablePartialAffine`.
    Its zoom, shift and rotation set the window of the focal plane the transform
    samples. With ``learnable`` set, they can be calibrated by gradient descent. The
    shift, in output pixels along ``(x, y)``, and the angle, from +x towards +y, are
    physical, and :meth:`_sampled_window` converts both to the pixel frame of the
    transforms. The transforms turn the window in radians per sample of the SLM grid,
    so the turn acts in metres for square SLM pixels.

    The transform runs on a grid of ``N`` samples, with a base magnification of
    ``lambda * f / (pixel_in * N * pixel_out)`` per wavelength. With the parameters at
    their identity values (scale 1, shift 0, angle 0), a focal pixel measures
    ``pixel_out``.

    With :attr:`output_roi` set, only that rectangle of the output grid is computed.
    The output takes the shape of the rectangle and the values the whole grid holds
    there.
    """

    def __init__(
        self,
        focal_length: float,
        resolution_out: tuple[int, int],
        pixel_size_out: tuple[float, float],
        shift: tuple[float, float] = (0.0, 0.0),
        angle: float = 0.0,
        learnable: bool = True,
        power_normalized: bool = True,
    ) -> None:
        """
        Args:
            focal_length: The focal length in metres.
            resolution_out: The ``(height, width)`` of the focal plane in pixels.
            pixel_size_out: The ``(height, width)`` pitch of the focal plane in metres.
            shift: The starting translation of the partial affine, in output pixels
                along ``(x, y)``.
            angle: The starting rotation of the partial affine, in degrees from +x
                towards +y.
            learnable: Whether the parameters of the partial affine require a gradient.
            power_normalized: Whether the output carries the optical power of the
                input, through the prefactor ``(du * dv) / (lambda * f)``.
        """
        super().__init__(pixel_size_out, resolution_out)
        self.focal_length: float = focal_length
        self.focal_plane_partial_affine = LearnablePartialAffine(
            resolution_out, shift=shift, angle=angle, learnable=learnable
        )
        self.power_normalized: bool = power_normalized
        self.output_roi: ROI | None = None

    def _sampled_window(
        self, resolution: tuple[int, int]
    ) -> tuple[Tensor, Tensor, float | Tensor]:
        """The window of the focal plane that the partial affine sets.

        Args:
            resolution: The ``(height, width)`` of the grid the transform runs on.

        Returns:
            tuple[Tensor, Tensor, float | Tensor]: The magnification and the window
            offset in rad/sample, each ``(n_wl, 2)`` with the column component first,
            and the angle in radians from +column towards +row. The angle is a tensor
            while a gradient flows through it and a float otherwise, so a transform
            can leave out a rotation of zero.
        """
        partial_affine = self.focal_plane_partial_affine
        scale_factor = partial_affine.scale_factor
        # The base magnification per wavelength, (n_wl, 2). The pitch and the
        # resolution are (y, x), so it is flipped to the (column, row) of the
        # transforms. Its magnitudes along the columns and the rows are those along x
        # and y, so the (x, y) zoom of the partial affine scales it directly.
        base_magnification = fourier_lens_magnification(
            self.input_geometry.wavelength.unsqueeze(-1),
            self.focal_length,
            self.pixel_size_in,
            torch.tensor(
                resolution, device=scale_factor.device, dtype=scale_factor.dtype
            ).unsqueeze(0),
            self.pixel_size_out,
        ).flip(-1)
        magnification = scale_factor.abs() * base_magnification
        # The shift and the angle of the partial affine are physical. The transforms
        # sample the pixel frame, so both are converted to it first.
        offset_column, offset_row = window_offset_from_pixels(
            pixel_frame_vector(partial_affine.shift),
            resolution,
            (magnification[:, 0], magnification[:, 1]),
        )
        angle = torch.deg2rad(pixel_frame_angle(partial_affine.angle))
        if not angle.requires_grad:
            angle = float(angle)
        return magnification, torch.stack((offset_column, offset_row), dim=-1), angle

    def _power_prefactor(self) -> Tensor:
        """The Fourier-lens amplitude prefactor ``(du * dv) / (lambda * f)`` per
        wavelength, shaped ``(1, n_wl, 1, 1)`` for the flattened field.

        The prefactor is computed in float64 and cast to the real dtype of the field.
        The global phase ``1 / i`` is left out, since a global phase leaves the power
        unchanged.
        """
        pixel_size_in = self.pixel_size_in
        area = pixel_area(pixel_size_in)
        wavelength = self.input_geometry.wavelength.to(torch.float64).reshape(-1)
        prefactor = fourier_lens_power_prefactor(area, wavelength, self.focal_length)
        return prefactor.to(pixel_size_in.dtype).reshape(1, -1, 1, 1)
