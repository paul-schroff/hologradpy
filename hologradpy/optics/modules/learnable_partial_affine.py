"""The learnable partial affine of an output plane."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import Parameter

from ...geometry import PartialAffineTransform
from ...grids import pixel_frame_angle, pixel_frame_vector, plane_center


class LearnablePartialAffine(nn.Module):
    """A learnable partial affine of an output plane, applied by the module that owns
    it.

    ``scale_factor`` is the zoom along ``(x, y)``, ``shift`` the translation in output
    pixels along ``(x, y)``, and ``angle`` the rotation in degrees from +x towards +y.
    All three are physical. The plane is turned by ``angle`` and then zoomed by
    ``scale_factor`` along the output x and y, both about the centre of the output
    plane, and then shifted by ``shift``. A Fourier lens or a field warp holds one,
    reads its parameters in the forward pass and converts them to the pixel frame. The
    camera-aware optical systems calibrate it from a camera mapping through :meth:`set`.

    The parameters are made in float64 on the CPU. The owning module moves them to the
    device and the dtype of its field when it is first run.
    """

    def __init__(
        self,
        resolution_out: tuple[int, int],
        scale_factor: tuple[float, float] = (1.0, 1.0),
        shift: tuple[float, float] = (0.0, 0.0),
        angle: float = 0.0,
        learnable: bool = True,
    ) -> None:
        """
        Args:
            resolution_out: The ``(height, width)`` of the output plane. The zoom and
                the rotation keep its centre fixed.
            scale_factor: The starting zoom along ``(x, y)``.
            shift: The starting translation in output pixels along ``(x, y)``.
            angle: The starting rotation in degrees from +x towards +y.
            learnable: Whether the parameters require a gradient.
        """
        super().__init__()
        self.center: tuple[int, int] = plane_center(resolution_out)
        self.scale_factor = Parameter(
            torch.tensor(scale_factor, dtype=torch.float64), requires_grad=learnable
        )
        self.shift = Parameter(
            torch.tensor(shift, dtype=torch.float64), requires_grad=learnable
        )
        self.angle = Parameter(
            torch.tensor(float(angle), dtype=torch.float64), requires_grad=learnable
        )

    def set(self, transform: PartialAffineTransform) -> None:
        """Set the parameters to the inverse of a fitted camera -> model similarity,
        starting from identity.

        The similarity maps camera pixels to model pixels. Its inverse is converted to
        the physical frame of the parameters.

        Args:
            transform: The camera -> model similarity fitted by the camera mappers.
        """
        scale, angle_degrees, shift = transform.inverse().components(self.center)
        with torch.no_grad():
            self.scale_factor.fill_(scale)
            self.shift.copy_(
                torch.tensor(
                    pixel_frame_vector(shift),
                    dtype=self.shift.dtype,
                    device=self.shift.device,
                )
            )
            self.angle.fill_(pixel_frame_angle(angle_degrees))

    def reset(self) -> None:
        """Return to identity, with unit zoom, no translation and no rotation."""
        with torch.no_grad():
            self.scale_factor.fill_(1.0)
            self.shift.zero_()
            self.angle.zero_()
