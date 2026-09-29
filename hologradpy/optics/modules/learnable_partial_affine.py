"""The learnable partial affine of an output plane."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import Parameter

from ...geometry import PartialAffineTransform, inverse_partial_affine_parameters
from ...grids import plane_center


class LearnablePartialAffine(nn.Module):
    """A learnable partial affine of an output plane, applied by the module that owns
    it.

    ``scale_factor`` is the zoom along ``(x, y)``, ``shift`` the translation in output
    pixels along ``(x, y)``, and ``angle`` the rotation in degrees. The zoom and the
    rotation keep the centre of the output plane fixed. A Fourier lens or a field warp
    holds one and reads its parameters in the forward pass. The camera-aware optical
    systems calibrate it from a camera mapping through :meth:`set`.

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
            angle: The starting rotation in degrees.
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

        Args:
            transform: The camera -> model similarity fitted by the camera mappers.
        """
        scale, angle_deg, shift = inverse_partial_affine_parameters(
            transform, self.center
        )
        with torch.no_grad():
            self.scale_factor.fill_(scale)
            self.shift.copy_(
                torch.tensor(shift, dtype=self.shift.dtype, device=self.shift.device)
            )
            self.angle.fill_(angle_deg)

    def reset(self) -> None:
        """Return to identity, with unit zoom, no translation and no rotation."""
        with torch.no_grad():
            self.scale_factor.fill_(1.0)
            self.shift.zero_()
            self.angle.zero_()
