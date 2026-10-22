"""The learnable partial affine of an output plane."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import Parameter

from ...geometry import PartialAffineTransform
from ...grids import pixel_frame_angle, pixel_frame_vector, plane_center


class LearnablePartialAffine(nn.Module):
    """A learnable partial affine that places the focal plane on an output plane,
    applied by the module that owns it.

    It runs from the model to the camera. The focal plane is turned by ``angle`` and
    then zoomed by ``scale_factor`` along the output x and y, both about the centre
    pixel of the output plane (:func:`~hologradpy.grids.plane_center`), and then moved
    by ``shift``. The zeroth order therefore lands ``shift`` output pixels from the
    centre pixel. ``scale_factor`` is the zoom along ``(x, y)``, ``shift`` is in output
    pixels along ``(x, y)``, and ``angle`` is in degrees from +x towards +y. All three
    are physical. A Fourier lens or a field warp holds one, reads its parameters in the
    forward pass and converts them to the pixel frame. The field warp applies a
    negative ``scale_factor`` as a mirror, and the chirp-z and NUFFT lenses take its
    magnitude.

    The camera-aware optical systems seed ``angle`` from ``camera_angle`` and ``shift``
    from ``camera_shift``, converted by :func:`~hologradpy.grids.metres_to_pixels`. They
    calibrate all three from a camera mapping through :meth:`set`.

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
            shift: The starting shift of the focal plane in output pixels along
                ``(x, y)``. The zeroth order lands this far from the centre pixel.
            angle: The starting turn of the focal plane in degrees from +x towards +y.
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
        """Set the parameters to the inverse of a camera -> model similarity, starting
        from identity.

        The similarity carries camera ``(column, row)`` pixels to model
        ``(column, row)`` pixels, as :attr:`~.CameraMapping.partial_affine` fits it.
        Its inverse runs from the model to the camera, the direction of the parameters.
        The inverse is decomposed about the centre pixel of the output plane and
        converted to the physical frame.

        Args:
            transform: The camera -> model similarity.
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
