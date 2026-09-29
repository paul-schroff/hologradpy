"""Partial-affine (similarity) geometric transforms."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from .affine import AffineTransform
from .matrices import homogeneous_matrix, rotation_matrix_from_angle


class PartialAffineTransform(AffineTransform):
    """A 4-DOF partial-affine (similarity) transform: uniform scale + rotation +
    translation, with no shear or mirror.

    Fitted with ``cv2.estimateAffinePartial2D``. This is the ``(scale, angle, shift)``
    parameterization the differentiable field warp and the Fourier lenses use.
    """

    @property
    def degrees_of_freedom(self) -> int:
        return 4

    @classmethod
    def fit(cls, source: ArrayLike, destination: ArrayLike) -> PartialAffineTransform:
        import cv2

        source = np.asarray(source, dtype=np.float64).reshape(-1, 1, 2)
        destination = np.asarray(destination, dtype=np.float64).reshape(-1, 1, 2)
        matrix, _ = cv2.estimateAffinePartial2D(source, destination)
        if matrix is None:
            raise ValueError(
                "estimateAffinePartial2D failed to fit a partial-affine transform."
            )
        return cls(matrix)

    @classmethod
    def from_components(
        cls,
        *,
        scale: float = 1.0,
        angle_deg: float = 0.0,
        shift: tuple[float, float] = (0.0, 0.0),
        center: tuple[float, float] = (0.0, 0.0),
    ) -> PartialAffineTransform:
        """Build a similarity transform from a uniform ``scale``, ``angle_deg`` and
        ``shift``, keeping ``center`` fixed before the shift.
        """
        angle = np.asarray(angle_deg, dtype=np.float64)
        linear = scale * rotation_matrix_from_angle(angle)
        return cls(
            homogeneous_matrix(
                linear,
                np.asarray(shift, dtype=np.float64),
                np.asarray(center, dtype=np.float64),
            )
        )

    @property
    def scale(self) -> float:
        """The uniform scale factor."""
        return float(np.sqrt(abs(np.linalg.det(self.linear))))

    @property
    def angle_degrees(self) -> float:
        """The rotation angle in degrees."""
        return self.rotation_degrees


def inverse_partial_affine_parameters(
    transform: PartialAffineTransform, center_xy: tuple[float, float]
) -> tuple[float, float, tuple[float, float]]:
    """The ``(scale, angle_deg, shift_xy)`` of the inverse of a fitted camera -> model
    similarity, the model -> camera map that a focal-plane partial affine applies.

    The scale and the rotation keep ``center_xy`` fixed. All quantities are in
    ``(x, y)`` output pixels.

    Args:
        transform: The camera -> model similarity fitted by the camera mappers.
        center_xy: The point that the scale and the rotation keep fixed.

    Returns:
        tuple[float, float, tuple[float, float]]: The uniform scale, the angle in
        degrees and the shift.
    """
    inverse = PartialAffineTransform.from_matrix(transform.inverse().matrix)
    # The shift about center_xy, inverting the translation of from_components.
    center = np.asarray(center_xy, dtype=np.float64)
    shift = inverse.translation - (center - inverse.linear @ center)
    return inverse.scale, inverse.angle_degrees, (float(shift[0]), float(shift[1]))
