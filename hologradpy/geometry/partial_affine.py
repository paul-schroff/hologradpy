"""Partial-affine (similarity) geometric transforms."""

from __future__ import annotations

from typing import Protocol

import numpy as np
from numpy.typing import ArrayLike

from .affine import AffineTransform, _matrix_from_linear, _rotation_matrix


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
        linear = scale * _rotation_matrix(angle_deg)
        return cls(_matrix_from_linear(linear, shift, center))

    @property
    def scale(self) -> float:
        """The uniform scale factor."""
        return float(np.sqrt(abs(np.linalg.det(self.linear))))

    @property
    def angle_degrees(self) -> float:
        """The rotation angle in degrees."""
        return self.rotation_degrees


class SupportsPartialAffine(Protocol):
    """An object that can absorb a :class:`PartialAffineTransform` as a residual on its
    own learnable ``(scale, angle, shift)``, so a coarse fitted transform can warm-start
    a differentiable registration (for example the field warp or a Fourier lens).
    """

    def apply_partial_affine(self, transform: PartialAffineTransform) -> None: ...


def recalibrated_partial_affine(
    scale: float,
    angle_deg: float,
    shift_xy: tuple[float, float],
    residual: PartialAffineTransform,
    center_xy: tuple[float, float],
) -> tuple[float, float, tuple[float, float]]:
    """Compose a fitted residual similarity onto a current ``(scale, angle, shift)``.

    Used to warm-start a differentiable focal-plane registration from a coarse camera
    mapping: ``residual`` is the camera -> model similarity (as fit by the camera
    mappers), and its inverse (the model -> camera map) composes on the left of the
    current focal-plane similarity. All quantities are in (x, y) output pixels about
    ``center_xy``. Returns the new uniform ``scale``, ``angle_deg`` and ``shift_xy``.
    """
    current = PartialAffineTransform.from_components(
        scale=scale, angle_deg=angle_deg, shift=shift_xy, center=center_xy
    )
    updated: PartialAffineTransform = PartialAffineTransform.from_matrix(
        residual.inverse().matrix @ current.matrix
    )
    # Recover the shift about ``center_xy`` (inverse of from_components' translation).
    center = np.asarray(center_xy, dtype=np.float64)
    shift = updated.translation - (center - updated.linear @ center)
    return updated.scale, updated.angle_degrees, (float(shift[0]), float(shift[1]))
