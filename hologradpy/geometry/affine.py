"""Affine geometric transforms."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .abstract import GeometricTransform


class AffineTransform(GeometricTransform):
    """A 6-DOF affine transform: a 2x2 linear map plus a translation.

    The homogeneous matrix has a ``[0, 0, 1]`` bottom row, so the linear part
    (rotation, scale, shear and an optional mirror) is meaningful and can be
    decomposed. Fitted from point pairs with ``cv2.estimateAffine2D``.
    """

    @property
    def degrees_of_freedom(self) -> int:
        return 6

    @classmethod
    def fit(cls, source, destination, *, robust: bool = True) -> AffineTransform:
        """Estimate an affine transform from ``source -> destination`` point pairs.

        ``robust=True`` (the default) uses ``cv2.estimateAffine2D`` (RANSAC).
        ``robust=False`` uses a plain least-squares fit
        (``[source | 1] -> destination``).
        """
        source = np.asarray(source, dtype=np.float64).reshape(-1, 2)
        destination = np.asarray(destination, dtype=np.float64).reshape(-1, 2)
        if robust:
            import cv2

            matrix, _ = cv2.estimateAffine2D(
                source.reshape(-1, 1, 2), destination.reshape(-1, 1, 2)
            )
            if matrix is None:
                raise ValueError("estimateAffine2D failed to fit an affine transform.")
            return cls(matrix)
        design = np.hstack([source, np.ones((len(source), 1))])
        solution, *_ = np.linalg.lstsq(design, destination, rcond=None)
        return cls(solution.T)

    @classmethod
    def from_components(
        cls,
        *,
        scale: float | tuple[float, float] = 1.0,
        angle_deg: float = 0.0,
        shift: tuple[float, float] = (0.0, 0.0),
        shear: float = 0.0,
        mirror: bool = False,
        center: tuple[float, float] = (0.0, 0.0),
    ) -> AffineTransform:
        """Build an affine transform from human-readable components.

        The linear part is ``R(angle) @ shear @ diag(scale) @ mirror`` and the
        translation keeps ``center`` fixed before adding ``shift``. ``scale`` is a
        scalar (isotropic) or ``(scale_x, scale_y)``.
        """
        scale_x, scale_y = (scale, scale) if np.isscalar(scale) else scale
        rotation = _rotation_matrix(angle_deg)
        shear_matrix = np.array([[1.0, shear], [0.0, 1.0]])
        scale_matrix = np.diag([scale_x, scale_y])
        mirror_matrix = np.diag([1.0, -1.0]) if mirror else np.eye(2)
        linear = rotation @ shear_matrix @ scale_matrix @ mirror_matrix
        return cls(_matrix_from_linear(linear, shift, center))

    @property
    def linear(self) -> NDArray:
        """The 2x2 linear part of the transform."""
        return self._matrix[:2, :2].copy()

    @property
    def translation(self) -> NDArray:
        """The ``(x, y)`` translation part."""
        return self._matrix[:2, 2].copy()

    @property
    def is_mirrored(self) -> bool:
        """True if the transform flips handedness (negative determinant)."""
        return bool(np.linalg.det(self.linear) < 0)

    @property
    def rotation_matrix(self) -> NDArray:
        """The orthonormal rotation-and-mirror part of the linear map (``U @ Vt``
        from its SVD, with the mirror preserved)."""
        left, _, right = np.linalg.svd(self.linear)
        return left @ right

    @property
    def rotation_degrees(self) -> float:
        """Rotation of the destination axes relative to the source, in degrees, with
        any reflection factored out."""
        left, _, right = np.linalg.svd(self.linear)
        if np.linalg.det(left @ right) < 0:
            left[:, -1] *= -1
        rotation = left @ right
        return float(np.degrees(np.arctan2(rotation[1, 0], rotation[0, 0])))

    @property
    def scales(self) -> tuple[float, float]:
        """The scale factors (singular values of the linear part, major then minor)."""
        singular_values = np.linalg.svd(self.linear, compute_uv=False)
        return (float(singular_values[0]), float(singular_values[1]))


# TODO: Move these to a utils.py or make static methods?
def _rotation_matrix(angle_deg: float) -> NDArray:
    theta = np.radians(angle_deg)
    cos, sin = np.cos(theta), np.sin(theta)
    return np.array([[cos, -sin], [sin, cos]])


def _matrix_from_linear(
    linear: NDArray, shift: tuple[float, float], center: tuple[float, float]
) -> NDArray:
    center = np.asarray(center, dtype=np.float64)
    translation = np.asarray(shift, dtype=np.float64) + center - linear @ center
    matrix = np.eye(3)
    matrix[:2, :2] = linear
    matrix[:2, 2] = translation
    return matrix
