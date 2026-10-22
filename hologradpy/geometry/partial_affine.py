"""Partial-affine (similarity) geometric transforms."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from .affine import AffineTransform
from .matrices import homogeneous_matrix, rotation_matrix_from_angle


class PartialAffineTransform(AffineTransform):
    """A 4-DOF partial-affine (similarity) transform, a uniform scale, a rotation and a
    translation, with no shear or mirror.

    :meth:`fit` runs ``cv2.estimateAffinePartial2D`` with its defaults, a RANSAC fit
    with a threshold of 3 destination units followed by a Levenberg-Marquardt
    refinement, so it leaves out the pairs it does not describe. A camera mapping fits
    one from camera to model pixels (:attr:`~.CameraMapping.partial_affine`). The field
    warp and the Fourier lenses hold its inverse, in the physical frame
    (:meth:`~hologradpy.optics.modules.learnable_partial_affine.LearnablePartialAffine.set`).
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

    def components(
        self, center: tuple[float, float] = (0.0, 0.0)
    ) -> tuple[float, float, tuple[float, float]]:
        """The ``(scale, angle_deg, shift)`` that :meth:`from_components` builds this
        transform from.

        The centre and the shift are in the frame of the points the transform maps.
        For a similarity fitted by the camera mappers, the points are ``(column, row)``
        pixels, and the angle turns +column towards +row. The parameters of a model come
        from the inverse of such a similarity
        (:meth:`~hologradpy.optics.modules.learnable_partial_affine.LearnablePartialAffine.set`).

        Args:
            center: The point that the scale and the rotation keep fixed.

        Returns:
            tuple[float, float, tuple[float, float]]: The uniform scale, the angle in
            degrees and the shift.
        """
        center = np.asarray(center, dtype=np.float64)
        shift = self.translation - (center - self.linear @ center)
        return self.scale, self.rotation_degrees, (float(shift[0]), float(shift[1]))
