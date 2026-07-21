"""The base class for 2D geometric-transform value objects.

One convention throughout: points are ``(x, y)``. A transform maps source-plane points
to destination-plane points and is represented by a 3x3 homogeneous matrix. Value
objects are immutable, so operations return new instances.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray


class GeometricTransform(ABC):
    """A 2D coordinate transform from a source plane to a destination plane.

    Subclasses constrain the degrees of freedom (partial affine, affine, and later
    perspective) and supply a type-specific :meth:`fit`; everything else (applying to
    points, inverting, composing) is shared and works on the 3x3 matrix.
    """

    def __init__(self, matrix: NDArray) -> None:
        matrix = np.asarray(matrix, dtype=np.float64)
        if matrix.shape == (2, 3):
            matrix = np.vstack([matrix, [0.0, 0.0, 1.0]])
        if matrix.shape != (3, 3):
            raise ValueError(
                f"Expected a (3, 3) or (2, 3) matrix, got {matrix.shape}."
            )
        self._matrix = matrix

    @property
    def matrix(self) -> NDArray:
        """The 3x3 homogeneous transform matrix (a read-only copy)."""
        return self._matrix.copy()

    # TODO: Do we really need this?
    @property
    @abstractmethod
    def degrees_of_freedom(self) -> int:
        """Number of free parameters the transform type carries."""

    @classmethod
    @abstractmethod
    def fit(cls, source, destination) -> GeometricTransform:
        """Estimate the transform from ``source -> destination`` point pairs.

        ``source`` and ``destination`` are ``(N, 2)`` arrays of ``(x, y)`` points.
        """

    @classmethod
    def from_matrix(cls, matrix) -> GeometricTransform:
        """Wrap an existing 3x3 (or 2x3) matrix as this transform type."""
        return cls(matrix)

    def transform_points(self, points) -> NDArray:
        """Map ``(N, 2)`` source ``(x, y)`` points to destination ``(x, y)`` points."""
        points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
        homogeneous = np.hstack([points, np.ones((points.shape[0], 1))])
        mapped = homogeneous @ self._matrix.T
        return mapped[:, :2] / mapped[:, 2:3]

    def inverse(self) -> GeometricTransform:
        """The inverse transform (destination -> source), of the same type."""
        return type(self).from_matrix(np.linalg.inv(self._matrix))

    def compose(self, other: GeometricTransform) -> GeometricTransform:
        """``self`` after ``other``: apply ``other`` first, then ``self``.

        The result carries the more general of the two transform types.
        """
        result_type = _more_general_type(type(self), type(other))
        return result_type.from_matrix(self._matrix @ other.matrix)

    def reprojection_error(
        self, source, destination
    ) -> tuple[NDArray, float]:
        """Residual vectors ``mapped - destination`` and their RMS length."""
        errors = self.transform_points(source) - np.asarray(
            destination, dtype=np.float64
        ).reshape(-1, 2)
        rms = float(np.sqrt(np.mean(np.sum(errors**2, axis=1))))
        return errors, rms

    def as_matrix(self, homogeneous: bool = True) -> NDArray:
        """The transform matrix: 3x3 homogeneous, or the 2x3 top rows."""
        return self.matrix if homogeneous else self._matrix[:2, :].copy()

    def to_torch(self, device=None, dtype=None):
        """The 3x3 matrix as a torch tensor (torch imported lazily)."""
        import torch

        return torch.as_tensor(self._matrix, device=device, dtype=dtype)

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, GeometricTransform)
            and type(self) is type(other)
            and np.array_equal(self._matrix, other._matrix)
        )

    def __repr__(self) -> str:
        return f"{type(self).__name__}(matrix={self._matrix.tolist()})"


def _more_general_type(
    first: type[GeometricTransform], second: type[GeometricTransform]
) -> type[GeometricTransform]:
    """The more general of two related transform types (the superclass in the
    partial-affine < affine < perspective chain)."""
    if issubclass(first, second):
        return second
    if issubclass(second, first):
        return first
    raise TypeError(
        f"Cannot combine unrelated transform types {first.__name__} and "
        f"{second.__name__}."
    )
