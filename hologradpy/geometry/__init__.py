"""Backend-agnostic 2D geometric-transform value objects.

One convention throughout: points are ``(x, y)`` and a transform maps source-plane
points to destination-plane points via a 3x3 homogeneous matrix. The hierarchy is a
chain of increasing generality:

    PartialAffineTransform (4 DOF)  <  AffineTransform (6 DOF)

The value objects hold NumPy matrices. The matrix builders in :mod:`.matrices` take
NumPy or torch arrays, so a torch module such as the field warp builds its matrices in
the same convention.
"""

from .abstract import GeometricTransform
from .affine import AffineTransform
from .matrices import homogeneous_matrix, rotation_matrix_from_angle
from .partial_affine import (
    PartialAffineTransform,
    SupportsPartialAffine,
    recalibrated_partial_affine,
)

__all__ = [
    "GeometricTransform",
    "AffineTransform",
    "PartialAffineTransform",
    "SupportsPartialAffine",
    "recalibrated_partial_affine",
    "homogeneous_matrix",
    "rotation_matrix_from_angle",
]
