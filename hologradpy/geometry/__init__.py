"""Backend-agnostic 2D geometric-transform value objects.

One convention throughout: points are ``(x, y)`` and a transform maps source-plane
points to destination-plane points via a 3x3 homogeneous matrix. The hierarchy is a
chain of increasing generality (a perspective transform, to be added later, sits
above affine):

    PartialAffineTransform (4 DOF)  <  AffineTransform (6 DOF)
"""

from .abstract import GeometricTransform
from .affine import AffineTransform
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
]
