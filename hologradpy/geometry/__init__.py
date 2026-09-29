"""Backend-agnostic 2D geometric-transform value objects.

One convention throughout: points are ``(x, y)`` and a transform maps source-plane
points to destination-plane points via a 3x3 homogeneous matrix. The hierarchy is a
chain of increasing generality:

    PartialAffineTransform (4 DOF)  <  AffineTransform (6 DOF)

The value objects hold NumPy matrices. The matrix builders in :mod:`.matrices` take
NumPy or torch arrays, so a torch module such as the field warp builds its matrices in
the same convention. :mod:`.dihedral` rotates and flips pixel arrays to match the
reorientation of a camera frame. It also finds the pixel-space affine of each
reorientation.
"""

from .abstract import GeometricTransform
from .affine import AffineTransform
from .dihedral import dihedral_affine_matrix, dihedral_array_transform
from .matrices import homogeneous_matrix, rotation_matrix_from_angle
from .partial_affine import PartialAffineTransform, inverse_partial_affine_parameters

__all__ = [
    "GeometricTransform",
    "AffineTransform",
    "PartialAffineTransform",
    "inverse_partial_affine_parameters",
    "homogeneous_matrix",
    "rotation_matrix_from_angle",
    "dihedral_array_transform",
    "dihedral_affine_matrix",
]
