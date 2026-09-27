"""Rotation and homogeneous matrices for the 2D transforms.

The functions follow the array API, so a NumPy array gives a NumPy matrix and a torch
tensor gives a torch matrix on the same device, differentiable in its inputs. Leading
``*batch`` axes build a stack of matrices, one per entry. Points are ``(x, y)``, as
everywhere in :mod:`hologradpy.geometry`.
"""

from __future__ import annotations

import math

import numpy as np
import torch
from array_api_compat import array_namespace, device
from jaxtyping import Float


def rotation_matrix_from_angle(
    angle_deg: Float[torch.Tensor, "*batch"] | Float[np.ndarray, "*batch"],
) -> Float[torch.Tensor, "*batch 2 2"] | Float[np.ndarray, "*batch 2 2"]:
    """The 2x2 matrix rotating ``(x, y)`` points by an angle.

    The matrix is ``[[cos, -sin], [sin, cos]]``, so a positive angle turns the x axis
    towards the y axis.

    Args:
        angle_deg: The rotation angle in degrees.

    Returns:
        The rotation matrix for each angle.
    """
    xp = array_namespace(angle_deg)
    angle = angle_deg * (math.pi / 180.0)
    cos, sin = xp.cos(angle), xp.sin(angle)
    first_row = xp.stack([cos, -sin], axis=-1)
    second_row = xp.stack([sin, cos], axis=-1)
    return xp.stack([first_row, second_row], axis=-2)


def homogeneous_matrix(
    linear: Float[torch.Tensor, "*batch 2 2"] | Float[np.ndarray, "*batch 2 2"],
    shift: Float[torch.Tensor, "*batch 2"] | Float[np.ndarray, "*batch 2"],
    center: Float[torch.Tensor, "*batch 2"] | Float[np.ndarray, "*batch 2"],
) -> Float[torch.Tensor, "*batch 3 3"] | Float[np.ndarray, "*batch 3 3"]:
    """The 3x3 homogeneous matrix of a linear map about a centre, followed by a shift.

    A point ``p`` maps to ``linear @ (p - center) + center + shift``, so the centre
    moves by ``shift`` alone.

    Args:
        linear: The 2x2 linear part.
        shift: The ``(x, y)`` shift applied after the linear map.
        center: The ``(x, y)`` point the linear map keeps fixed.

    Returns:
        The homogeneous matrix, with a ``[0, 0, 1]`` bottom row.
    """
    xp = array_namespace(linear, shift, center)
    translation = shift + center - (linear @ center[..., None])[..., 0]
    top_rows = xp.concat([linear, translation[..., None]], axis=-1)
    bottom_row = xp.asarray(
        [0.0, 0.0, 1.0], dtype=top_rows.dtype, device=device(linear)
    )
    bottom_row = xp.broadcast_to(bottom_row, (*top_rows.shape[:-2], 1, 3))
    return xp.concat([top_rows, bottom_row], axis=-2)
