"""The eight rotations and flips of a pixel array, and their pixel-space affines.

:func:`dihedral_array_transform` builds the function that applies one of them to an
array. :func:`dihedral_affine_matrix` finds the ``(2, 3)`` affine of that function. The
affine maps the position of a pixel in the input array to its position in the
reoriented array. Points are ``(x, y)``, as everywhere in :mod:`hologradpy.geometry`.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import reduce

import numpy as np
from array_api_compat import array_namespace
from numpy.typing import NDArray


def dihedral_array_transform(
    rot: str | int = "0", fliplr: bool = False, flipud: bool = False
) -> Callable[[NDArray], NDArray]:
    """The function that rotates a pixel array in 90 degree steps and then flips it.

    The rotation is applied before the flips, as in slmsuite. The function therefore
    matches the ``transform`` that a camera applies to its raw frames. It follows the
    array API, so a NumPy array comes back as NumPy and a torch tensor as torch on the
    same device.

    Args:
        rot: The rotation, as ``"90"``, ``"180"`` or ``"270"`` degrees or as the
            :func:`numpy.rot90` count ``1``, ``2`` or ``3``. Any other value leaves the
            array unrotated.
        fliplr: Mirror the array left-right.
        flipud: Mirror the array up-down.

    Returns:
        Callable[[NDArray], NDArray]: The function mapping an array to the reoriented
        array.
    """
    transforms = []

    if fliplr:
        transforms.append(lambda img: array_namespace(img).fliplr(img))
    if flipud:
        transforms.append(lambda img: array_namespace(img).flipud(img))

    if rot == "90" or rot == 1:
        transforms.append(lambda img: array_namespace(img).rot90(img, 1))
    elif rot == "180" or rot == 2:
        transforms.append(lambda img: array_namespace(img).rot90(img, 2))
    elif rot == "270" or rot == 3:
        transforms.append(lambda img: array_namespace(img).rot90(img, 3))

    return reduce(lambda f, g: lambda x: f(g(x)), transforms, lambda x: x)


def dihedral_affine_matrix(
    array_transform: Callable[[NDArray], NDArray], shape: tuple[int, int]
) -> NDArray:
    """The ``(2, 3)`` pixel-space affine of a rotation and flip of a pixel array.

    The matrix maps the ``(x, y, 1)`` of a pixel in an input array of ``shape`` to the
    ``(x, y)`` of the same pixel in the output of ``array_transform``. It is found by
    applying ``array_transform`` to arrays of the row and column indices and reading
    where three corners of the input land. Three corners determine the affine of each
    of the eight rotations and flips.

    Args:
        array_transform: A rotation in 90 degree steps followed by flips.
            :func:`dihedral_array_transform` builds such a function, and a camera
            applies one as its ``transform``.
        shape: The ``(height, width)`` of the input array, which sets the translation.

    Returns:
        NDArray: The ``(2, 3)`` matrix.
    """
    height, width = int(shape[0]), int(shape[1])
    rows = np.broadcast_to(np.arange(height)[:, None], (height, width))
    columns = np.broadcast_to(np.arange(width)[None, :], (height, width))
    source_rows = np.asarray(array_transform(rows))
    source_columns = np.asarray(array_transform(columns))
    out_h, out_w = source_rows.shape

    # Output corner (i, j) came from input (source_rows[i, j], source_columns[i, j]).
    # fit input (x=col, y=row) -> output (x'=j, y'=i) at three corners.
    corners = [(0, 0), (0, out_w - 1), (out_h - 1, 0)]
    source = np.array(
        [[source_columns[i, j], source_rows[i, j], 1.0] for i, j in corners],
        dtype=np.float64,
    )
    destination = np.array([[j, i] for i, j in corners], dtype=np.float64)
    return np.linalg.solve(source, destination).T
