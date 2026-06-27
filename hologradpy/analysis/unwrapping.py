"""Phase-unwrapping utilities."""

import numpy as np
from numpy.typing import NDArray

from scipy.spatial import Delaunay
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import lsqr


def wrap(x: NDArray[np.float_]) -> NDArray[np.float_]:
    """Wrap phase values into the interval ``(-pi, pi]``.

    Args:
        x (NDArray): Phase values.

    Returns:
        NDArray: Wrapped phase values.
    """
    return (x + np.pi) % (2 * np.pi) - np.pi


def unwrap_2d_mask(
    phase: NDArray[np.float_], mask: NDArray[np.float_], **kwargs
) -> NDArray[np.float_]:
    """Unwrap a phase image within a region of interest defined by a mask.

    The phase is unwrapped row by row and then column by column with
    ``np.unwrap``, considering only the pixels inside ``mask``. Pixels outside
    the mask are set to zero.

    Args:
        phase (NDArray): Wrapped phase image.
        mask (NDArray): Boolean mask defining the region of interest.
        **kwargs: Keyword arguments forwarded to ``np.unwrap``.

    Returns:
        NDArray: Unwrapped phase image, zero outside the mask.
    """
    if kwargs is None:
        kwargs = {"period": 2 * np.pi}
    unwarpped_phase_1D = np.zeros_like(phase)
    for i in range(mask.shape[0]):
        unwarpped_phase_1D[i, mask[i, :]] = np.unwrap(phase[i, mask[i, :]], **kwargs)

    unwrapped_phase = np.zeros_like(phase)
    for i in range(mask.shape[1]):
        unwrapped_phase[mask[:, i], i] = (
            np.unwrap(unwarpped_phase_1D[mask[:, i], i], **kwargs)
        )
    unwrapped_phase[~mask] = 0
    return unwrapped_phase


def unwrap_nonuniform(
    x: NDArray[np.float_],
    y: NDArray[np.float_],
    phase: NDArray[np.float_],
) -> NDArray[np.float_]:
    """Unwrap phase sampled at non-uniform (scattered) points.

    Builds a Delaunay triangulation of the ``(x, y)`` points and solves a
    weighted least-squares problem over the edges, where each edge constrains
    the unwrapped phase difference to the wrapped measured difference.

    Args:
        x (NDArray): X coordinates of the sample points.
        y (NDArray): Y coordinates of the sample points.
        phase (NDArray): Wrapped phase at each sample point.

    Returns:
        NDArray: Unwrapped phase at each sample point.
    """
    points = np.column_stack([x, y])
    number_of_points = len(phase)

    # Build Delaunay triangulation to find neighbours
    tessellation = Delaunay(points)
    edges = set()
    for simplex in tessellation.simplices:
        for a, b in [(0, 1), (1, 2), (0, 2)]:
            i, j = simplex[a], simplex[b]
            edges.add((min(i, j), max(i, j)))
    edges = list(edges)

    number_of_edges = len(edges)

    # Build system
    A = lil_matrix((number_of_edges, number_of_points))
    b = np.zeros(number_of_edges)

    for k, (i, j) in enumerate(edges):
        distance = np.sqrt((x[i] - x[j]) ** 2 + (y[i] - y[j]) ** 2)

        weight = 1.0 / distance

        A[k, j] = weight
        A[k, i] = -weight
        b[k] = weight * wrap(phase[j] - phase[i])

    # Fix one point (removes gauge freedom)
    A[0, :] = 0
    A[0, 0] = 1
    b[0] = phase[0]

    phi, *_ = lsqr(A.tocsr(), b)
    return phi
