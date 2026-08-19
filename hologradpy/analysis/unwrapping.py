"""Phase-unwrapping utilities."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from scipy.fft import dctn, idctn
from scipy.spatial import Delaunay
from scipy.sparse import coo_matrix, csr_matrix, lil_matrix
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import LinearOperator, cg, lsqr, splu


def wrap(x: NDArray[np.float64]) -> NDArray[np.float64]:
    """Wrap phase values into the interval ``[-pi, pi)``.

    Note the half-open end: an input of exactly pi comes back as ``-pi``.
    :func:`numpy.angle` uses the other half-open interval, ``(-pi, pi]``. The two agree
    everywhere else.

    Args:
        x: Phase values.

    Returns:
        NDArray: Wrapped phase values.
    """
    return (x + np.pi) % (2 * np.pi) - np.pi


def _neighbour_pairs(
    mask: NDArray[np.bool_],
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    """Four-connected neighbour pairs inside a mask.

    Args:
        mask: Boolean mask defining the region of interest.

    Returns:
        tuple[NDArray, NDArray]: Two index arrays naming the ends of each pair, indexing
        the masked pixels in the order ``phase[mask]`` gives them.
    """
    index = np.full(mask.shape, -1, dtype=np.int64)
    index[mask] = np.arange(int(mask.sum()))

    starts, ends = [], []
    for axis in (0, 1):
        # Both ends inside the mask. The roll wraps around, so drop the far edge.
        pair = mask & np.roll(mask, -1, axis=axis)
        if axis == 0:
            pair[-1, :] = False
        else:
            pair[:, -1] = False
        starts.append(index[pair])
        ends.append(index[np.roll(pair, 1, axis=axis)])

    return np.concatenate(starts), np.concatenate(ends)


def _normal_equations(
    phase: NDArray[np.float64], mask: NDArray[np.bool_]
) -> tuple[csr_matrix, NDArray[np.float64], NDArray[np.float64]]:
    """The least-squares system both unwrappers solve.

    One row per four-connected neighbour pair, reading -1 at one end and +1 at the
    other, so the row measures the difference across that pair. The normal equations of
    that system have the mask's graph laplacian as their matrix, singular by one
    constant per connected region, which is the 2 pi gauge freedom of the problem.

    Args:
        phase: Wrapped phase image.
        mask: Boolean mask defining the region of interest.

    Returns:
        tuple: The laplacian, the right-hand side, and the wrapped phase of the masked
        pixels in the order ``phase[mask]`` gives them.
    """
    number_of_pixels = int(mask.sum())
    wrapped = phase[mask].astype(float)
    starts, ends = _neighbour_pairs(mask)

    number_of_pairs = len(starts)
    differences = wrap(wrapped[ends] - wrapped[starts])
    gradient = coo_matrix(
        (
            np.concatenate([-np.ones(number_of_pairs), np.ones(number_of_pairs)]),
            (
                np.tile(np.arange(number_of_pairs), 2),
                np.concatenate([starts, ends]),
            ),
        ),
        shape=(number_of_pairs, number_of_pixels),
    ).tocsr()

    return (gradient.T @ gradient).tocsr(), gradient.T @ differences, wrapped


def _region_anchors(laplacian: csr_matrix) -> tuple[int, NDArray, NDArray]:
    """One pixel per connected region, to pin that region's gauge against."""
    number_of_regions, labels = connected_components(laplacian, directed=False)
    anchors = np.array(
        [np.flatnonzero(labels == region)[0] for region in range(number_of_regions)]
    )
    return number_of_regions, labels, anchors


def _anchored(
    solution: NDArray[np.float64],
    wrapped: NDArray[np.float64],
    labels: NDArray,
    anchors: NDArray,
) -> NDArray[np.float64]:
    """Set each region's free constant so it agrees with the measured phase there."""
    for region, anchor in enumerate(anchors):
        pixels = labels == region
        solution[pixels] += wrapped[anchor] - solution[anchor]
    return solution


def unwrap_2d_laplace(
    phase: NDArray[np.float64], mask: NDArray[np.bool_]
) -> NDArray[np.float64]:
    """Unwrap a phase image within a region of interest, by a direct sparse solve.

    Least-squares unwrapping over the four-connected pixels inside ``mask``. Exact but
    slow.

    Args:
        phase: Wrapped phase image.
        mask: Boolean mask defining the region of interest.

    Returns:
        NDArray: Unwrapped phase image, zero outside the mask.
    """
    mask = np.asarray(mask, dtype=bool)
    number_of_pixels = int(mask.sum())

    unwrapped_phase = np.zeros_like(phase)
    if number_of_pixels == 0:
        return unwrapped_phase

    laplacian, right_hand_side, wrapped = _normal_equations(phase, mask)
    _, labels, anchors = _region_anchors(laplacian)

    free = np.ones(number_of_pixels, dtype=bool)
    free[anchors] = False

    solution = np.zeros(number_of_pixels)
    if free.any():
        # Pinning one pixel per region removes the singularity and leaves a system a
        # direct solve handles.
        reduced = laplacian[free][:, free].tocsc()
        solution[free] = splu(reduced).solve(right_hand_side[free])

    unwrapped_phase[mask] = _anchored(solution, wrapped, labels, anchors)
    return unwrapped_phase


def _poisson_preconditioner(mask: NDArray[np.bool_]):
    """Approximate inverse of the masked laplacian, via a DCT Poisson solve."""
    rows, columns = mask.shape
    eigenvalues = (
        2 * (np.cos(np.pi * np.arange(rows) / rows) - 1)[:, None]
        + 2 * (np.cos(np.pi * np.arange(columns) / columns) - 1)[None, :]
    )
    # The constant mode is the gauge freedom and has no inverse. Held at one to keep the
    # divide finite, then zeroed, which projects it out.
    eigenvalues[0, 0] = 1.0

    def apply(vector: NDArray[np.float64]) -> NDArray[np.float64]:
        grid = np.zeros(mask.shape)
        grid[mask] = vector
        transformed = dctn(grid, type=2, norm="ortho") / -eigenvalues
        transformed[0, 0] = 0.0
        return idctn(transformed, type=2, norm="ortho")[mask]

    return apply


def unwrap_2d_poisson(
    phase: NDArray[np.float64],
    mask: NDArray[np.bool_],
    tolerance: float = 1e-10,
    max_iterations: int = 500,
) -> NDArray[np.float64]:
    """Unwrap a phase image in a region of interest, by preconditioned Poisson solve.

    This is the method of Ghiglia and Romero, "Robust two-dimensional weighted and
    unweighted phase unwrapping that uses fast transforms and iterative methods",
    J. Opt. Soc. Am. A 11, 107 (1994), https://doi.org/10.1364/JOSAA.11.000107. Much
    faster than :func:`unwrap_2d_laplace` for large images.

    Args:
        phase: Wrapped phase image.
        mask: Boolean mask defining the region of interest.
        tolerance: Relative residual conjugate gradients stops at. Defaults to 1e-10.
        max_iterations: Iteration cap. Defaults to 500.

    Returns:
        NDArray: Unwrapped phase image, zero outside the mask.

    Raises:
        RuntimeError: when conjugate gradients does not converge.
    """
    mask = np.asarray(mask, dtype=bool)
    number_of_pixels = int(mask.sum())

    unwrapped_phase = np.zeros_like(phase)
    if number_of_pixels == 0:
        return unwrapped_phase

    laplacian, right_hand_side, wrapped = _normal_equations(phase, mask)
    _, labels, anchors = _region_anchors(laplacian)

    preconditioner = LinearOperator(
        (number_of_pixels, number_of_pixels), matvec=_poisson_preconditioner(mask)
    )
    solution, info = cg(
        laplacian,
        right_hand_side,
        M=preconditioner,
        rtol=tolerance,
        maxiter=max_iterations,
    )
    if info != 0:
        raise RuntimeError(
            "Preconditioned conjugate gradients failed to unwrap the phase "
            f"(scipy returned {info} after at most {max_iterations} iterations). "
            "unwrap_2d_laplace solves the same system directly."
        )

    unwrapped_phase[mask] = _anchored(solution, wrapped, labels, anchors)
    return unwrapped_phase


def unwrap_nonuniform(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    phase: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Unwrap phase sampled at non-uniform (scattered) points.

    Builds a Delaunay triangulation of the ``(x, y)`` points and solves a
    weighted least-squares problem over the edges, where each edge constrains
    the unwrapped phase difference to the wrapped measured difference.

    Args:
        x: X coordinates of the sample points.
        y: Y coordinates of the sample points.
        phase: Wrapped phase at each sample point.

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
