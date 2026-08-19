from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve

from scipy.ndimage import distance_transform_edt


def inpaint(
    image: NDArray[np.float64], covered: NDArray[np.bool_]
) -> NDArray[np.float64]:
    missing = ~covered

    number_of_pixels = image.size
    flat_index = np.arange(number_of_pixels).reshape(image.shape)

    result = image.copy().astype(float)

    nearest: NDArray[np.int32]
    _, nearest = distance_transform_edt(missing, return_indices=True)

    result[missing] = image[tuple(nearest)][missing]

    sparse_rows = []
    sparse_cols = []
    sparse_vals = []

    covered_flat = flat_index[covered]

    sparse_rows.append(covered_flat)
    sparse_cols.append(covered_flat)
    sparse_vals.append(np.ones(covered_flat.size))

    for pixel_indices, neighbor_indices, hole_mask in [
        (flat_index[1:, :], flat_index[:-1, :], missing[1:, :]),
        (flat_index[:-1, :], flat_index[1:, :], missing[:-1, :]),
        (flat_index[:, 1:], flat_index[:, :-1], missing[:, 1:]),
        (flat_index[:, :-1], flat_index[:, 1:], missing[:, :-1]),
    ]:
        pixel = pixel_indices[hole_mask]
        neighbor = neighbor_indices[hole_mask]

        sparse_rows.append(pixel)
        sparse_cols.append(neighbor)
        sparse_vals.append(-np.ones(pixel.size))

        sparse_rows.append(pixel)
        sparse_cols.append(pixel)
        sparse_vals.append(np.ones(pixel.size))

    A = csr_matrix(
        (
            np.concatenate(sparse_vals),
            (np.concatenate(sparse_rows), np.concatenate(sparse_cols)),
        ),
        shape=(number_of_pixels, number_of_pixels),
    )
    b = np.zeros(number_of_pixels)
    b[covered_flat] = image[covered].ravel()

    return spsolve(A, b).reshape(image.shape)
