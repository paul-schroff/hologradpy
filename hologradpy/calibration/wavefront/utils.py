import numpy as np
from numpy.typing import NDArray

from scipy.spatial import Delaunay

from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import lsqr, spsolve

from scipy.ndimage import distance_transform_edt

def wrap(x: NDArray[np.float_]) -> NDArray[np.float_]:
    return (x + np.pi) % (2 * np.pi) - np.pi

def unwrap_nonuniform(
    x: NDArray[np.float_], 
    y: NDArray[np.float_], 
    phase: NDArray[np.float_], 
) -> NDArray[np.float_]:
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

def inpaint(
    image: NDArray[np.float_], covered: NDArray[np.bool_]
) -> NDArray[np.float_]:
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


