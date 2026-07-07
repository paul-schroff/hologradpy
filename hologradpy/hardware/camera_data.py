from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable
import pickle

import numpy as np
from numpy.typing import NDArray

from slmsuite.hardware.cameras.camera import Camera


def probe_orientation(
    transform_fn: Callable[[NDArray], NDArray], shape: tuple[int, int]
) -> NDArray:
    """Pixel-space affine ``(x, y)_out = M @ [x, y, 1]`` of a discrete image transform 
    (a rot90/flip composition, e.g. ``Camera.transform``).

    Found by applying ``transform_fn`` to row/column index arrays of ``shape`` (height,
    width) and reading where three input corners land. Robust for any of the 8
    dihedral orientations. Returns a ``(2, 3)`` matrix.
    """
    height, width = int(shape[0]), int(shape[1])
    rows = np.broadcast_to(np.arange(height)[:, None], (height, width))
    columns = np.broadcast_to(np.arange(width)[None, :], (height, width))
    source_rows = np.asarray(transform_fn(rows))
    source_columns = np.asarray(transform_fn(columns))
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


@dataclass(frozen=True, unsafe_hash=True)
class CameraData:
    name: str
    shape: tuple[int, int]
    bitdepth: int
    bitresolution: int
    pitch_um: tuple[float, float]
    exposure_s: float
    exposure_bounds_s: tuple[float, float]
    averaging: int
    hdr: bool
    woi: tuple[int, int, int, int]
    default_shape: tuple[int, int]
    orientation: NDArray = field(compare=False, hash=False)

    @classmethod
    def from_camera(cls: type[CameraData], camera: Camera) -> CameraData:
        return cls(
            name=camera.name,
            shape=camera.shape,
            bitdepth=camera.bitdepth,
            bitresolution=camera.bitresolution,
            pitch_um=camera.pitch_um,
            exposure_s=camera.exposure_s,
            exposure_bounds_s=camera.exposure_bounds_s,
            averaging=camera.averaging,
            hdr=camera.hdr,
            woi=camera.woi,
            default_shape=camera.default_shape,
            orientation=probe_orientation(camera.transform, camera.default_shape),
        )

    def save(self, filename: str):
        with open(filename, "wb") as file:
            pickle.dump(self, file)

    @staticmethod
    def load(filename: str) -> CameraData:
        with open(filename, "rb") as file:
            camera_data: CameraData = pickle.load(file)
        return camera_data
