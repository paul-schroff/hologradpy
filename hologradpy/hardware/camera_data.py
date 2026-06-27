from __future__ import annotations
from dataclasses import dataclass
import pickle

from numpy.typing import NDArray

from slmsuite.hardware.cameras.camera import Camera


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
    transform: NDArray

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
            transform=camera.transform,
        )

    def save(self, filename: str):
        with open(filename, "wb") as file:
            pickle.dump(self, file)

    @staticmethod
    def load(filename: str) -> CameraData:
        with open(filename, "rb") as file:
            camera_data: CameraData = pickle.load(file)
        return camera_data
