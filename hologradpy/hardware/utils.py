from __future__ import annotations
from dataclasses import dataclass
import pickle

from numpy.typing import NDArray

from slmsuite.hardware.slms.slm import SLM
from slmsuite.hardware.cameras.camera import Camera

@dataclass(frozen=True, unsafe_hash=True)
class SLMData:
    name: str
    shape: tuple[int, int]
    bitdepth: int
    bitresolution: int
    pitch_um: tuple[float, float]
    pitch: tuple[float, float]
    settle_time_s: float
    wav_um: float
    wav_design_um: float
    phase_scaling: float

    @classmethod
    def from_slm(cls: type[SLMData], slm: SLM) -> SLMData:
        return cls(
            name=slm.name,
            shape=slm.shape,
            bitdepth=slm.bitdepth,
            bitresolution=slm.bitresolution,
            pitch_um=slm.pitch_um,
            pitch=slm.pitch,
            settle_time_s=slm.settle_time_s,
            wav_um=slm.wav_um,
            wav_design_um=slm.wav_design_um,
            phase_scaling=slm.phase_scaling,
        )

    def save(self, filename: str):
        with open(filename, "wb") as file:
            pickle.dump(self, file)

    @staticmethod
    def load(filename: str) -> SLMData:
        with open(filename, "rb") as file:
            slm_data: SLMData = pickle.load(file)
        return slm_data


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