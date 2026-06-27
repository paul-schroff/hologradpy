from __future__ import annotations
from dataclasses import dataclass
import pickle

from slmsuite.hardware.slms.slm import SLM


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
