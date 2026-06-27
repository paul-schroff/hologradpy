from __future__ import annotations

import torch
from numpy.typing import NDArray

from slmsuite.hardware.slms.slm import SLM

from ..propagation.complex_amplitude import FieldGeometry
from ..propagation.virtual_slms import VirtualSLM
from ..propagation.utils.fourier_utils import get_spatial_grid


class SimulatedSLMTorch(SLM):
    def __init__(
        self,
        input_geometry: FieldGeometry,
        bitdepth: int = 8,
        name: str = "SimulatedSLM",
        wav_design_um: float | None = None,
        settle_time_s: float = 0.3,
    ) -> None:
        if input_geometry.wavelength.ndim != 0:
            raise ValueError("Only single-wavelength is supported.")

        super().__init__(
            resolution=input_geometry.resolution[
                ::-1
            ],  # Convert to (width, height) for slmsuite
            bitdepth=bitdepth,
            name=name,
            wav_um=input_geometry.wavelength.tolist() * 1e6,
            wav_design_um=wav_design_um,
            pitch_um=(input_geometry.pixel_size * 1e6).tolist(),
            settle_time_s=settle_time_s,
        )

        self.input_geometry = input_geometry

        self.virtual_slm: VirtualSLM = VirtualSLM.from_slm(
            slm=self,
            init_phase=None,
        )

    def _set_phase_hw(self, grayscales: NDArray | torch.Tensor) -> None:
        self.virtual_slm.set_phase(grayscales / self.bitresolution * 2 * torch.pi)

    def close(self) -> None:
        pass

    def get_spatial_grid(
        self, device: torch.device = torch.device("cpu")
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return get_spatial_grid(
            self.input_geometry.resolution,
            tuple(self.input_geometry.pixel_size.tolist()),
            device=device,
        )
