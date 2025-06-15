import torch
from numpy.typing import NDArray

from . import SLM

from ...torch_modules.elements import VirtualSLM

class SimulatedSLMTorch(SLM):
    def __init__(
            self,
            resolution: tuple[int, int] = (1024, 1280),
            bitdepth: int = 8,
            name: str = "SimulatedSLM",
            wav_um: float = 1,
            wav_design_um: float = None,
            pitch_um: tuple[float, float] = (8, 8),
            settle_time_s: float = 0.3,
            torch_device: torch.device = torch.device("cpu")
        ) -> None:
        super().__init__(
            resolution=resolution,
            bitdepth=bitdepth,
            name=name,
            wav_um=wav_um,
            wav_design_um=wav_design_um,
            pitch_um=pitch_um,
            settle_time_s=settle_time_s
        )

        self.virtual_slm: VirtualSLM = VirtualSLM(
            slm=self,
            init_phase=None,
            device=torch_device,
        )
    
    def _set_phase_hw(self, phase: NDArray | torch.Tensor) -> None:
        self.virtual_slm.set_phase(phase)