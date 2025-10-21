import torch
import torch.nn as nn

from ..virtual_slms.abstract import VirtualSLM


class SLMCameraModel(nn.Sequential):
    virtual_slm: VirtualSLM

    def forward(self, phase: torch.Tensor | None = None) -> torch.Tensor:
        return super().forward(phase)
