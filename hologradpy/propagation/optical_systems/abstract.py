import torch
import torch.nn as nn

from ..virtual_slms.abstract import VirtualSLM
from ...hardware.utils import CameraData


class SLMCameraModel(nn.Sequential):
    virtual_slm: VirtualSLM
    camera_data: CameraData

    def forward(self, phase: torch.Tensor | None = None) -> torch.Tensor:
        return super().forward(phase)
