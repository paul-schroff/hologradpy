import torch.nn as nn

from ..elements import VirtualSLM


class SLMCameraModel(nn.Sequential):
    virtual_slm: VirtualSLM
    # def forward(self, phase: torch.Tensor | None = None) -> torch.Tensor: ...
    # def __call__(self, phase: torch.Tensor | None = None) -> torch.Tensor: ...
