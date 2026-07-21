from abc import ABC, abstractmethod

import torch

from ..phase_retrieval import PhaseRetrieverBase

from ...hardware import Camera, SLM

# TODO: Implement camera feedback base class.
class CameraFeedbackBase(ABC):
    def __init__(
        self,
        phase_retriever: PhaseRetrieverBase,
        camera: Camera, 
        slm: SLM, 
        target: torch.Tensor
    ) -> None:
        self.phase_retriever = phase_retriever
        self.camera = camera
        self.slm = slm
        self._target = target
        self._corrected_target = target
    
    @property
    def target(self) -> torch.Tensor:
        return self._target
    
    def update_target(self, target: torch.Tensor) -> None:
        self._corrected_target = target
        self.phase_retriever.set_target(self._corrected_target)

    @abstractmethod
    def run(self):
        pass
        
