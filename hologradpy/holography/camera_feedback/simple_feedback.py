# TODO: Implement simple camera feedback algorithm
from abc import ABC, abstractmethod

import torch

from .abstract import CameraFeedbackBase
from ..phase_retrieval import PhaseRetrieverBase
from ...hardware import Camera, SLM

class SimpleCameraFeedback(CameraFeedbackBase):
    def __init__(
        self,
        phase_retriever: PhaseRetrieverBase,
        camera: Camera,
        slm: SLM,
        target: torch.Tensor,
    ) -> None:
        self.phase_retriever = phase_retriever
        self.camera = camera
        self.slm = slm
        self._target = target
        self._corrected_target = target
    
    def run(self):
        # Capture an image from the camera
        captured_image = self.camera.get_image()

        