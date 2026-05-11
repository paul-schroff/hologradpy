from __future__ import annotations

import torch

from .abstract import PhaseRetrieverBase

from ..utils import Timer

from ...propagation.optical_systems import SLMFourierLensModel

from ...propagation.utils.optics_utils import linear_phase

class LinearSuperpositionPhaseRetriever(PhaseRetrieverBase):
    def __init__(
        self,
        slm_camera_model: SLMFourierLensModel,
        target_positions: torch.Tensor,
        target_intensities: torch.Tensor | None = None,
        target_phases: torch.Tensor | None = None,
        device: str = "cpu",
    ) -> None:
        super().__init__(slm_camera_model, device)
        
        self.target_positions: torch.Tensor = target_positions
        self.number_of_positions: int = target_positions.shape[0]

        if target_intensities is None:
            self.target_intensities = torch.ones_like(
                target_positions[:, 0]
            )
        self.target_intensities: torch.Tensor = target_intensities

        if target_phases is None:
            self.target_phases = torch.zeros_like(
                target_positions[:, 0]
            )
        self.target_phases: torch.Tensor = target_phases
   

    def retrieve_phase(
        self: LinearSuperpositionPhaseRetriever
    ) -> torch.Tensor:
        field_superposition = torch.zeros(
            *self.slm_camera_model.virtual_slm.resolution_in, 
            dtype=self.slm_camera_model.virtual_slm.dtype_c,
            device=self.device
        )

        for i in range(self.number_of_positions):
            blazed_grating = linear_phase(
                *self.slm_camera_model.virtual_slm.get_spatial_grid_input(), 
                self.target_positions[i, 0], 
                self.target_positions[i, 1],
                wavenumber=self.slm_camera_model.virtual_slm.wavenumber,
                focal_length=self.slm_camera_model.fourier_lens.focal_length,
            )

            field_superposition += (
                self.target_intensities[i] ** 2 
                * torch.exp(1j * (blazed_grating + self.target_phases[i]))
            )

        return field_superposition.angle()