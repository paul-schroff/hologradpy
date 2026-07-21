from __future__ import annotations

import torch

from .abstract import PhaseRetrieverBase

from ...optics.systems import SLMFourierLensModel

from ...profiles.phase import linear_phase


class LinearSuperpositionPhaseRetriever(PhaseRetrieverBase):
    def __init__(
        self,
        slm_camera_model: SLMFourierLensModel,
        target_positions: torch.Tensor,
        target_intensities: torch.Tensor | None = None,
        target_phases: torch.Tensor | None = None,
    ) -> None:
        super().__init__(slm_camera_model)

        self.target_positions: torch.Tensor = target_positions
        self.number_of_positions: int = target_positions.shape[0]

        if target_intensities is None:
            target_intensities = torch.ones_like(target_positions[:, 0])
        self.target_intensities: torch.Tensor = target_intensities

        if target_phases is None:
            target_phases = torch.zeros_like(target_positions[:, 0])
        self.target_phases: torch.Tensor = target_phases

    def retrieve_phase(self: LinearSuperpositionPhaseRetriever) -> torch.Tensor:
        geometry = self.slm_camera_model.input_geometry
        complex_dtype = (
            torch.complex128
            if geometry.wavelength.dtype == torch.float64
            else torch.complex64
        )
        grid_x, grid_y = geometry.get_spatial_grid()
        # Single-wavelength retriever: collapse the wavenumber to a scalar so it
        # broadcasts against the grid regardless of whether the geometry stores a
        # 0-dim or shape-(1,) wavelength.
        wavenumber = geometry.wavenumber.reshape(())

        field_superposition = torch.zeros(
            *geometry.resolution,
            dtype=complex_dtype,
            device=self.device,
        )

        for i in range(self.number_of_positions):
            blazed_grating = linear_phase(
                grid_x,
                grid_y,
                self.target_positions[i, 0],
                self.target_positions[i, 1],
                wavenumber=wavenumber,
                focal_length=self.slm_camera_model.fourier_lens.focal_length,
            )

            field_superposition += self.target_intensities[i] ** 2 * torch.exp(
                1j * (blazed_grating + self.target_phases[i])
            )

        return field_superposition.angle()
