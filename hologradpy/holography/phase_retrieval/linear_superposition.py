from __future__ import annotations

import torch

from .abstract import PhaseRetrieverBase
from .recorder import RetrievalRun

from ...optics.systems import SLMFourierLensModel

from ...profiles.phase import linear_phase
from ...utils import ProgressBar


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

    def set_target(
        self,
        target: torch.Tensor,
        signal_region: torch.Tensor | None = None,
    ) -> None:
        """Not available: this retriever has no intensity target to replace.

        It superposes blazed gratings at ``target_positions``, so there is nothing for
        an intensity pattern to set.
        """
        raise NotImplementedError(
            "LinearSuperpositionPhaseRetriever optimizes target_positions, "
            "target_intensities and target_phases rather than an intensity pattern, so "
            "it cannot be retargeted with one. Set those attributes instead."
        )

    # TODO: Liskov is sad.
    def retrieve_phase(
        self,
        number_of_iterations: int = 0,
        *,
        run: RetrievalRun | None = None,
        verbose: bool = True,
        progress_bar: ProgressBar | None = None,
        **_: object,
    ) -> torch.Tensor:
        """Superpose the gratings and set the model with the resulting phase.

        Args:
            number_of_iterations: Unused, and only accepted so this retriever can be
                driven like any other, and so :meth:`~PhaseRetrieverBase.retrieve` works
                on it.
            run: The run to record into. A new one is made when none is given.
            verbose: Unused, accepted for the same reason.
            progress_bar: Unused, accepted for the same reason.

        Returns:
            torch.Tensor: The phase the SLM is now showing.
        """
        self.timer.start()
        self.run = run if run is not None else RetrievalRun()

        geometry = self.slm_camera_model.input_geometry
        complex_dtype = (
            torch.complex128
            if geometry.wavelength.dtype == torch.float64
            else torch.complex64
        )
        grid_x, grid_y = geometry.get_spatial_grid()
        
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

        virtual_slm = self.slm_camera_model.virtual_slm
        virtual_slm.set_phase(field_superposition.angle().to(torch.float32))
        self.timer.stop()
        return virtual_slm.get_phase().detach()
