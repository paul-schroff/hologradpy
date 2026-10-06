from __future__ import annotations

from typing import Literal

import torch
from numpy.typing import ArrayLike

from .abstract import PhaseRetrieverBase
from .recorder import RetrievalRun

from ...optics.systems import SLMFourierLensModel

from ...profiles.phase import linear_phase
from ...utils import ProgressBar


class LinearSuperpositionPhaseRetriever(PhaseRetrieverBase):
    """Superposes one blazed grating per focal spot, without iterating.

    Each grating sends light to one position, with the amplitude for its intensity and
    its own phase. Gratings to a regular array with equal phases add up symmetrically
    and make ghost spots, which random phases avoid.
    """

    def __init__(
        self,
        slm_camera_model: SLMFourierLensModel,
        target_positions: ArrayLike,
        target_intensities: ArrayLike | None = None,
        target_phases: ArrayLike | Literal["random"] | None = None,
        seed: int = 0,
    ) -> None:
        """
        Args:
            slm_camera_model: The model whose virtual SLM is set.
            target_positions: ``(N, 2)`` focal-plane positions, ``(x, y)`` in metres.
            target_intensities: ``(N,)`` relative intensities. Defaults to equal ones.
            target_phases: ``(N,)`` phases in radians, or ``"random"`` for uniformly
                random ones drawn with ``seed``. Defaults to zero.
            seed: Seeds the random phases, so the same call gives the same hologram.
        """
        super().__init__(slm_camera_model)

        self.target_positions: torch.Tensor = torch.as_tensor(
            target_positions, dtype=torch.float64, device=self.device
        )
        self.number_of_positions: int = self.target_positions.shape[0]

        if target_intensities is None:
            target_intensities = torch.ones(self.number_of_positions)
        self.target_intensities: torch.Tensor = torch.as_tensor(
            target_intensities, dtype=torch.float64, device=self.device
        )

        if target_phases is None:
            target_phases = torch.zeros(self.number_of_positions)
        elif isinstance(target_phases, str):
            if target_phases != "random":
                raise ValueError(
                    f'target_phases is phases or "random", got "{target_phases}".'
                )
            target_phases = 2 * torch.pi * torch.rand(
                self.number_of_positions,
                dtype=torch.float64,
                generator=torch.Generator().manual_seed(seed),
            )
        self.target_phases: torch.Tensor = torch.as_tensor(
            target_phases, dtype=torch.float64, device=self.device
        )

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
                focal_length=self.slm_camera_model.focal_length,
            )

            field_superposition += self.target_intensities[i].sqrt() * torch.exp(
                1j * (blazed_grating + self.target_phases[i])
            )

        virtual_slm = self.slm_camera_model.virtual_slm
        virtual_slm.set_phase(field_superposition.angle().to(torch.float32))
        self.timer.stop()
        return virtual_slm.get_phase().detach()
