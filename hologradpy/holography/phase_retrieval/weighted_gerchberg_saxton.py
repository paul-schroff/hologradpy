from __future__ import annotations

import numpy as np
import torch
from numpy.typing import ArrayLike

from ...optics.systems import SLMFourierLensModel
from ...utils import ProgressBar
from .abstract import PhaseRetrieverBase
from .recorder import RetrievalRun


class WeightedGerchbergSaxtonPhaseRetriever(PhaseRetrieverBase):
    """Focal spots of set relative intensities, by adaptive weighted Gerchberg-Saxton.

    The algorithm of D. Kim et al., Opt. Lett. 44, 3178 (2019). The field is evaluated
    at the spots only, as a sum over the SLM pixels (R. Di Leonardo et al., Opt.
    Express 15, 1913 (2007)), so each spot lands exactly at its position. An iteration
    costs two matrix products, whose size grows with the number of spots.

    The beam is the model's SLM field. Light falls on the SLM only, and the phase of the
    beam is corrected on the SLM.
    """

    def __init__(
        self,
        slm_camera_model: SLMFourierLensModel,
        target_positions: ArrayLike,
        target_intensities: ArrayLike | None = None,
        init_slm_phase: torch.Tensor | None = None,
        focal_phase_iterations: int = 20,
        seed: int = 0,
    ) -> None:
        """
        Args:
            slm_camera_model: The model of the optical system. Its SLM field, virtual
                SLM and focal length are used.
            target_positions: ``(N, 2)`` focal-plane positions, ``(x, y)`` in metres
                from the zeroth order, within the field the SLM addresses.
            target_intensities: ``(N,)`` relative intensities. Defaults to equal ones.
            init_slm_phase: The SLM phase to start from, such as the hologram being
                corrected. Defaults to a random phase drawn with ``seed``.
            focal_phase_iterations: For how many iterations the focal-plane phase
                follows the field, at least 1. After them it is held while the weights
                keep adapting. With ``init_slm_phase``, 1 holds the focal-plane phase
                of that hologram.
            seed: Seeds the random starting phase.

        Raises:
            ValueError: The intensities are not one per position, or one is zero,
                negative or not finite. A position repeats another or lies beyond the
                field the SLM addresses. ``focal_phase_iterations`` is below 1.
        """
        super().__init__(slm_camera_model)

        self.target_positions: np.ndarray = np.asarray(target_positions, dtype=float)
        number_of_positions = len(self.target_positions)
        if target_intensities is None:
            target_intensities = np.ones(number_of_positions)
        self.target_intensities: np.ndarray = np.asarray(
            target_intensities, dtype=float
        )
        if self.target_intensities.shape != (number_of_positions,):
            raise ValueError(
                f"There are {number_of_positions} positions, got intensities of shape "
                f"{self.target_intensities.shape}."
            )
        positive_and_finite = np.isfinite(self.target_intensities) & (
            self.target_intensities > 0
        )
        if not positive_and_finite.all():
            raise ValueError(
                f"Intensities {np.flatnonzero(~positive_and_finite).tolist()} are "
                "zero, negative or not finite."
            )
        half_x, half_y = slm_camera_model.addressable_half_extent()
        beyond = (np.abs(self.target_positions[:, 0]) > half_x) | (
            np.abs(self.target_positions[:, 1]) > half_y
        )
        if beyond.any():
            raise ValueError(
                f"Positions {np.flatnonzero(beyond).tolist()} lie beyond the field the "
                f"SLM addresses, {half_x:.3g} m and {half_y:.3g} m from the zeroth "
                "order along x and y."
            )
        _, group_of_position, positions_in_group = np.unique(
            self.target_positions, axis=0, return_inverse=True, return_counts=True
        )
        repeated = positions_in_group[group_of_position.reshape(-1)] > 1
        if repeated.any():
            raise ValueError(
                f"Positions {np.flatnonzero(repeated).tolist()} each repeat another "
                "position."
            )
        if focal_phase_iterations < 1:
            raise ValueError(
                "focal_phase_iterations must be at least 1, got "
                f"{focal_phase_iterations}."
            )
        self.init_slm_phase: torch.Tensor | None = init_slm_phase
        self.focal_phase_iterations: int = focal_phase_iterations
        self.seed: int = seed

    def retrieve_phase(
        self,
        number_of_iterations: int = 100,
        *,
        run: RetrievalRun | None = None,
        verbose: bool = True,
        progress_bar: ProgressBar | None = None,
        **_: object,
    ) -> torch.Tensor:
        """Iterate, and put the resulting phase onto the model.

        Args:
            number_of_iterations: Gerchberg-Saxton iterations.
            run: The run to record into. A new one is made when none is given. Its loss
                is the squared error of the spots' fractions of the light in them.
            verbose: Show a progress bar when one is not supplied.
            progress_bar: A bar to borrow, reset here and handed back untouched.

        Returns:
            torch.Tensor: The phase the SLM is now showing.
        """
        self.timer.start()
        self.run = run if run is not None else RetrievalRun()
        borrowed = progress_bar is not None
        if borrowed:
            progress_bar.reset(total=number_of_iterations)
        else:
            progress_bar = ProgressBar(
                total=number_of_iterations,
                description="Weighted Gerchberg-Saxton",
                verbose=verbose,
            ).__enter__()
        self.run.progress_bar = progress_bar
        try:
            self._iterate(number_of_iterations)
        finally:
            self.run.progress_bar = None
            if not borrowed:
                progress_bar.close()

        self.timer.stop()
        return self.slm_camera_model.virtual_slm.get_phase().detach()

    def _spot_phasors(
        self, height: int, width: int, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """The phase factors that carry the SLM field to the spots, ``(N, height)``
        along the rows of the SLM and ``(N, width)`` along its columns.

        The field at spot ``n`` is the sum over the SLM of the field times
        ``row_phasors[n, :, None] * column_phasors[n, None, :]``, which is the Fourier
        transform of the lens evaluated at the position of the spot.
        """
        geometry = self.slm_camera_model.input_geometry
        pitch_y, pitch_x = geometry.pixel_size.reshape(-1, 2)[0].tolist()
        wavelength = float(geometry.wavelength.reshape(-1)[0])
        # Radians per metre of focal-plane position and per metre across the SLM.
        scale = 2 * np.pi / (wavelength * self.slm_camera_model.focal_length)
        positions = torch.as_tensor(
            self.target_positions, dtype=torch.float64, device=self.device
        )
        rows = torch.arange(height, dtype=torch.float64, device=self.device)
        columns = torch.arange(width, dtype=torch.float64, device=self.device)
        rows = (rows - height // 2) * pitch_y
        columns = (columns - width // 2) * pitch_x
        row_phasors = torch.exp(-1j * scale * positions[:, 1:] * rows)
        column_phasors = torch.exp(-1j * scale * positions[:, :1] * columns)
        return row_phasors.to(dtype), column_phasors.to(dtype)

    def _iterate(self, number_of_iterations: int) -> None:
        """Iterate, and set the virtual SLM to the result, beam phase corrected.

        A recorded step is read off the SLM, so the SLM is also set at every iteration
        while steps are recorded.
        """
        model = self.slm_camera_model
        with torch.no_grad():
            beam = model.slm_field.get_wavefront().detach().to(self.device)
        height, width = beam.shape[-2:]
        amplitude = beam.abs()
        row_phasors, column_phasors = self._spot_phasors(height, width, beam.dtype)

        def propagate_to_spots(field: torch.Tensor) -> torch.Tensor:
            """The ``(N,)`` field at the spots."""
            return (row_phasors * (field @ column_phasors.T).T).sum(-1)

        def propagate_from_spots(spot_field: torch.Tensor) -> torch.Tensor:
            """The ``(height, width)`` SLM field, by the adjoint of
            ``propagate_to_spots``.
            """
            return (row_phasors.conj().T * spot_field) @ column_phasors.conj()

        def set_slm_phase(field_phase: torch.Tensor) -> None:
            """Set the SLM to the phase that turns the beam into a field of phase
            ``field_phase``.
            """
            phase = torch.remainder(field_phase - beam.angle(), 2 * torch.pi)
            model.virtual_slm.set_phase(phase.to(torch.float32))

        weights = torch.as_tensor(
            self.target_intensities, dtype=amplitude.dtype, device=self.device
        )
        weights = weights / weights.sum()
        target_amplitude = weights.sqrt()

        if self.init_slm_phase is None:
            generator = torch.Generator().manual_seed(self.seed)
            field_phase = (
                2 * torch.pi * torch.rand((height, width), generator=generator)
            )
            field_phase = field_phase.to(device=self.device, dtype=amplitude.dtype)
        else:
            field_phase = (
                torch.as_tensor(
                    self.init_slm_phase, dtype=amplitude.dtype, device=self.device
                )
                + beam.angle()
            )

        focal_phase = None
        for iteration in range(number_of_iterations):
            spot_field = propagate_to_spots(amplitude * torch.exp(1j * field_phase))
            spot_intensity = spot_field.abs() ** 2
            fractions = spot_intensity / spot_intensity.sum()
            self.run.record_loss(float(((fractions - weights) ** 2).sum()))

            # Each spot's target amplitude is weighted by how far it falls short, as
            # in Kim et al.
            shortfall = (spot_intensity / weights).sqrt()
            target_amplitude = target_amplitude * shortfall.mean() / shortfall
            if iteration < self.focal_phase_iterations or focal_phase is None:
                focal_phase = torch.exp(1j * spot_field.angle())

            field_phase = propagate_from_spots(target_amplitude * focal_phase).angle()
            if self.run.steps is not None:
                set_slm_phase(field_phase)
            self.run.record_iteration(iteration + 1, model)

        set_slm_phase(field_phase)
