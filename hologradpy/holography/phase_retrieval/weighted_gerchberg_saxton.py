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

    The algorithm of D. Kim et al., Opt. Lett. 44, 3178 (2019). The SLM plane
    and the focal plane are related by an FFT on a grid padded around the SLM, so each
    spot lands on the nearest pixel of the focal-plane grid, whose pitch is
    ``wavelength * focal_length / (N * pitch)``.

    The beam is the model's SLM field. Light falls on the SLM only, and the phase of the
    beam is corrected on the SLM.
    """

    def __init__(
        self,
        slm_camera_model: SLMFourierLensModel,
        target_positions: ArrayLike,
        target_intensities: ArrayLike | None = None,
        padded_resolution: tuple[int, int] = (4096, 4096),
        init_slm_phase: torch.Tensor | None = None,
        focal_phase_iterations: int = 20,
        seed: int = 0,
    ) -> None:
        """
        Args:
            slm_camera_model: The model of the optical system. Its SLM field, virtual
                SLM and focal length are used.
            target_positions: ``(N, 2)`` focal-plane positions, ``(x, y)`` in metres.
            target_intensities: ``(N,)`` relative intensities. Defaults to equal ones.
            padded_resolution: The ``(height, width)`` of the FFT grid, at least the
                SLM's. A larger grid gives a finer focal-plane grid.
            init_slm_phase: The SLM phase to start from, such as the hologram being
                corrected. Defaults to a random phase drawn with ``seed``.
            focal_phase_iterations: For how many iterations the focal-plane phase
                follows the field, before it is held. The adaptive part of the
                algorithm. Starting from a hologram, 1 keeps its focal-plane phase.
            seed: Seeds the random starting phase.
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
        self.padded_resolution: tuple[int, int] = tuple(padded_resolution)
        self.init_slm_phase: torch.Tensor | None = init_slm_phase
        self.focal_phase_iterations: int = focal_phase_iterations
        self.seed: int = seed

    def focal_plane_pitch(self) -> tuple[float, float]:
        """The ``(y, x)`` pitch of the focal-plane grid the spots land on, in metres."""
        geometry = self.slm_camera_model.input_geometry
        pitch = geometry.pixel_size.reshape(-1, 2)[0].tolist()
        scale = (
            float(geometry.wavelength.reshape(-1)[0])
            * self.slm_camera_model.focal_length
        )
        return tuple(
            scale / (padded * pitch)
            for padded, pitch in zip(self.padded_resolution, pitch)
        )

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

        Raises:
            ValueError: A position lies off the focal-plane grid.
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
            phase = self._iterate(number_of_iterations)
        finally:
            self.run.progress_bar = None
            if not borrowed:
                progress_bar.close()

        virtual_slm = self.slm_camera_model.virtual_slm
        virtual_slm.set_phase(phase.to(torch.float32))
        self.timer.stop()
        return virtual_slm.get_phase().detach()

    def _iterate(self, number_of_iterations: int) -> torch.Tensor:
        """The SLM phase after ``number_of_iterations``, beam phase corrected."""
        model = self.slm_camera_model
        with torch.no_grad():
            beam = model.slm_field.get_wavefront().detach().to(self.device)
        height, width = beam.shape[-2:]
        padded_height, padded_width = self.padded_resolution
        if padded_height < height or padded_width < width:
            raise ValueError(
                f"The padded resolution {self.padded_resolution} is smaller than the "
                f"SLM's {(height, width)}."
            )
        # The SLM's centre pixel, height // 2, sits on the grid's, padded_height // 2.
        slm = (
            slice(
                padded_height // 2 - height // 2,
                padded_height // 2 - height // 2 + height,
            ),
            slice(
                padded_width // 2 - width // 2, padded_width // 2 - width // 2 + width
            ),
        )
        amplitude = torch.zeros(
            self.padded_resolution, dtype=beam.real.dtype, device=self.device
        )
        amplitude[slm] = beam.abs()

        pitch_y, pitch_x = self.focal_plane_pitch()
        rows = np.rint(self.target_positions[:, 1] / pitch_y).astype(int)
        columns = np.rint(self.target_positions[:, 0] / pitch_x).astype(int)
        rows += padded_height // 2
        columns += padded_width // 2
        off = (
            (rows < 0)
            | (rows >= padded_height)
            | (columns < 0)
            | (columns >= padded_width)
        )
        if off.any():
            raise ValueError(
                f"Positions {np.flatnonzero(off).tolist()} lie off the focal plane."
            )
        sites = (
            torch.as_tensor(rows, device=self.device),
            torch.as_tensor(columns, device=self.device),
        )

        weights = torch.as_tensor(
            self.target_intensities, dtype=amplitude.dtype, device=self.device
        )
        weights = weights / weights.sum()
        target_amplitude = torch.zeros_like(amplitude)
        target_amplitude[sites] = weights.sqrt()

        if self.init_slm_phase is None:
            generator = torch.Generator().manual_seed(self.seed)
            phase = (
                2 * torch.pi * torch.rand(self.padded_resolution, generator=generator)
            )
            phase = phase.to(device=self.device, dtype=amplitude.dtype)
        else:
            phase = torch.zeros_like(amplitude)
            phase[slm] = (
                torch.as_tensor(
                    self.init_slm_phase, dtype=amplitude.dtype, device=self.device
                )
                + beam.angle()
            )
        field = amplitude * torch.exp(1j * phase)

        focal_phase = None
        for iteration in range(number_of_iterations):
            focal = torch.fft.fftshift(torch.fft.fft2(field))
            site_intensity = focal[sites].abs() ** 2
            fractions = site_intensity / site_intensity.sum()
            self.run.record_loss(float(((fractions - weights) ** 2).sum()))

            # Kim et al.: weight each spot's target amplitude by how far it falls short.
            shortfall = (site_intensity / weights).sqrt()
            target_amplitude[sites] *= shortfall.mean() / shortfall
            if iteration < self.focal_phase_iterations or focal_phase is None:
                focal_phase = torch.exp(1j * focal.angle())

            back = torch.fft.ifft2(torch.fft.ifftshift(target_amplitude * focal_phase))
            field = amplitude * torch.exp(1j * back.angle())
            self.run.record_iteration(iteration + 1, model)

        return torch.remainder(field.angle()[slm] - beam.angle(), 2 * torch.pi)
