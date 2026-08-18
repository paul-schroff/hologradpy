from __future__ import annotations

import numpy as np
import torch
from numpy.typing import NDArray

from ...optics.complex_amplitude import FieldGeometry
from ...optics.modules.virtual_slms import VirtualSLM
from ...phase_levels import LinearResponse, PhaseResponse, level_dtype

from .abstract import SLM


class SimulatedSLMTorch(SLM):
    """A native HoloGradPy SLM backed by a differentiable :class:`VirtualSLM`.

    Implements the :class:`~hologradpy.hardware.slm.SLM` interface directly (no
    slmsuite base). ``set_phase`` quantizes the desired optical phase to the SLM's bit
    depth (the slmsuite phase-to-grayscale routine, including the sign-flip and
    ``phase_scaling`` wrap) and feeds the quantized phase to the virtual SLM, so the
    simulation sees the same discretized pattern the hardware would display.
    """

    def __init__(
        self,
        input_geometry: FieldGeometry,
        bitdepth: int = 8,
        name: str = "SimulatedSLM",
        wav_design_um: float | None = None,
        settle_time_s: float = 0.3,
    ) -> None:
        if input_geometry.wavelength.ndim != 0:
            raise ValueError("Only single-wavelength is supported.")

        self.input_geometry = input_geometry
        self.name = str(name)
        self._resolution: tuple[int, int] = tuple(
            int(size) for size in input_geometry.resolution
        )
        self._pixel_size = (
            torch.as_tensor(input_geometry.pixel_size)
            .detach()
            .cpu()
            .numpy()
            .astype(np.float64)
        )
        self._wavelength = float(input_geometry.wavelength)
        self.settle_time_s = float(settle_time_s)

        self._bitdepth = int(bitdepth)

        # Phase response scales with wavelength.
        wav_um = self._wavelength * 1e6
        wav_design = wav_um if wav_design_um is None else float(wav_design_um)
        self._response: PhaseResponse = LinearResponse(
            bitdepth=self._bitdepth, phase_scaling=wav_design / wav_um
        )

        self.display: NDArray = np.zeros(
            self._resolution, dtype=level_dtype(self._bitdepth)
        )

        self.virtual_slm: VirtualSLM = VirtualSLM.from_slm(slm=self, init_phase=None)

    @property
    def pixel_size(self) -> NDArray[np.float64]:
        """Pixel pitch ``(y, x)`` in metres."""
        return self._pixel_size

    @property
    def resolution(self) -> tuple[int, int]:
        """SLM resolution ``(height, width)`` in pixels."""
        return self._resolution

    @property
    def wavelength(self) -> float:
        """Design wavelength in metres."""
        return self._wavelength

    @property
    def bitdepth(self) -> int:
        """Bits per pixel, read from the level-to-phase response."""
        return self.phase_response.bitdepth

    @property
    def phase_response(self) -> PhaseResponse:
        """Phase realized by applying a certain gray level, read from the virtual SLM.
        """
        virtual = getattr(self, "virtual_slm", None)
        if virtual is None:
            return self._response
        return virtual.phase_response.response

    def set_levels(self, levels: NDArray | torch.Tensor) -> None:
        """Display gray levels directly, without going through a phase."""
        levels = self._as_frame(levels, "Levels")
        self.display = levels.astype(level_dtype(self.bitdepth))
        self.virtual_slm.set_levels(self.display, self.bitdepth)

    def _as_frame(self, pattern: NDArray | torch.Tensor, label: str) -> NDArray:
        """A displayable numpy frame, with the virtual SLM ready to be given it."""
        if isinstance(pattern, torch.Tensor):
            pattern = pattern.detach().cpu().numpy()
        pattern = np.asarray(pattern)
        if pattern.shape != self._resolution:
            raise ValueError(
                f"{label} shape {pattern.shape} does not match the SLM resolution "
                f"{self._resolution}."
            )

        # The virtual SLM is lazily initialized. Make sure its state exists even if no
        # image has been captured yet.
        if not self.virtual_slm.initialized:
            self.virtual_slm.initialize_for_slm_plane(self.input_geometry)
        return pattern

    def close(self) -> None:
        pass
