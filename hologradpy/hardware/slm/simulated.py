from __future__ import annotations

import numpy as np
import torch
from numpy.typing import NDArray

from ...optics.complex_amplitude import FieldGeometry
from ...optics.modules.pixel_crosstalk import PixelCrosstalk
from ...optics.modules.virtual_slms import VirtualSLM
from ...phase_levels import (
    LinearResponse,
    PhaseResponse,
    PhaseResponseModule,
    level_dtype,
)

from .abstract import SLM


class SimulatedSLMTorch(SLM):
    """A native HoloGradPy SLM backed by a differentiable :class:`VirtualSLM`.

    Implements the :class:`~hologradpy.hardware.slm.SLM` interface directly (no
    slmsuite base). ``set_phase`` converts the desired optical phase to whole gray
    levels through the phase response, as for a hardware SLM. The virtual SLM then
    shows those levels, so the simulation sees the same discretized pattern as the
    hardware. The simulated SLM shows each pattern at once, and ``settle_time`` is
    only recorded for ``SLMData``.

    ``pixel_crosstalk`` adds fringing fields between neighbouring pixels to
    :attr:`virtual_slm`, and so to every model that shares it.
    :meth:`VirtualSLM.from_slm` copies only the pixel pitch and the phase response, so
    a model built from this SLM carries no crosstalk.
    """

    def __init__(
        self,
        input_geometry: FieldGeometry,
        bitdepth: int = 8,
        name: str = "SimulatedSLM",
        wav_design_um: float | None = None,
        settle_time: float = 0.3,
        pixel_crosstalk: PixelCrosstalk | None = None,
    ) -> None:
        if input_geometry.number_of_wavelengths != 1:
            raise ValueError("Only single-wavelength is supported.")

        self.input_geometry = input_geometry
        self.name = str(name)
        self._resolution: tuple[int, int] = tuple(
            int(size) for size in input_geometry.resolution
        )
        self._pixel_size = (
            torch.as_tensor(input_geometry.pixel_size)
            .reshape(2)
            .detach()
            .cpu()
            .numpy()
            .astype(np.float64)
        )
        self._wavelength = float(input_geometry.wavelength)
        self.settle_time = float(settle_time)

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

        self.virtual_slm: VirtualSLM = VirtualSLM.from_slm(
            slm=self, init_phase=None, pixel_crosstalk=pixel_crosstalk
        )

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
        """Operating wavelength in metres.

        The displayed phase is meant for this wavelength.
        """
        return self._wavelength

    @property
    def bitdepth(self) -> int:
        """Bits per pixel, read from the level-to-phase response."""
        return self.phase_response.bitdepth

    @property
    def phase_response(self) -> PhaseResponse:
        """Phase realized by a given gray level, read from the virtual SLM."""
        virtual = getattr(self, "virtual_slm", None)
        if virtual is None:
            return self._response
        return virtual.phase_response.response

    @property
    def _nominal_phase_response(self) -> PhaseResponse:
        """The response built at construction, from the two wavelengths."""
        return self._response

    def load_phase_response(self, response: PhaseResponse | None) -> None:
        """Take the gray level to phase response of the simulated SLM.

        The displayed levels stay, and their phase follows the new response. The
        response therefore changes the simulated optics as well as the conversion of
        every later phase.

        Args:
            response: The response at the SLM's bit depth. None returns to the
                response built at construction.

        Raises:
            TypeError: ``response`` is not a ``PhaseResponse``.
            ValueError: ``response`` is at another bit depth.
        """
        if response is None:
            response = self._nominal_phase_response
        else:
            self._check_phase_response(response)

        self.virtual_slm.phase_response = PhaseResponseModule(response)
        if self.virtual_slm.initialized:
            self.virtual_slm.set_levels(self.display, response.bitdepth)

    def set_levels(self, levels: NDArray | torch.Tensor) -> None:
        """Display gray levels as given, without either correction.

        Args:
            levels: Whole gray levels at the SLM resolution, each from 0 to
                ``2**bitdepth - 1``.

        Raises:
            TypeError: ``levels`` are not integers.
            ValueError: ``levels`` are not the SLM's shape, or lie outside the range
                from 0 to ``2**bitdepth - 1``.
        """
        checked = self._checked_levels(levels)

        # The virtual SLM is lazily initialized. Make sure its state exists even if no
        # image has been captured yet.
        if not self.virtual_slm.initialized:
            self.virtual_slm.initialize_for_slm_plane(self.input_geometry)
        self.display = checked
        self.virtual_slm.set_levels(self.display, self.bitdepth)

    def close(self) -> None:
        pass
