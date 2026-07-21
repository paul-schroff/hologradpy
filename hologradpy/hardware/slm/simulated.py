from __future__ import annotations

import numpy as np
import torch
from numpy.typing import NDArray

from ...propagation.complex_amplitude import FieldGeometry
from ...propagation.virtual_slms import VirtualSLM

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

        self.bitdepth = int(bitdepth)
        self.bitresolution = 2 ** self.bitdepth
        self._display_dtype = np.uint8 if self.bitdepth <= 8 else np.uint16

        # Multiplier for when the target wavelength differs from the design wavelength
        # (slmsuite convention: phase_scaling = wav_um / wav_design_um).
        wav_um = self._wavelength * 1e6
        wav_design = wav_um if wav_design_um is None else float(wav_design_um)
        self.phase_scaling: float = wav_um / wav_design

        self.display: NDArray = np.zeros(self._resolution, dtype=self._display_dtype)

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

    def _phase_to_gray(self, phase: NDArray) -> NDArray:
        """Desired optical phase (radians) to bit-depth grayscale integers.

        A native port of ``slmsuite.hardware.slms.slm.SLM._phase2gray``: the sign is
        flipped so phase zero maps to the maximum level, and the pattern is wrapped
        into the modulation range set by :attr:`phase_scaling`. ``phase`` is modified
        in place, as in slmsuite.
        """
        out = np.zeros(self._resolution, dtype=self._display_dtype)
        bitresolution = self.bitresolution

        if self.phase_scaling == 1:
            # Prepare the 2pi -> integer conversion factor and convert.
            factor = -(bitresolution / 2 / np.pi)
            phase *= factor

            # Casting positive floats to integers is not deterministic, so go
            # all negative first.
            maximum = np.amax(phase)
            if maximum >= 0:
                toshift = bitresolution * 2 * float(np.ceil(maximum / bitresolution))
                phase -= toshift

            np.rint(phase, out=phase)
            np.copyto(out, phase, casting="unsafe")

            # Restore phase, as the operations above are in place.
            phase *= 1 / factor

            # Shift by one so that phase 0 -> display max (more continuous).
            out -= 1

            # A fast modulo for power-of-two bit depths.
            if bitresolution & (bitresolution - 1) == 0:
                np.bitwise_and(out, int(bitresolution - 1), out=out)
            else:
                np.mod(out, bitresolution, out=out)
        else:
            # phase_scaling is folded into the scaling factor.
            factor = -(bitresolution * self.phase_scaling / 2 / np.pi)
            phase *= factor

            # Only wrap when the phase leaves the SLM bounds.
            if np.amin(phase) <= -bitresolution or np.amax(phase) > 0:
                phase -= 1  # Conform with the in-bound case.
                np.mod(phase, bitresolution * self.phase_scaling, out=phase)
                phase += bitresolution * (1 - self.phase_scaling)

                # Values still out of range are set to zero phase.
                if self.phase_scaling > 1:
                    phase[phase < 0] = bitresolution - 1
            else:
                phase += bitresolution - 1

            np.copyto(out, phase, casting="unsafe")

        return out

    def set_phase(self, phase: NDArray | torch.Tensor) -> None:
        """Display the desired optical phase (same argument convention as
        ``slmsuite.SLM.set_phase``): the phase is quantized to the SLM bit depth and
        fed to the virtual SLM.
        """
        if isinstance(phase, torch.Tensor):
            phase = phase.detach().cpu().numpy()
        phase = np.array(phase, dtype=np.float64)
        if phase.shape != self._resolution:
            raise ValueError(
                f"Phase shape {phase.shape} does not match the SLM resolution "
                f"{self._resolution}."
            )

        self.display = self._phase_to_gray(phase)

        # The virtual SLM is lazily initialised. Make sure its phase Parameter exists
        # even if no image has been captured yet.
        if not self.virtual_slm.initialized:
            self.virtual_slm.initialize_from_geometry(self.input_geometry)
        # slmsuite negates the desired phase when converting to grayscale, so undo that
        # here. The virtual SLM expects the desired optical phase.
        self.virtual_slm.set_phase(
            -(self.display / self.bitresolution * 2 * torch.pi)
        )

    def close(self) -> None:
        pass
