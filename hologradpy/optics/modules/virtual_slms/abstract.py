from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

import torch
import torch.nn as nn

from ....phase_levels import PhaseResponse, PhaseResponseModule, LinearResponse
from ....utils import unsqueeze_to
from ..abstract import OpticsModule
from ...complex_amplitude import ComplexAmplitude

from slmsuite.hardware.slms.slm import SLM

if TYPE_CHECKING:
    from ....hardware.slm import SLMData


class VirtualSLM(OpticsModule):
    """Differentiable phase-only SLM module.

    Sign convention: ``phase`` holds the *desired* optical phase. The field picks up
    ``exp(1j * phase)``, wrapped to the modulation range. The value the hardware
    actually displays is the negative of it to match slmsuite's convention.

    The phase may be a single pattern ``(H, W)`` or a batch ``(N, H, W)``. A batch
    produces a field of rank ``(N, n_wavelengths, H, W)`` from a single forward pass, so
    a whole set of patterns propagates at once.
    """

    def __init__(
        self: VirtualSLM,
        phase_scaling: float,
        init_phase: torch.Tensor | None = None,
        phase_response: PhaseResponse | None = None,
    ) -> None:
        super().__init__()

        self.phase_scaling: float = phase_scaling
        self.init_phase: torch.Tensor | None = init_phase

        self.phase_response = PhaseResponseModule(
            phase_response
            or LinearResponse(bitdepth=8, phase_scaling=phase_scaling)
        )

    def lazy_init(self, complex_amplitude: ComplexAmplitude) -> None:
        if self.init_phase is None:
            self.init_phase = torch.zeros(
                self.resolution_in,
                device=complex_amplitude.device,
                dtype=complex_amplitude.dtype_r,
            )
        else:
            self.init_phase = self.init_phase.to(
                device=complex_amplitude.device, dtype=complex_amplitude.dtype_r
            )

        self.levels = nn.Parameter(
            self.phase_response.fraction_at(self.init_phase), requires_grad=False
        )

    @classmethod
    def from_slm(
        cls: type[VirtualSLM],
        slm: SLM,
        init_phase: torch.Tensor | None = None,
    ) -> VirtualSLM:
        if slm.pixel_size[0] != slm.pixel_size[1]:
            raise ValueError("Non-square pixel pitch is not supported.")
        return cls(
            phase_scaling=slm.phase_scaling,
            init_phase=init_phase,
            phase_response=getattr(slm, "phase_response", None),
        )

    @classmethod
    def from_slm_data(
        cls: type[VirtualSLM],
        slm_data: SLMData,
        init_phase: torch.Tensor | None = None,
    ) -> VirtualSLM:
        if slm_data.pixel_size[0] != slm_data.pixel_size[1]:
            raise ValueError("Non-square pixel pitch is not supported.")
        return cls(
            phase_scaling=slm_data.phase_scaling,
            init_phase=init_phase,
            phase_response=getattr(slm_data, "phase_response", None),
        )

    def set_phase(self, phase: torch.Tensor | NDArray) -> None:
        """Set the desired optical phase (same argument convention as
        ``slmsuite.SLM.set_phase``).

        Accepts either a single pattern ``(H, W)`` or a batch ``(N, H, W)``. A batch is
        imprinted in one forward pass, which is far cheaper than looping: the whole
        chain then runs once on a batched field instead of once per pattern.
        """
        if isinstance(phase, np.ndarray):
            phase = torch.as_tensor(phase)
        phase = phase.to(dtype=self.levels.dtype, device=self.levels.device)

        if phase.ndim not in (2, 3):
            raise ValueError(
                f"Phase must be a single (H, W) pattern or a batch of them "
                f"(N, H, W), got shape {tuple(phase.shape)}."
            )
        if tuple(phase.shape[-2:]) != tuple(self.resolution_in):
            raise ValueError(
                f"Phase resolution {tuple(phase.shape[-2:])} does not match the "
                f"SLM resolution {tuple(self.resolution_in)}."
            )

        self.levels.data = self.phase_response.fraction_at(phase)

    def get_phase(self) -> torch.Tensor:
        """The desired optical phase imprinted on the field."""
        return self.phase_response.phase_at(self.displayed_levels())

    def displayed_levels(self) -> torch.Tensor:
        """The fraction of full scale the panel is showing."""
        return self.levels % self.phase_scaling

    def set_levels(
        self, levels: torch.Tensor | NDArray, bitdepth: int | None = None
    ) -> None:
        """Imprint displayed grayscale levels, rather than a desired phase."""
        if not isinstance(levels, torch.Tensor):
            levels = torch.as_tensor(np.asarray(levels))
        self.set_phase(self.levels_to_phase(levels, bitdepth))

    def _checked_bitdepth(self, bitdepth: int | None) -> int:
        """The response's own bit depth, and a loud error when a caller disagrees."""
        mine = self.phase_response.bitdepth
        if bitdepth is not None and int(bitdepth) != int(mine):
            raise ValueError(
                f"These levels are {bitdepth}-bit but this SLM's response is "
                f"{mine}-bit, so they do not mean the same phase."
            )
        return mine

    def levels_to_phase(
        self, levels: torch.Tensor | NDArray, bitdepth: int | None = None
    ) -> torch.Tensor:
        """The desired optical phase that displaying ``levels`` imposes."""
        self._checked_bitdepth(bitdepth)
        if not isinstance(levels, torch.Tensor):
            levels = torch.as_tensor(np.asarray(levels))
        return self.phase_response.response.to_phase(levels)

    def phase_to_levels(
        self, phase: torch.Tensor | NDArray, bitdepth: int | None = None
    ) -> NDArray:
        """The levels that impose ``phase``, ready to display."""
        self._checked_bitdepth(bitdepth)
        if isinstance(phase, torch.Tensor):
            phase = phase.detach().cpu().numpy()
        return self.phase_response.response.display_levels(np.asarray(phase))

    def get_displayed_phase(self) -> torch.Tensor:
        """The phase pattern as displayed on the SLM before grayscale conversion: the
        hardware displays the negative of the desired phase.
        """
        return (-self.get_phase()).remainder(self.phase_scaling * 2 * torch.pi)

    # TODO: Add discretization and pixel crosstallk here
    def apply_phase_transforms(self: VirtualSLM, phase: torch.Tensor) -> torch.Tensor:
        """Hook for subclasses to transform the applied phase; identity by default."""
        return phase

    def forward(
        self: VirtualSLM, complex_amplitude: ComplexAmplitude
    ) -> ComplexAmplitude:
        phase = self.apply_phase_transforms(self.get_phase())

        if phase.ndim >= 3:
            # A batch of patterns (N, H, W). Insert the wavelength axis to get
            # (N, 1, H, W).
            transformed_phase = phase.unsqueeze(-3)
        else:
            transformed_phase = unsqueeze_to(phase, complex_amplitude.ndim)

        complex_amplitude = complex_amplitude * torch.exp(1j * transformed_phase)

        return complex_amplitude.with_geometry(
            wavelength=complex_amplitude.wavelength,
            pixel_size=self.pixel_size_out,
        )
