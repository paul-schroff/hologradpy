from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

import torch
import torch.nn as nn

from ....grids import get_spatial_grid
from ....phase_levels import PhaseResponse, PhaseResponseModule, LinearResponse
from ....utils import unsqueeze_to
from ..abstract import OpticsModule
from ..pixel_crosstalk import PixelCrosstalk
from ...complex_amplitude import ComplexAmplitude, FieldGeometry

if TYPE_CHECKING:
    from ....hardware.slm import SLM, SLMData


class VirtualSLM(OpticsModule):
    """Differentiable phase-only SLM module.

    Sign convention: ``phase`` holds the *desired* optical phase. The field picks up
    ``exp(1j * phase)``, wrapped to the modulation range. The value the hardware
    actually displays is the negative of it to match slmsuite's convention.

    The phase may be a single pattern ``(H, W)`` or a batch ``(N, H, W)``. A batch
    produces a field of rank ``(N, n_wavelengths, H, W)`` from a single forward pass, so
    a whole set of patterns propagates at once.

    With a :class:`~hologradpy.optics.modules.pixel_crosstalk.PixelCrosstalk` mounted,
    the field arriving must be on the sub-pixel grid, ``P`` times finer than the SLM in
    each direction. This can be achieved with a
    :class:`~hologradpy.optics.modules.grid_adapter.GridAdapter` before this module. The
    learnable ``levels`` stay at :attr:`slm_resolution`, one per real SLM pixel.
    """

    def __init__(
        self: VirtualSLM,
        phase_scaling: float = 1.0,
        init_phase: torch.Tensor | None = None,
        phase_response: PhaseResponse | None = None,
        pixel_crosstalk: PixelCrosstalk | None = None,
        quantize: bool = False,
    ) -> None:
        """
        Args:
            phase_scaling: The phase the SLM reaches at full scale, in cycles. Ignored
                when ``phase_response`` is given.
            init_phase: Desired phase to start from, at the SLM resolution.
            phase_response: The gray level to phase curve of the device.
            pixel_crosstalk: Fringing fields between neighbouring pixels. The field
                arriving must be finer than the SLM by its ``upscale_factor``.
            quantize: Round the phase to whole gray levels before the crosstalk, as the
                hardware does. The gradient passes straight through.
        """
        super().__init__()

        self.init_phase: torch.Tensor | None = init_phase
        self.quantize: bool = quantize

        self.phase_response = PhaseResponseModule(
            phase_response
            or LinearResponse(bitdepth=8, phase_scaling=phase_scaling)
        )
        self.pixel_crosstalk: PixelCrosstalk | None = pixel_crosstalk
        self._slm_pixel_size: tuple[float, float] | None = None

    @property
    def phase_scaling(self) -> float:
        """The reachable phase range in cycles, read from the response."""
        return self.phase_response.phase_scaling

    @property
    def upscale_factor(self) -> int:
        """Sub-pixels across one SLM pixel, one without crosstalk."""
        if self.pixel_crosstalk is None:
            return 1
        return self.pixel_crosstalk.upscale_factor

    @property
    def slm_resolution(self) -> tuple[int, int]:
        """The resolution of the SLM itself, which the phase pattern is stored at.

        Coarser than the field by :attr:`upscale_factor`, so 
        :meth:`get_spatial_grid_input` describes a different grid.
        """
        factor = self.upscale_factor
        return tuple(length // factor for length in self.resolution_in)

    def get_slm_grid(
        self: VirtualSLM, index: int = 0
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """The ``(x, y)`` coordinate grid of the SLM plane, in metres, using the native
        resolution of the SLM.
        """
        factor = self.upscale_factor
        pixel_size = (self.pixel_size_in[index] * factor).tolist()
        return get_spatial_grid(
            resolution=self.slm_resolution,
            pixel_size=pixel_size,
            device=self.pixel_size_in.device,
        )

    def initialize_for_slm_plane(
        self: VirtualSLM, geometry: FieldGeometry
    ) -> None:
        """Build the lazy state from the geometry of the SLM plane itself.

        The field this stage receives is finer than the SLM by
        :attr:`upscale_factor`, so a caller holding the SLM's own geometry cannot hand
        it to :meth:`initialize_from_geometry`.
        """
        factor = self.upscale_factor
        if factor > 1:
            geometry = FieldGeometry(
                wavelength=geometry.wavelength,
                pixel_size=geometry.pixel_size / factor,
                resolution=tuple(
                    length * factor for length in geometry.resolution
                ),
            )
        self.initialize_from_geometry(geometry)

    def _check_upscaled_input(self: VirtualSLM) -> None:
        """Refuse a field that is not the sub-pixel grid the crosstalk model needs."""
        factor = self.upscale_factor
        if factor == 1:
            return

        if any(length % factor for length in self.resolution_in):
            raise ValueError(
                f"{type(self).__name__} models crosstalk at {factor} sub-pixels per "
                f"SLM pixel, so the field must arrive at a multiple of that. Got "
                f"{tuple(self.resolution_in)}. Put a GridAdapter(factor={factor}) "
                "before this stage."
            )

        if self._slm_pixel_size is None:
            return

        expected = torch.tensor(
            self._slm_pixel_size,
            dtype=self.pixel_size_in.dtype,
            device=self.pixel_size_in.device,
        )
        arrived = self.pixel_size_in[0] * factor
        if not torch.allclose(arrived, expected, rtol=1e-6):
            raise ValueError(
                f"{type(self).__name__} was built for an SLM of pitch "
                f"{tuple(expected.tolist())} m, but a field {factor} times finer than "
                f"the arriving one has pitch {tuple(arrived.tolist())} m. The upstream "
                "GridAdapter factor does not match the crosstalk model."
            )

    def prepare_crosstalk(self, complex_amplitude: ComplexAmplitude) -> None:
        """Check the arriving grid and put the crosstalk model on it.

        Every subclass has to call this from ``lazy_init``, including any that builds
        its phase without calling ``VirtualSLM.lazy_init``.
        """
        self._check_upscaled_input()
        if self.pixel_crosstalk is not None:
            self.pixel_crosstalk.to(
                device=complex_amplitude.device, dtype=complex_amplitude.dtype_r
            )

    def lazy_init(self, complex_amplitude: ComplexAmplitude) -> None:
        self.prepare_crosstalk(complex_amplitude)

        if self.init_phase is None:
            self.init_phase = torch.zeros(
                self.slm_resolution,
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
    def _from_source(
        cls: type[VirtualSLM], source: SLM | SLMData, **extra
    ) -> VirtualSLM:
        """Build from anything describing an SLM: a device, or a saved record."""
        pixel_size = source.pixel_size
        if pixel_size[0] != pixel_size[1]:
            raise ValueError("Non-square pixel pitch is not supported.")
        virtual_slm = cls(
            phase_response=getattr(source, "phase_response", None), **extra
        )

        virtual_slm._slm_pixel_size = tuple(float(size) for size in pixel_size)
        return virtual_slm

    @classmethod
    def from_slm(
        cls: type[VirtualSLM],
        slm: SLM,
        init_phase: torch.Tensor | None = None,
    ) -> VirtualSLM:
        return cls._from_source(slm, init_phase=init_phase)

    @classmethod
    def from_slm_data(
        cls: type[VirtualSLM],
        slm_data: SLMData,
        init_phase: torch.Tensor | None = None,
    ) -> VirtualSLM:
        return cls._from_source(slm_data, init_phase=init_phase)

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
        if tuple(phase.shape[-2:]) != tuple(self.slm_resolution):
            raise ValueError(
                f"Phase resolution {tuple(phase.shape[-2:])} does not match the "
                f"SLM resolution {tuple(self.slm_resolution)}."
            )

        self.levels.data = self.phase_response.fraction_at(phase)

    def get_phase(self) -> torch.Tensor:
        """The desired optical phase imprinted on the field."""
        return self.phase_response.phase_at(self.displayed_levels())

    def displayed_levels(self) -> torch.Tensor:
        """The fraction of full scale the SLM is showing."""
        return self.phase_response.wrap_fraction(self.levels)

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

    def apply_phase_transforms(self: VirtualSLM, phase: torch.Tensor) -> torch.Tensor:
        """Everything between the desired phase and the phase the beam meets.

        Rounds to whole gray levels when ``quantize`` is set, then applies the crosstalk
        model, so the returned phase is ``upscale_factor`` times finer than the one
        passed in.
        """
        if self.quantize:
            levels = self.phase_response.to_levels(phase)
            phase = self.phase_response.to_phase(
                self.phase_response.quantize(levels)
            )
        if self.pixel_crosstalk is None:
            return phase
        return self.pixel_crosstalk(phase)

    def align_phase(self, phase: torch.Tensor, field_ndim: int) -> torch.Tensor:
        """Give the phase the rank the field expects, so it broadcasts."""
        if phase.ndim >= 3:
            # A batch of patterns (N, H, W). Insert the wavelength axis to get
            # (N, 1, H, W).
            return phase.unsqueeze(-3)
        return unsqueeze_to(phase, field_ndim)

    def forward(
        self: VirtualSLM, complex_amplitude: ComplexAmplitude
    ) -> ComplexAmplitude:
        phase = self.apply_phase_transforms(self.get_phase())
        transformed_phase = self.align_phase(phase, complex_amplitude.ndim)

        complex_amplitude = complex_amplitude * torch.exp(1j * transformed_phase)

        return complex_amplitude.with_geometry(
            wavelength=complex_amplitude.wavelength,
            pixel_size=self.pixel_size_out,
        )
