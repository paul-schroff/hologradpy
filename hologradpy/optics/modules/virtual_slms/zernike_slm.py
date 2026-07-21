from __future__ import annotations

import torch
from torch import nn

from ...complex_amplitude import (
    ComplexAmplitude,
    broadcast_wavelength_operand,
)
from ....profiles.zernike import (
    Zernike,
    Conventions,
    make_per_wavelength_coefficients,
)

from slmsuite.hardware.slms.slm import SLM

from .abstract import VirtualSLM


class ZernikeSLM(VirtualSLM):
    """A virtual SLM with its phase parametrized by learnable Zernike
    coefficients, with an independent set of coefficients per wavelength.

    Unlike :class:`VirtualSLM`, which learns a per-pixel phase, ``ZernikeSLM``
    learns ``(n_wavelengths, n_coefficients)`` Zernike coefficients. The
    desired phase is reconstructed from a fixed Zernike basis built lazily
    for the input field's resolution.
    """

    def __init__(
        self: ZernikeSLM,
        phase_scaling: float,
        number_of_radial_orders: int = 5,
        initial_coefficients: torch.Tensor | None = None,
        convention: Conventions = "Noll",
        unit_disk_mode: str = "fill",
    ) -> None:
        """
        Args:
            phase_scaling: SLM phase scaling factor (``phase`` displayed modulo
                ``phase_scaling * 2 * pi``).
            number_of_radial_orders: Number of radial Zernike orders to
                include (orders ``0 .. number_of_radial_orders - 1``).
            initial_coefficients: Optional initial coefficients. May be a 1D
                tensor ``(n_coefficients,)`` broadcast across all wavelengths,
                or a 2D tensor ``(n_wavelengths, n_coefficients)`` to seed each
                wavelength independently. Defaults to small random values.
            convention: Zernike ordering/normalization convention.
            unit_disk_mode: How the unit disk maps onto the resolution
                (``"fill"`` covers the corners, ``"fit"`` inscribes it).
        """
        super().__init__(phase_scaling=phase_scaling)

        self.number_of_radial_orders: int = number_of_radial_orders
        self.initial_coefficients: torch.Tensor | None = initial_coefficients
        self.convention: Conventions = convention
        self.unit_disk_mode: str = unit_disk_mode

    def lazy_init(self: ZernikeSLM, complex_amplitude: ComplexAmplitude) -> None:
        # Intentionally does NOT call VirtualSLM.lazy_init: ZernikeSLM builds its
        # phase from Zernike coefficients, not VirtualSLM's per-pixel Parameter.
        number_of_wavelengths = complex_amplitude.number_of_wavelengths

        # Build the (n_coefficients, H, W) Zernike basis for the input
        # resolution and keep it as a buffer.
        zernike = Zernike(
            resolution=self.resolution_in,
            unit_disk_mode=self.unit_disk_mode,
            number_of_radial_orders=self.number_of_radial_orders,
            convention=self.convention,
            device=complex_amplitude.device,
        )
        self.register_buffer(
            "zernike_basis",
            zernike.zernike_array.to(dtype=complex_amplitude.dtype_r),
        )
        self.number_of_coefficients: int = zernike.number_of_zernikes

        coefficients = make_per_wavelength_coefficients(
            self.initial_coefficients,
            number_of_wavelengths,
            self.number_of_coefficients,
            complex_amplitude.dtype_r,
            complex_amplitude.device,
        )
        self.zernike_coefficients = nn.Parameter(
            coefficients, requires_grad=True
        )

    def set_phase(self, phase: torch.Tensor) -> None:
        raise NotImplementedError(
            "ZernikeSLM does not support setting the phase directly; optimise "
            "the Zernike coefficients instead."
        )

    def get_phase(self: ZernikeSLM) -> torch.Tensor:
        """Reconstruct the per-wavelength phase ``(n_wavelengths, H, W)`` from
        the learnable coefficients and the Zernike basis."""
        return torch.einsum(
            "lc,chw->lhw", self.zernike_coefficients, self.zernike_basis
        )

    def forward(
        self: ZernikeSLM, complex_amplitude: ComplexAmplitude
    ) -> ComplexAmplitude:
        # Wrapped desired phase, per wavelength: (n_wavelengths, H, W).
        phase = self.get_phase().remainder(self.phase_scaling * 2 * torch.pi)

        # Align the wavelength axis at dim -3 and broadcast over any leading
        # batch dimensions (dropping the wavelength axis for a 2D field).
        phase = broadcast_wavelength_operand(phase, complex_amplitude.ndim)

        modulated = complex_amplitude * torch.exp(1j * phase)

        return modulated.with_geometry(
            wavelength=complex_amplitude.wavelength,
            pixel_size=self.pixel_size_out,
        )

    @classmethod
    def from_slm(
        cls: type[ZernikeSLM],
        slm: SLM,
        number_of_radial_orders: int = 5,
        initial_coefficients: torch.Tensor | None = None,
        convention: Conventions = "Noll",
        unit_disk_mode: str = "fill",
    ) -> ZernikeSLM:
        return cls(
            phase_scaling=slm.phase_scaling,
            number_of_radial_orders=number_of_radial_orders,
            initial_coefficients=initial_coefficients,
            convention=convention,
            unit_disk_mode=unit_disk_mode,
        )
