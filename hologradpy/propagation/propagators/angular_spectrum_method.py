from __future__ import annotations

import torch
from torch import Tensor

from ...utils import pad_to_shape_2D, crop_to_shape_2D
from ..fourier import get_frequency_grid, fft_2d, ifft_2d

from ..optics_module import OpticsModule
from ..complex_amplitude import (
    ComplexAmplitude,
    broadcast_wavelength_operand,
)


class AngularSpectrumMethod(OpticsModule):
    """Near-field propagation by the angular spectrum method.

    Propagates a field a distance ``propagation_distance`` while preserving the
    sampling (output pixel size and resolution equal the input). The field is
    zero-padded to ``padded_resolution`` before the transform to avoid circular
    convolution wraparound, then cropped back. Orthonormal FFTs are used so the
    transform conserves energy and :meth:`adjoint` (back-propagation with the
    conjugate transfer function) is the exact conjugate transpose of
    :meth:`forward`.
    """

    def __init__(
        self: AngularSpectrumMethod,
        propagation_distance: float,
        padded_resolution: tuple[int, int] | None = None,
    ) -> None:
        super().__init__()

        self.propagation_distance: float = propagation_distance
        self._padded_resolution_init: tuple[int, int] | None = padded_resolution

    def lazy_init(
        self: AngularSpectrumMethod, complex_amplitude: ComplexAmplitude
    ) -> None:
        super().lazy_init(complex_amplitude)

        resolution_in = complex_amplitude.resolution

        if self._padded_resolution_init is None:
            self._padded_resolution = tuple(2 * resolution_in[i] for i in range(2))
        else:
            if (
                self._padded_resolution_init[0] < resolution_in[0]
                or self._padded_resolution_init[1] < resolution_in[1]
            ):
                raise ValueError(
                    "Padded resolution must be at least as large as input resolution."
                )
            if any(self._padded_resolution_init[i] % 2 for i in range(2)):
                raise ValueError("Padded resolution must be even.")
            self._padded_resolution = self._padded_resolution_init

        self.register_buffer("phase_factor", self._get_phase_factor(complex_amplitude))

    def _get_phase_factor(
        self: AngularSpectrumMethod, complex_amplitude: ComplexAmplitude
    ) -> Tensor:
        """Angular-spectrum transfer function ``(n_wavelengths, H, W)``."""
        frequency_grid_x, frequency_grid_y = get_frequency_grid(
            self._padded_resolution,
            tuple(complex_amplitude.pixel_size[0].tolist()),
            complex_amplitude.device,
        )

        # Per-wavelength wavenumber broadcast over the (padded) frequency grid.
        wavenumber = complex_amplitude.wavenumber.reshape(-1, 1, 1)
        argument = (
            wavenumber**2
            - frequency_grid_x.unsqueeze(0) ** 2
            - frequency_grid_y.unsqueeze(0) ** 2
        )
        # ``+ 0j`` allows the square root to go imaginary for evanescent waves.
        return torch.exp(1j * self.propagation_distance * torch.sqrt(argument + 0j))

    def _propagate(
        self: AngularSpectrumMethod,
        complex_amplitude: ComplexAmplitude,
        transfer_function: Tensor,
    ) -> ComplexAmplitude:
        transfer_function = broadcast_wavelength_operand(
            transfer_function, complex_amplitude.ndim
        )
        padded = pad_to_shape_2D(complex_amplitude, self._padded_resolution)
        spectrum = fft_2d(padded, norm="ortho")
        propagated = ifft_2d(spectrum * transfer_function, norm="ortho")
        out = crop_to_shape_2D(propagated, self.resolution_out)
        return out.with_geometry(
            wavelength=complex_amplitude.wavelength,
            pixel_size=self.pixel_size_out,
        )

    def forward(
        self: AngularSpectrumMethod, complex_amplitude: ComplexAmplitude
    ) -> ComplexAmplitude:
        return self._propagate(complex_amplitude, self.phase_factor)

    def adjoint(
        self: AngularSpectrumMethod, complex_amplitude: ComplexAmplitude
    ) -> ComplexAmplitude:
        """Back-propagation by ``-propagation_distance`` — the conjugate
        transpose of :meth:`forward`."""
        self._ensure_initialized()
        return self._propagate(complex_amplitude, self.phase_factor.conj())
