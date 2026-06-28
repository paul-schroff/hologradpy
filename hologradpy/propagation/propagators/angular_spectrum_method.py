from __future__ import annotations

import torch
from torch import Tensor

from ...utils import pad_to_shape_2D, crop_to_shape_2D
from ..fourier import FastFourierTransform, FourierBase

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
    convolution wraparound, then cropped back.

    The transform to and from the angular-spectrum (frequency) domain is a
    composable :class:`FourierBase` -- by default an orthonormal
    :class:`FastFourierTransform`, for which the transform conserves energy and
    :meth:`adjoint` (back-propagation with the conjugate transfer function) is
    the exact conjugate transpose of :meth:`forward`. A different transform (e.g.
    a ``ChirpZZoom`` for a zoomed / off-axis angular spectrum) can be supplied;
    the transfer function is evaluated at that transform's ``frequencies``. Note
    that for a non-unitary transform ``adjoint`` is the conjugate transpose, not
    the inverse, so the round trip is a band-limited variant of ``F^-1 H F``.
    """

    def __init__(
        self: AngularSpectrumMethod,
        propagation_distance: float,
        transform: FourierBase | None = None,
        padded_resolution: tuple[int, int] | None = None,
    ) -> None:
        super().__init__()

        self.propagation_distance: float = propagation_distance
        self._transform_init: FourierBase | None = transform
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

        if self._transform_init is None:
            self._transform: FourierBase = FastFourierTransform(
                self._padded_resolution,
                device=complex_amplitude.device,
                norm="ortho",
            )
        else:
            self._transform = self._transform_init

        self.register_buffer("phase_factor", self._get_phase_factor(complex_amplitude))

    def _get_phase_factor(
        self: AngularSpectrumMethod, complex_amplitude: ComplexAmplitude
    ) -> Tensor:
        """Angular-spectrum transfer function ``(n_wavelengths, H, W)``.

        The transfer function is evaluated at the transform's own k-space sample
        points ``self._transform.frequencies`` (rad/sample), converted to
        physical angular frequency (rad/m) by dividing by the pixel pitch
        (``pixel_size`` is ordered ``(pitch_y, pitch_x)``). For the default FFT
        this reproduces ``get_frequency_grid(padded_resolution, pixel_size)``.
        """
        omega = self._transform.frequencies  # (2, H*W), rad/sample, [0]=x, [1]=y
        pixel_size = complex_amplitude.pixel_size[0]  # (pitch_y, pitch_x)
        frequency_grid_x = omega[0].reshape(self._padded_resolution) / pixel_size[1]
        frequency_grid_y = omega[1].reshape(self._padded_resolution) / pixel_size[0]

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
        spectrum = self._transform.forward(padded)
        propagated = self._transform.adjoint(spectrum * transfer_function)
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
