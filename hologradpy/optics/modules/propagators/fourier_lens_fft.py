from __future__ import annotations

import torch
from torch import nn, Tensor

from ....fourier_transforms import FastFourierTransform
from ....utils import to_canvas

from ..abstract import capture_init, OpticsModule
from ....fourier_optics import (
    fourier_lens_pixel_size,
    fourier_lens_power_prefactor,
    fourier_lens_resolution,
)
from ...complex_amplitude import (
    ComplexAmplitude,
    broadcast_wavelength_operand,
    pixel_area,
)


class FourierLensFFT(OpticsModule):
    @capture_init
    def __init__(
        self,
        focal_length: float,
        pixel_size_out: tuple[float, float] | None = None,
        padded_resolution: tuple[int, int] | None = None,
        power_normalized: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(pixel_size_out, padded_resolution)

        if pixel_size_out is not None and padded_resolution is not None:
            raise ValueError(
                "Specify either pixel_size_out or padded_resolution, not both."
            )

        self.focal_length: float = nn.Parameter(
            torch.tensor(focal_length, dtype=torch.float32),
            requires_grad=False,
        )
        self._padded_resolution_init: tuple[int, int] | None = padded_resolution

        # Scale the transform by the physical Fourier-optics prefactor so a
        # lossless lens conserves optical power. On by default.
        self.power_normalized: bool = power_normalized

        self.kwargs = kwargs

    def lazy_init(self, complex_amplitude: ComplexAmplitude) -> None:
        # This lens genuinely computes its output geometry from the input; it sets
        # _resolution_out / _pixel_size_out directly (pixel_size_out is also a
        # dynamic property below). Branch on which constructor arg was given.
        # Only padded_resolution is provided
        if self._padded_resolution_init is not None:
            # check if the provided padded resolution is at least
            # as large as the input resolution
            if (
                self._padded_resolution_init[0] < complex_amplitude.resolution[0]
                or self._padded_resolution_init[1] < complex_amplitude.resolution[1]
            ):
                raise ValueError(
                    "Padded resolution must be at least as large as input resolution."
                )

            # Check if padded_resolution is even
            parity = tuple(self._padded_resolution_init[i] % 2 for i in range(2))

            if parity[0] != 0 or parity[1] != 0:
                raise ValueError("Padded resolution must be even.")

            self._padded_resolution = torch.tensor(
                self._padded_resolution_init,
                device=complex_amplitude.device,
                dtype=torch.int64,
            )
            self._pixel_size_out = self._get_pixel_size_out(
                complex_amplitude.wavelength,
                self.focal_length,
                self.pixel_size_in,
                self._padded_resolution,
            )
        # Only pixel_size_out is provided
        elif (
            self._padded_resolution_init is None
            and self._pixel_size_out_init is not None
        ):
            requested_pixel_size_out = torch.tensor(
                self._pixel_size_out_init,
                device=complex_amplitude.device,
                dtype=torch.float32,
            )

            max_pixel_size_out = self._get_pixel_size_out(
                complex_amplitude.wavelength[0],
                self.focal_length,
                self.pixel_size_in[0],
                torch.tensor(
                    complex_amplitude.resolution,
                    device=complex_amplitude.device,
                ),
            )

            # Check if the requested pixel size is larger than the maximum
            # allowed by the input geometry and focal length
            if (
                requested_pixel_size_out[0] > max_pixel_size_out[0]
                or requested_pixel_size_out[1] > max_pixel_size_out[1]
            ):
                raise ValueError(
                    "Requested pixel size out is too large for the given "
                    "input geometry and focal length. Maximum pixel size out "
                    f"for the first wavelength is {max_pixel_size_out}."
                )

            # Uses the first wavelength and pixel size for calculating the
            # padded resolution.
            self._padded_resolution = self._get_padded_resolution(
                complex_amplitude.wavelength[0],
                self.focal_length,
                self.pixel_size_in[0],
                requested_pixel_size_out,
            )

            self._pixel_size_out = self._get_pixel_size_out(
                complex_amplitude.wavelength,
                self.focal_length,
                self.pixel_size_in,
                self._padded_resolution,
            )
        # If neither is provided, default to zero-padding to double the
        # input resolution
        else:
            self._padded_resolution_init = tuple(
                2 * complex_amplitude.resolution[i] for i in range(2)
            )
            self._padded_resolution = torch.tensor(
                self._padded_resolution_init,
                device=complex_amplitude.device,
                dtype=torch.int64,
            )
            self._pixel_size_out = self._get_pixel_size_out(
                complex_amplitude.wavelength,
                self.focal_length,
                self.pixel_size_in,
                self._padded_resolution,
            )

        self._resolution_out = tuple(self._padded_resolution.tolist())

        # Compose the resolution-preserving FFT on the padded grid. ``self.kwargs``
        # carries any ``norm`` / ``fft_shift`` (defaults match ``fft_2d``), so
        # this is behaviour-preserving.
        self._transform = FastFourierTransform(
            self._resolution_out,
            device=complex_amplitude.device,
            **self.kwargs,
        )

    @property
    def padded_resolution(self) -> Tensor:
        return self._padded_resolution

    @staticmethod
    def _get_padded_resolution(
        wavelength: Tensor,
        focal_length: Tensor,
        pixel_size_in: Tensor,
        pixel_size_out: Tensor,
    ) -> Tensor[torch.int64]:
        # Rounded down to even.
        padded_resolution = (
            fourier_lens_resolution(
                wavelength, focal_length, pixel_size_in, pixel_size_out
            )
            // 2
            * 2
        )
        return padded_resolution.to(torch.int64)

    @staticmethod
    def _get_pixel_size_out(
        wavelength: Tensor,
        focal_length: Tensor,
        pixel_size_in: Tensor,
        padded_resolution: Tensor,
    ) -> Tensor:
        return fourier_lens_pixel_size(
            wavelength, focal_length, pixel_size_in, padded_resolution
        )

    def _power_prefactor(self, field_ndim: int) -> Tensor:
        """Physical Fourier-optics amplitude prefactor ``(du * dv) / (lambda *
        f)`` per wavelength, so the transform conserves optical power
        (``integral|E_focal|^2 dx == integral|E_slm|^2 du`` with ``norm=
        "backward"``). Computed in float64 then cast to the field's real dtype;
        the ``1/i`` global phase is omitted as it does not affect power."""
        pixel_size_in = self.pixel_size_in
        area = pixel_area(pixel_size_in)
        wavelength = self.input_geometry.wavelength.to(torch.float64)
        focal_length = self.focal_length.to(torch.float64)
        prefactor = fourier_lens_power_prefactor(area, wavelength, focal_length)
        prefactor = prefactor.to(pixel_size_in.dtype).reshape(-1, 1, 1)
        return broadcast_wavelength_operand(prefactor, field_ndim)

    def forward(self, complex_amplitude: ComplexAmplitude) -> ComplexAmplitude:
        padded_complex_amplitude = to_canvas(
            complex_amplitude, self.resolution_out
        )

        # Perform 2D FFT and FFT shift if specified
        out = self._transform.forward(padded_complex_amplitude)

        out = out.with_geometry(
            wavelength=complex_amplitude.wavelength,
            pixel_size=self.pixel_size_out,
        )

        if self.power_normalized:
            out = out * self._power_prefactor(out.ndim)

        return out

    def adjoint(self, complex_amplitude: ComplexAmplitude) -> ComplexAmplitude:
        """Conjugate transpose of :meth:`forward`, which is not its inverse.

        The forward is ``c * F(pad(x))``, so the adjoint is ``c * crop(F^H(y))``:
        cropping is the adjoint of the zero-padding, and ``F^H`` is the transform's own
        :attr:`~hologradpy.fourier_transforms.FastFourierTransform.adjoint_scale` times
        its ``adjoint``. Taken from the transform, since the factor follows the ``norm``
        this lens forwards to it through ``**kwargs``.
        """
        padded_complex_amplitude = self._transform.adjoint(complex_amplitude)

        out: ComplexAmplitude = to_canvas(
            padded_complex_amplitude, self.resolution_in
        )

        out = out.with_geometry(
            wavelength=complex_amplitude.wavelength,
            pixel_size=self.pixel_size_in,
        )
        
        out = out * self._transform.adjoint_scale

        if self.power_normalized:
            out = out * self._power_prefactor(out.ndim)

        return out
