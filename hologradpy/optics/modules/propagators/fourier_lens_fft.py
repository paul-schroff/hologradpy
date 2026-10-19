from __future__ import annotations

import math
import warnings

import torch
from torch import Tensor

from ....fourier_transforms import FastFourierTransform
from ....utils import resolve_padded_resolution, to_canvas

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

# The number of samples above which an FFT sized from pixel_size_out warns of its
# memory and time, 4096 x 4096.
FFT_SAMPLES_WARNING = 4096 * 4096


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


        self.register_buffer(
            "focal_length", torch.tensor(focal_length, dtype=torch.float32)
        )
        self.focal_length: Tensor
        self._padded_resolution_init: tuple[int, int] | None = padded_resolution

        # Scale the transform by the physical Fourier-optics prefactor so a
        # lossless lens conserves optical power. On by default.
        self.power_normalized: bool = power_normalized

        self.kwargs = kwargs

    def lazy_init(self, complex_amplitude: ComplexAmplitude) -> None:
        self.focal_length = self.focal_length.to(complex_amplitude.device)

        # The output geometry is computed from the input, and set on _resolution_out
        # and _pixel_size_out directly. Without pixel_size_out, the padded resolution
        # is the one given, or twice the input resolution.
        if self._pixel_size_out_init is None:
            self._padded_resolution = torch.tensor(
                resolve_padded_resolution(
                    self._padded_resolution_init, complex_amplitude.resolution
                ),
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
        else:
            requested_pixel_size_out = torch.tensor(
                self._pixel_size_out_init,
                device=complex_amplitude.device,
                dtype=torch.float32,
            )

            # The padded resolution comes from the first wavelength and pixel size.
            self._padded_resolution = self._get_padded_resolution(
                complex_amplitude.wavelength[0],
                self.focal_length,
                self.pixel_size_in[0],
                requested_pixel_size_out,
            )
            smallest = torch.tensor(
                [2 * math.ceil(length / 2) for length in complex_amplitude.resolution],
                device=complex_amplitude.device,
                dtype=torch.int64,
            )
            if bool((self._padded_resolution < smallest).any()):
                coarsest = self._get_pixel_size_out(
                    complex_amplitude.wavelength[0],
                    self.focal_length,
                    self.pixel_size_in[0],
                    smallest,
                )
                raise ValueError(
                    f"The output pitch {self._pixel_size_out_init} m needs an FFT of "
                    f"{tuple(self._padded_resolution.tolist())} samples, fewer than "
                    f"the input's {tuple(complex_amplitude.resolution)}. The coarsest "
                    "pitch this input reaches is "
                    f"{tuple(float(f'{value:.4g}') for value in coarsest.tolist())} m."
                )
            if int(self._padded_resolution.prod()) > FFT_SAMPLES_WARNING:
                warnings.warn(
                    f"The output pitch {self._pixel_size_out_init} m sizes the FFT at "
                    f"{tuple(self._padded_resolution.tolist())} samples, more than "
                    "4096 x 4096. Pass a coarser pixel_size_out, or a "
                    "padded_resolution, to limit its memory and time.",
                    stacklevel=2,
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
        # this is behavior-preserving.
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
        the ``1/i`` global phase is omitted as it does not affect power.
        """
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
