"""Resampling between the grids of adjacent stages."""

from __future__ import annotations

import warnings

import torch

from ...fourier_transforms import fft_resample

from .abstract import OpticsModule, capture_init
from ..complex_amplitude import ComplexAmplitude


class GridAdapter(OpticsModule):
    """Resample a field onto a finer or coarser grid over the same physical extent.

    Sits between two modules that want different sampling. The pixel size shrinks by
    ``factor`` and the resolution grows by it, so the spatial extent is preserved.

    The resampling is band-limited
    (:func:`~hologradpy.fourier_transforms.fft_resample`), so it is exact for anything
    the coarse grid already resolves. A ``factor`` of one passes the field through
    untouched.
    """

    @capture_init
    def __init__(
        self: GridAdapter,
        factor: int = 1,
        cache: bool = False,
    ) -> None:
        """
        Args:
            factor: Integer ratio of output to input resolution. One is a pass-through.
            cache: Hold the first output and return it for every later call. Call
                :meth:`clear_cache` after anything upstream moves.
        """
        super().__init__()

        if int(factor) < 1:
            raise ValueError(f"factor must be a positive integer, got {factor}.")

        self.factor: int = int(factor)
        self.cache: bool = cache
        self._cached: ComplexAmplitude | None = None

    def lazy_init(self: GridAdapter, complex_amplitude: ComplexAmplitude) -> None:
        if self.factor == 1:
            return

        self.set_output_geometry(
            resolution=tuple(length * self.factor for length in self.resolution_in),
            pixel_size=self.pixel_size_in / self.factor,
        )

        if complex_amplitude.device.type != "cuda":
            height, width = self.resolution_out
            warnings.warn(
                f"{type(self).__name__} is upscaling by {self.factor} to "
                f"{height}x{width} on {complex_amplitude.device}. Every stage after it "
                f"pays {self.factor ** 2} times the work, which is slow without a GPU.",
                stacklevel=2,
            )

    def clear_cache(self: GridAdapter) -> None:
        """Drop the held field, so the next call resamples again."""
        self._cached = None

    def _apply(self, *args, **kwargs) -> GridAdapter:
        """Drop the held field on any ``.to()`` / ``.cuda()`` / dtype change.

        The cache is a plain attribute, so ``nn.Module`` never moves it. A field held
        from before the move would be handed to a chain that has left it behind.
        """
        self.clear_cache()
        return super()._apply(*args, **kwargs)

    @property
    def is_stochastic(self) -> bool:
        return False

    def _resampled(
        self: GridAdapter,
        complex_amplitude: ComplexAmplitude,
        resolution: tuple[int, int],
        pixel_size: torch.Tensor,
        gain: float = 1.0,
    ) -> ComplexAmplitude:
        return ComplexAmplitude.from_tensor(
            data=fft_resample(complex_amplitude.as_tensor(), resolution) * gain,
            wavelength=complex_amplitude.wavelength,
            pixel_size=pixel_size,
        )

    def forward(
        self: GridAdapter, complex_amplitude: ComplexAmplitude
    ) -> ComplexAmplitude:
        if self.factor == 1:
            return complex_amplitude
        if self.cache and self._cached is not None:
            return self._cached

        resampled = self._resampled(
            complex_amplitude, self.resolution_out, self.pixel_size_out
        )
        if self.cache:
            self._cached = resampled
        return resampled

    def adjoint(
        self: GridAdapter, complex_amplitude: ComplexAmplitude
    ) -> ComplexAmplitude:
        """Resample an output-plane field back onto the input grid."""
        if self.factor == 1:
            return complex_amplitude
        return self._resampled(
            complex_amplitude,
            self.resolution_in,
            self.pixel_size_in,
            gain=float(self.factor) ** 2,
        )
