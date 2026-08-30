from __future__ import annotations

import warnings

import torch
from torch import Tensor

from ....fourier_transforms import SemiAnalyticalFourierTransform, fft_2d
from ....grids import get_frequency_grid, get_pixel_grid
from ....utils import to_canvas

from ..abstract import OpticsModule, capture_init
from ...complex_amplitude import ComplexAmplitude, broadcast_wavelength_operand


class AngularSpectrumSAFT(OpticsModule):
    """Angular spectrum propagation onto a plane sampled as finely as asked for.

    The IFFT in the standard angular spectrum method is replaced with an inverse
    semi-analytical Fourier transform, which allows the output plane to be sampled at a 
    smaller pitch than the input.
    """

    @capture_init
    def __init__(
        self,
        propagation_distance: float,
        pixel_size_out: Tensor | tuple[float, float] | None = None,
        resolution_out: tuple[int, int] | None = None,
        padded_resolution: tuple[int, int] | None = None,
    ) -> None:
        """
        Args:
            propagation_distance: How far to propagate, in metres.
            pixel_size_out: Output pixel size ``(height, width)`` in metres. Defaults
                to the Fresnel pitch, where the residual is smoothest.
            resolution_out: Output resolution ``(height, width)``. Defaults to the
                input's.
            padded_resolution: Resolution to transform on. Defaults to twice the input.
        """
        super().__init__(
            pixel_size_out=pixel_size_out, resolution_out=resolution_out
        )
        self.propagation_distance = propagation_distance
        self._padded_resolution_init = padded_resolution
        self._warned = False

    def lazy_init(self, complex_amplitude: ComplexAmplitude) -> None:
        resolution_in = complex_amplitude.resolution
        if self._padded_resolution_init is None:
            self._padded_resolution = tuple(2 * length for length in resolution_in)
        else:
            self._padded_resolution = self._padded_resolution_init

        pitch = complex_amplitude.pixel_size.reshape(-1, 2)[0].to(torch.float64)

        if self._pixel_size_out_init is None:
            self.set_output_geometry(
                pixel_size=(float(pitch[0]), float(pitch[1]))
            )

        pixel_out = self.pixel_size_out.reshape(-1, 2)[0].to(torch.float64)

        self._curvature = tuple(
            -torch.pi
            * float(pixel_out[axis])
            / (self._padded_resolution[axis] * float(pitch[axis]))
            for axis in range(2)
        )

        self._transform = SemiAnalyticalFourierTransform(
            self._padded_resolution,
            (self._curvature[1], 0.0, self._curvature[0]),
            inverse=True,
            device=complex_amplitude.device,
        )
        self.register_buffer(
            "residual_transfer", self._residual_transfer(complex_amplitude)
        )

    def _residual_transfer(self, complex_amplitude: ComplexAmplitude) -> Tensor:
        """The transfer function with the extracted quadratic divided out."""
        pitch = complex_amplitude.pixel_size.reshape(-1, 2)[0].to(torch.float64)
        frequency_x, frequency_y = get_frequency_grid(
            self._padded_resolution, pitch, complex_amplitude.device
        )
        wavenumber = complex_amplitude.wavenumber.reshape(-1, 1, 1).to(torch.float64)

        argument = (
            wavenumber**2
            - frequency_x.unsqueeze(0) ** 2
            - frequency_y.unsqueeze(0) ** 2
        )
        transfer = torch.exp(
            1j * float(self.propagation_distance) * torch.sqrt(argument + 0j)
        )

        index_x, index_y = get_pixel_grid(
            self._padded_resolution, complex_amplitude.device, dtype=torch.float64
        )
        extracted = (
            self._curvature[1] * index_x**2 + self._curvature[0] * index_y**2
        )
        return transfer * torch.exp(-1j * extracted).unsqueeze(0)

    def sampling_margin(self) -> tuple[float, float]:
        """How hard the residual transfer function is to sample, per axis.

        Returns:
            tuple[float, float]: The margin along ``x`` and along ``y``.
        """
        phase = torch.angle(self.residual_transfer[0])

        step_x = torch.angle(torch.exp(1j * torch.diff(phase, dim=-1)))
        step_y = torch.angle(torch.exp(1j * torch.diff(phase, dim=-2)))

        lattice_x, lattice_y = self._transform.sampling_margin()
        return (
            max(float(step_x.abs().max() / torch.pi), lattice_x),
            max(float(step_y.abs().max() / torch.pi), lattice_y),
        )

    def _warn_if_aliased(self) -> None:
        if self._warned:
            return
        self._warned = True
        margin_x, margin_y = self.sampling_margin()
        if max(margin_x, margin_y) <= 1.0:
            return
        pitch = self.pixel_size_out.reshape(-1, 2)[0]
        lattice = max(self._transform.sampling_margin())
        cause = (
            "the output lattice reaches past one period, so the plane comes back "
            "tiled with copies of itself"
            if lattice > 1.0
            else "the transfer function outruns its samples"
        )
        warnings.warn(
            f"This propagation is not resolved on the grid it was given: {cause}. "
            f"Sampling margin ({margin_x:.2f}, {margin_y:.2f}), which has to be at "
            f"most 1. Propagating {float(self.propagation_distance) * 1e3:.3f} mm "
            f"onto a pitch of {float(pitch[0]) * 1e6:.3f} x "
            f"{float(pitch[1]) * 1e6:.3f} um asks for more than it holds. Ask for a "
            "finer pixel_size_out, which is the direction this propagator is for, or "
            "pad further.",
            RuntimeWarning,
            stacklevel=3,
        )

    def forward(self, complex_amplitude: ComplexAmplitude) -> ComplexAmplitude:
        """Propagate, landing on the output pitch this was built for."""
        self._warn_if_aliased()

        transfer = broadcast_wavelength_operand(
            self.residual_transfer, complex_amplitude.ndim
        )
        padded = to_canvas(complex_amplitude, self._padded_resolution)
        spectrum = fft_2d(padded)

        out = self._transform(spectrum * transfer) / (
            self._padded_resolution[0] * self._padded_resolution[1]
        )

        return to_canvas(out, self.resolution_out).with_geometry(
            wavelength=complex_amplitude.wavelength,
            pixel_size=self.pixel_size_out,
        )
