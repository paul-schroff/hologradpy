from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
from torch import Tensor

from scipy.fft import next_fast_len

from ..grids import centered_indices
from ..utils import resolve_device, to_canvas
from .abstract import FourierBase
from .sampling import (
    get_zoom_frequency_axes,
    get_zoom_frequency_grid,
    rotated_zoom_trajectory,
)
from .shear import fft_shear, windowed_fft_shear, windowed_shift_half_width

if TYPE_CHECKING:
    from ..roi import ROI


def _convolution_length(length: int, points: int) -> int:
    return next_fast_len(length + points - 1)


def _czt_forward_last(
    field: Tensor, omega0: Tensor, step: Tensor, points: int
) -> Tensor:
    """1D chirp-z along the last axis: evaluate the centered DFT
    ``X_k = sum_n field[n] * exp(-i * omega_k * (n - N//2))`` at the ``points``
    frequencies ``omega_k = omega0 + k * step`` (rad/sample), exactly.

    The chirp phases grow as ``step * n ** 2``, so they are built in double precision,
    from the float64 ``omega0`` and ``step`` of
    :func:`~.sampling.get_zoom_frequency_axes`. Only the chirps take the complex dtype
    of the field.
    """
    length = field.shape[-1]
    device = field.device
    chirp_dtype = torch.promote_types(field.dtype, torch.complex64)

    n = torch.arange(length, device=device, dtype=torch.float64)
    k = torch.arange(points, device=device, dtype=torch.float64)
    m = torch.arange(-(length - 1), points, device=device, dtype=torch.float64)

    pre_chirp = torch.exp(-1j * (omega0 * n + 0.5 * step * n * n)).to(chirp_dtype)
    kernel = torch.exp(0.5j * step * m * m).to(chirp_dtype)

    convolution_length = _convolution_length(length, points)
    convolution = torch.fft.ifft(
        torch.fft.fft(field * pre_chirp, n=convolution_length, dim=-1)
        * torch.fft.fft(kernel, n=convolution_length),
        dim=-1,
    )
    convolution = convolution[..., length - 1: length - 1 + points]

    # The post-chirp and the centring, as one phase.
    uniform = omega0 + step * k
    post_chirp = torch.exp(1j * (uniform * (length // 2) - 0.5 * step * k * k))
    return convolution * post_chirp.to(chirp_dtype)


def _czt_adjoint_last(
    samples: Tensor, omega0: Tensor, step: Tensor, length: int
) -> Tensor:
    """Conjugate transpose of :func:`_czt_forward_last`: map ``samples`` at the
    frequencies ``omega_k = omega0 + k * step`` back to a ``length``-point input axis,
    ``x_n = sum_k samples[k] * exp(+i * omega_k * (n - N//2))``. The chirps are built as
    in :func:`_czt_forward_last`.
    """
    points = samples.shape[-1]
    device = samples.device
    chirp_dtype = torch.promote_types(samples.dtype, torch.complex64)

    n = torch.arange(length, device=device, dtype=torch.float64)
    k = torch.arange(points, device=device, dtype=torch.float64)
    m = torch.arange(-(points - 1), length, device=device, dtype=torch.float64)

    # The centring and the pre-chirp, as one phase.
    uniform = omega0 + step * k
    pre_chirp = torch.exp(1j * (0.5 * step * k * k - uniform * (length // 2)))
    kernel = torch.exp(-0.5j * step * m * m).to(chirp_dtype)

    convolution_length = _convolution_length(points, length)
    convolution = torch.fft.ifft(
        torch.fft.fft(
            samples * pre_chirp.to(chirp_dtype), n=convolution_length, dim=-1
        )
        * torch.fft.fft(kernel, n=convolution_length),
        dim=-1,
    )
    convolution = convolution[..., points - 1: points - 1 + length]

    post_chirp = torch.exp(1j * (0.5 * step * n * n + omega0 * n))
    return convolution * post_chirp.to(chirp_dtype)


class ChirpZPartialAffine(FourierBase):
    """The exact DFT sampled on a partial affine of the spectrum.

    ``magnification`` and ``shift`` are ``(column, row)`` pairs, and the shift is the
    window offset in radians per sample that
    :func:`~.sampling.window_offset_from_pixels` gives. ``angle`` is in radians in the
    pixel frame, from +column towards +row, so a positive angle turns the image
    clockwise on a screen with row 0 at the top
    (:func:`~.sampling.rotated_zoom_trajectory`).

    With ``output_roi``, only that rectangle of the ``resolution_out`` grid is
    evaluated.
    """

    def __init__(
        self,
        resolution: tuple[int, int],
        resolution_out: tuple[int, int],
        magnification: tuple[float, float],
        shift: tuple[float, float] = (0.0, 0.0),
        angle: float | Tensor = 0.0,
        device: torch.device | None = None,
        output_roi: ROI | None = None,
    ) -> None:
        device = resolve_device(device)
        omega_x, omega_y = get_zoom_frequency_grid(
            resolution, resolution_out, magnification, shift, device, output_roi
        )
        (first_x, step_x), (first_y, step_y) = get_zoom_frequency_axes(
            resolution, resolution_out, magnification, shift, device, output_roi
        )
        if output_roi is not None:
            resolution_out = (output_roi.height, output_roi.width)
        rotates = torch.is_tensor(angle) or angle != 0.0
        angle = torch.as_tensor(angle, dtype=omega_x.dtype, device=omega_x.device)
        cosine, sine = torch.cos(angle), torch.sin(angle)

        super().__init__(
            resolution,
            frequencies=None,
            resolution_out=resolution_out,
            device=device,
        )
        self.omega_x = omega_x
        self.omega_y = omega_y
        self.angle = angle

        # The triangular part of the rotation, as two rescaled grids and one skew. The
        # grids are the (first, step) of the frequencies each chirp-z samples, in double
        # precision.
        double_cosine = torch.cos(angle.to(torch.float64))
        self._omega_x_grid = (first_x / double_cosine, step_x / double_cosine)
        self._omega_y_grid = (double_cosine * first_y, double_cosine * step_y)
        self._skew = sine * omega_x

        # The shear that is left, one shift per input column.
        columns = centered_indices(resolution[1], device)
        self._shear = torch.tan(angle) * columns.to(omega_x.dtype)
        self._rotates = rotates

        self._shear_band: tuple[float, float] | None = None
        self._shear_tolerance = float(torch.finfo(omega_y.dtype).eps)
        self._sheared_rows = resolution[0]
        if rotates:
            self._set_shear_band(resolution[0], cosine * omega_y)

    def _set_shear_band(self, rows: int, row_frequencies: Tensor) -> None:
        """Set the band of the windowed shear and the rows the field is padded to.

        The row chirp-z samples ``row_frequencies - sin * omega_x`` at every output
        sample, with ``row_frequencies`` the points ``cos * omega_y``. The padded rows
        hold the largest shift and the half-width of the windowed shift at both ends.
        """
        row_frequencies, skew = row_frequencies.detach(), self._skew.detach()
        low = float(row_frequencies.min() - skew.max())
        high = float(row_frequencies.max() - skew.min())
        if high - low >= 2 * math.pi:
            return
        self._shear_band = (low, high)
        reach = math.ceil(float(self._shear.detach().abs().max()))
        reach += windowed_shift_half_width(self._shear_band, self._shear_tolerance)
        self._sheared_rows = next_fast_len(rows + 2 * reach)

    def _shear_field(self, field: Tensor) -> Tensor:
        """Shift every column of ``field`` along the rows by its entry of the shear."""
        if self._shear_band is None:
            return fft_shear(field, -2, self._shear)
        padded = to_canvas(field, (self._sheared_rows, field.shape[-1]))
        return windowed_fft_shear(
            padded, -2, self._shear, self._shear_band, self._shear_tolerance
        )

    def _shear_field_adjoint(self, field: Tensor) -> Tensor:
        """The conjugate transpose of :meth:`_shear_field`."""
        if self._shear_band is None:
            return fft_shear(field, -2, -self._shear)
        sheared = windowed_fft_shear(
            field, -2, -self._shear, self._shear_band, self._shear_tolerance
        )
        return to_canvas(sheared, (self.resolution[0], sheared.shape[-1]))

    def _build_frequencies(self) -> Tensor:
        return rotated_zoom_trajectory(self.omega_x, self.omega_y, self.angle)

    def _skew_phase(self, field: Tensor) -> Tensor:
        rows = centered_indices(field.shape[-2], field.device)
        return torch.exp(1j * self._skew[None, :] * rows.to(self._skew.dtype)[:, None])

    def forward(self, input: Tensor) -> Tensor:
        field = input
        if self._rotates:
            field = self._shear_field(field)
        # Separable chirp-z: along x (last axis), then along y (via transpose).
        height, width = self.resolution_out
        field = _czt_forward_last(field, *self._omega_x_grid, width)
        if self._rotates:
            field = field * self._skew_phase(field)
        field = _czt_forward_last(
            field.transpose(-1, -2), *self._omega_y_grid, height
        ).transpose(-1, -2)
        return field

    def adjoint(self, samples: Tensor) -> Tensor:
        field = _czt_adjoint_last(
            samples.transpose(-1, -2), *self._omega_y_grid, self._sheared_rows
        ).transpose(-1, -2)
        if self._rotates:
            field = field * self._skew_phase(field).conj()
        field = _czt_adjoint_last(field, *self._omega_x_grid, self.resolution[1])
        if self._rotates:
            field = self._shear_field_adjoint(field)
        return field
