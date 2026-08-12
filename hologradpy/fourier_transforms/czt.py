from __future__ import annotations

import torch
from torch import Tensor

from scipy.fft import next_fast_len

from .abstract import FourierBase
from .sampling import get_zoom_frequency_grid
from .shear import fft_shear


def _convolution_length(length: int, points: int) -> int:
    return next_fast_len(length + points - 1)


def _czt_forward_last(field: Tensor, omega: Tensor) -> Tensor:
    """1D chirp-z along the last axis: evaluate the centred DFT
    ``X_k = sum_n field[n] * exp(-i * omega_k * (n - N//2))`` at the uniformly spaced
    sample frequencies ``omega`` (rad/sample), exactly.
    """
    length = field.shape[-1]
    points = omega.shape[0]
    device = field.device

    step = omega[1] - omega[0]
    omega0 = omega[0]

    dtype = omega.dtype
    n = torch.arange(length, device=device, dtype=dtype)
    k = torch.arange(points, device=device, dtype=dtype)
    m = torch.arange(-(length - 1), points, device=device, dtype=dtype)

    pre_chirp = torch.exp(-1j * (omega0 * n + 0.5 * step * n * n))
    kernel = torch.exp(0.5j * step * m * m)

    convolution_length = _convolution_length(length, points)
    convolution = torch.fft.ifft(
        torch.fft.fft(field * pre_chirp, n=convolution_length, dim=-1)
        * torch.fft.fft(kernel, n=convolution_length),
        dim=-1,
    )
    convolution = convolution[..., length - 1: length - 1 + points]

    post_chirp = torch.exp(-0.5j * step * k * k)
    centring = torch.exp(1j * omega * (length // 2))
    return convolution * post_chirp * centring


def _czt_adjoint_last(samples: Tensor, omega: Tensor, length: int) -> Tensor:
    """Conjugate transpose of :func:`_czt_forward_last`: map ``samples`` on the
    ``omega`` grid back to a ``length``-point input axis, ``x_n = sum_k samples[k] *
    exp(+i * omega_k * (n - N//2))``.
    """
    points = omega.shape[0]
    device = samples.device

    step = omega[1] - omega[0]
    omega0 = omega[0]

    dtype = omega.dtype
    n = torch.arange(length, device=device, dtype=dtype)
    k = torch.arange(points, device=device, dtype=dtype)
    m = torch.arange(-(points - 1), length, device=device, dtype=dtype)

    samples = samples * torch.exp(-1j * omega * (length // 2))

    pre_chirp = torch.exp(0.5j * step * k * k)
    kernel = torch.exp(-0.5j * step * m * m)

    convolution_length = _convolution_length(points, length)
    convolution = torch.fft.ifft(
        torch.fft.fft(samples * pre_chirp, n=convolution_length, dim=-1)
        * torch.fft.fft(kernel, n=convolution_length),
        dim=-1,
    )
    convolution = convolution[..., points - 1: points - 1 + length]

    post_chirp = torch.exp(0.5j * step * n * n)
    return convolution * post_chirp * torch.exp(1j * omega0 * n)


class ChirpZPartialAffine(FourierBase):
    """The exact DFT sampled on a partial affine of the spectrum."""

    def __init__(
        self,
        resolution: tuple[int, int],
        resolution_out: tuple[int, int],
        magnification: tuple[float, float],
        shift: tuple[float, float] = (0.0, 0.0),
        angle: float | Tensor = 0.0,
        device: torch.device = "cpu",
    ) -> None:
        omega_x, omega_y = get_zoom_frequency_grid(
            resolution, resolution_out, magnification, shift, device
        )
        rotates = torch.is_tensor(angle) or angle != 0.0
        angle = torch.as_tensor(angle, dtype=omega_x.dtype, device=omega_x.device)
        cosine, sine = torch.cos(angle), torch.sin(angle)

        super().__init__(
            resolution,
            frequencies=None,
            is_gridded=True,
            resolution_out=resolution_out,
            device=device,
        )
        self.omega_x = omega_x
        self.omega_y = omega_y
        self.angle = angle

        # The triangular part of the rotation, as two rescaled grids and one skew.
        self._omega_x = omega_x / cosine
        self._omega_y = cosine * omega_y
        self._skew = sine * omega_x

        # The shear that is left, one shift per input column.
        columns = torch.arange(resolution[1], device=device) - resolution[1] // 2
        self._shear = torch.tan(angle) * columns.to(omega_x.dtype)
        self._rotates = rotates

    def _build_frequencies(self) -> Tensor:
        cosine, sine = torch.cos(self.angle), torch.sin(self.angle)
        grid_x, grid_y = torch.meshgrid(self.omega_x, self.omega_y, indexing="xy")
        return torch.stack(
            (
                (grid_x * cosine + grid_y * sine).flatten(),
                (grid_y * cosine - grid_x * sine).flatten(),
            ),
            dim=0,
        )

    def _skew_phase(self, field: Tensor) -> Tensor:
        length = field.shape[-2]
        rows = torch.arange(length, device=field.device) - length // 2
        return torch.exp(1j * self._skew[None, :] * rows.to(self._skew.dtype)[:, None])

    def forward(self, input: Tensor) -> Tensor:
        field = input
        if self._rotates:
            field = fft_shear(field, -2, self._shear)
        # Separable chirp-z: along x (last axis), then along y (via transpose).
        field = _czt_forward_last(field, self._omega_x)
        if self._rotates:
            field = field * self._skew_phase(field)
        field = _czt_forward_last(
            field.transpose(-1, -2), self._omega_y
        ).transpose(-1, -2)
        return field

    def adjoint(self, samples: Tensor) -> Tensor:
        field = _czt_adjoint_last(
            samples.transpose(-1, -2), self._omega_y, self.resolution[0]
        ).transpose(-1, -2)
        if self._rotates:
            field = field * self._skew_phase(field).conj()
        field = _czt_adjoint_last(field, self._omega_x, self.resolution[1])
        if self._rotates:
            field = fft_shear(field, -2, -self._shear)
        return field
