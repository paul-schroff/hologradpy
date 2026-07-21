from __future__ import annotations

import torch
from torch import Tensor

from .abstract import FourierBase
from ...grids import get_zoom_frequency_grid
from .shear_rotation import shear_rotate


def _next_pow2(n: int) -> int:
    return 1 << (n - 1).bit_length()


def _czt_forward_last(field: Tensor, omega: Tensor) -> Tensor:
    """1D chirp-z along the last axis: evaluate the *centred* DFT
    ``X_k = sum_n field[n] * exp(-i * omega_k * (n - N//2))`` at the uniformly spaced
    sample frequencies ``omega`` (rad/sample), exactly.
    """
    length = field.shape[-1]
    points = omega.shape[0]
    device = field.device

    step = omega[1] - omega[0]
    omega0 = omega[0]
    n = torch.arange(length, device=device, dtype=torch.float32)
    k = torch.arange(points, device=device, dtype=torch.float32)
    m = torch.arange(-(length - 1), points, device=device, dtype=torch.float32)

    pre_chirp = torch.exp(-1j * (omega0 * n + 0.5 * step * n * n))
    kernel = torch.exp(0.5j * step * m * m)

    convolution_length = _next_pow2(2 * length + points)
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
    n = torch.arange(length, device=device, dtype=torch.float32)
    k = torch.arange(points, device=device, dtype=torch.float32)
    m = torch.arange(-(points - 1), length, device=device, dtype=torch.float32)

    samples = samples * torch.exp(-1j * omega * (length // 2))

    pre_chirp = torch.exp(0.5j * step * k * k)
    kernel = torch.exp(-0.5j * step * m * m)

    convolution_length = _next_pow2(2 * points + length)
    convolution = torch.fft.ifft(
        torch.fft.fft(samples * pre_chirp, n=convolution_length, dim=-1)
        * torch.fft.fft(kernel, n=convolution_length),
        dim=-1,
    )
    convolution = convolution[..., points - 1: points - 1 + length]

    post_chirp = torch.exp(0.5j * step * n * n)
    return convolution * post_chirp * torch.exp(1j * omega0 * n)


class ChirpZZoom(FourierBase):
    """Exact scale + shift + rotate Fourier zoom.

    The scale and shift come from the chirp-z (Bluestein) transform applied separably
    along each axis at the :func:`get_zoom_frequency_grid` sample points; the (optional)
    rotation is a 3-shear FFT rotation of the input, using ``F(R k) = FT(f o R)(k)``.
    Unlike the KbNufft this is the *exact* DFT at those points (``is_gridded=True``), so
    its power is correct -- no interpolation/normalization fudge. Fully differentiable.
    """

    def __init__(
        self,
        resolution: tuple[int, int],
        resolution_out: tuple[int, int],
        magnification: tuple[float, float],
        shift: tuple[float, float] = (0.0, 0.0),
        angle: float = 0.0,
        device: torch.device = "cpu",
    ) -> None:
        omega_x, omega_y = get_zoom_frequency_grid(
            resolution, resolution_out, magnification, shift, device
        )
        grid_x, grid_y = torch.meshgrid(omega_x, omega_y, indexing="xy")
        frequencies = torch.stack((grid_x.flatten(), grid_y.flatten()), dim=0)

        super().__init__(
            resolution,
            frequencies=frequencies,
            is_gridded=True,
            resolution_out=resolution_out,
            device=device,
        )
        self.omega_x = omega_x
        self.omega_y = omega_y
        self.angle = angle

    def forward(self, input: Tensor) -> Tensor:
        field = input
        if self.angle != 0.0:
            field = shear_rotate(field, self.angle)
        # Separable chirp-z: along x (last axis), then along y (via transpose).
        field = _czt_forward_last(field, self.omega_x)
        field = _czt_forward_last(
            field.transpose(-1, -2), self.omega_y
        ).transpose(-1, -2)
        return field

    def adjoint(self, samples: Tensor) -> Tensor:
        field = _czt_adjoint_last(
            samples.transpose(-1, -2), self.omega_y, self.resolution[0]
        ).transpose(-1, -2)
        field = _czt_adjoint_last(field, self.omega_x, self.resolution[1])
        if self.angle != 0.0:
            field = shear_rotate(field, -self.angle)
        return field
