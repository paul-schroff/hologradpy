from __future__ import annotations

import math
from typing import Any

import torch
from torch import Tensor

from .abstract import FourierBase

try:
    from pytorch_finufft.functional import finufft_type1, finufft_type2
except ImportError:
    # Only this transform needs it, so the rest of the package imports without it and
    # the constructor below raises with an instruction instead.
    finufft_type1 = finufft_type2 = None

NUFFT_INSTALL_HINT = (
    "NUFFTPartialAffine needs pytorch-finufft, which is an optional dependency. "
    "Install it with: pip install hologradpy[nufft], or hologradpy[nufft-cuda] to "
    "add the GPU transform."
)

FORWARD_SIGN = -1
ADJOINT_SIGN = +1
CENTERED_MODES = 0
RESERVED_KWARGS = ("isign", "modeord")


def _as_per_wavelength(
    value: tuple[float, float] | Tensor, device: torch.device, dtype: torch.dtype
) -> Tensor:
    """Normalize a magnification / shift argument to a ``(n_wl, 2)`` float
    tensor in ``(x, y)`` order. Accepts a 2-tuple ``(x, y)`` (treated as a
    single wavelength) or an already-batched ``(n_wl, 2)`` tensor/sequence.
    """
    tensor = torch.as_tensor(value, dtype=dtype, device=device)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor


def _build_rotated_trajectory(
    grid_size: tuple[int, int],
    resolution_out: tuple[int, int],
    magnification: Tensor,
    shift: Tensor,
    angle: float | Tensor,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    """Per-wavelength k-space sample points of a scaled + shifted + rotated zoom
    window, in rad/sample, laid out as ``(2, n_wl, hw)`` (``[0] = kx``,
    ``[1] = ky``).

    The un-rotated window matches :func:`get_zoom_frequency_grid`: along each
    axis the native bin spacing ``2*pi / grid_size`` is divided by the
    magnification and offset by ``shift``. ``magnification`` / ``shift`` are
    ``(n_wl, 2)`` in ``(x, y)`` order; ``grid_size`` / ``resolution_out`` are
    ``(height, width)``. The rotation ``R(angle)`` mixes the two axes:
    ``kx = gx*cos - gy*sin``, ``ky = gx*sin + gy*cos``.
    """
    number_of_wavelengths = max(magnification.shape[0], shift.shape[0])
    magnification = magnification.expand(number_of_wavelengths, 2)
    shift = shift.expand(number_of_wavelengths, 2)

    grid_height, grid_width = grid_size
    out_height, out_width = resolution_out

    def axis(length_in: int, length_out: int, mag: Tensor, offset: Tensor) -> Tensor:
        step = (2 * torch.pi / length_in) / mag  # (n_wl,)
        indices = torch.arange(
            -(length_out // 2), length_out - length_out // 2,
            device=device, dtype=dtype,
        )  # (length_out,)
        return indices.unsqueeze(0) * step.unsqueeze(-1) + offset.unsqueeze(-1)

    omega_x = axis(grid_width, out_width, magnification[:, 0], shift[:, 0])
    omega_y = axis(grid_height, out_height, magnification[:, 1], shift[:, 1])

    grid_x = omega_x[:, None, :].expand(number_of_wavelengths, out_height, out_width)
    grid_y = omega_y[:, :, None].expand(number_of_wavelengths, out_height, out_width)

    angle = torch.as_tensor(angle, dtype=dtype, device=device)
    cos_angle, sin_angle = torch.cos(angle), torch.sin(angle)
    sample_x = grid_x * cos_angle - grid_y * sin_angle
    sample_y = grid_x * sin_angle + grid_y * cos_angle

    return torch.stack(
        (
            sample_x.reshape(number_of_wavelengths, -1),
            sample_y.reshape(number_of_wavelengths, -1),
        ),
        dim=0,
    )


class NUFFTPartialAffine(FourierBase):
    """A partial affine of the spectrum via the non-uniform FFT (``FINUFFT``).

    The sample points come from :func:`_build_rotated_trajectory` -- the same
    scaled + shifted window the chirp-z zoom uses, with an extra rotation that,
    because the NUFFT evaluates arbitrary k-space points, is simply a rotation of
    the trajectory (no shears). ``magnification`` and ``shift`` may carry a
    leading wavelength axis ``(n_wl, 2)`` so the trajectory differs per
    wavelength; ``angle`` is a scalar in radians. The points are evaluated by the
    general (interpolating) non-uniform FFT, so ``is_gridded`` is ``False`` and
    the amplitude is the exact DFT sum to within the requested ``eps``, which
    FINUFFT defaults to ``1e-6``.

    ``forward``/``adjoint`` operate on a flattened ``(n_images, n_wl, H, W)``
    field (the leading batch already collapsed by the caller). FINUFFT takes one
    trajectory per call and broadcasts it over leading batch dimensions, so the
    image axis rides along for free and only the wavelength axis is a loop.
    """

    def __init__(
        self,
        resolution: tuple[int, int],
        resolution_out: tuple[int, int],
        magnification: tuple[float, float] | Tensor,
        shift: tuple[float, float] | Tensor = (0.0, 0.0),
        angle: float | Tensor = 0.0,
        grid_size: tuple[int, int] | None = None,
        dtype: torch.dtype = torch.float32,
        device: torch.device = "cpu",
        norm: str | None = None,
        **nufft_kwargs: Any,
    ) -> None:
        if finufft_type1 is None:
            raise ImportError(NUFFT_INSTALL_HINT)

        reserved = [name for name in RESERVED_KWARGS if name in nufft_kwargs]
        if reserved:
            raise TypeError(
                f"{type(self).__name__} sets {', '.join(reserved)} itself; passing "
                "it would silently change which direction is the forward transform."
            )

        if grid_size is None:
            grid_size = resolution

        magnification = _as_per_wavelength(magnification, device, dtype)
        shift = _as_per_wavelength(shift, device, dtype)
        frequencies = _build_rotated_trajectory(
            grid_size, resolution_out, magnification, shift, angle, device, dtype
        )

        super().__init__(
            resolution,
            frequencies=frequencies,
            is_gridded=False,
            resolution_out=resolution_out,
            device=device,
        )

        self.angle = angle
        self.grid_size = grid_size
        self.norm = norm
        self.nufft_kwargs = nufft_kwargs
        self._normalization = (
            1.0 / math.sqrt(grid_size[0] * grid_size[1]) if norm == "ortho" else 1.0
        )

    def _points(self, index: int, values: Tensor) -> Tensor:
        """The trajectory for one wavelength, as ``(2, hw)`` laid out for FINUFFT."""
        trajectory = self.frequencies
        trajectory = trajectory[:, index if trajectory.shape[1] > 1 else 0]
        trajectory = trajectory.flip(0)
        wrapped = torch.remainder(trajectory + torch.pi, 2 * torch.pi) - torch.pi
        return wrapped.to(device=values.device, dtype=values.real.dtype).contiguous()

    def forward(self, input: Tensor) -> Tensor:
        """``input``: ``(n_images, n_wl, H, W)`` -> ``(n_images, n_wl, H_out,
        W_out)``.
        """
        number_of_images, number_of_wavelengths = input.shape[0], input.shape[1]
        samples = [
            finufft_type2(
                self._points(index, input),
                input[:, index].contiguous(),
                isign=FORWARD_SIGN,
                modeord=CENTERED_MODES,
                **self.nufft_kwargs,
            ).reshape(number_of_images, *self.resolution_out)
            for index in range(number_of_wavelengths)
        ]
        return torch.stack(samples, dim=1) * self._normalization

    def adjoint(self, samples: Tensor) -> Tensor:
        """``samples``: ``(n_images, n_wl, H_out, W_out)`` -> ``(n_images, n_wl,
        H, W)``.
        """
        number_of_images, number_of_wavelengths = samples.shape[0], samples.shape[1]
        flat = samples.reshape(number_of_images, number_of_wavelengths, -1)
        images = [
            finufft_type1(
                self._points(index, flat),
                flat[:, index].contiguous(),
                tuple(self.resolution),
                isign=ADJOINT_SIGN,
                modeord=CENTERED_MODES,
                **self.nufft_kwargs,
            )
            for index in range(number_of_wavelengths)
        ]
        return torch.stack(images, dim=1) * self._normalization
