from __future__ import annotations

import math
from typing import Any

import torch
from torch import Tensor

from .abstract import FourierBase

try:
    import torchkbnufft as tkbn
except ImportError:
    # Only this transform needs it, so the rest of the package imports without it and
    # the constructor below raises with an instruction instead.
    tkbn = None

NUFFT_INSTALL_HINT = (
    "KbNufftPartialAffine needs torchkbnufft, which is an optional dependency. "
    "Install it with: pip install hologradpy[nufft]"
)


def _as_per_wavelength(
    value: tuple[float, float] | Tensor, device: torch.device
) -> Tensor:
    """Normalize a magnification / shift argument to a ``(n_wl, 2)`` float
    tensor in ``(x, y)`` order. Accepts a 2-tuple ``(x, y)`` (treated as a
    single wavelength) or an already-batched ``(n_wl, 2)`` tensor/sequence.
    """
    tensor = torch.as_tensor(value, dtype=torch.float32, device=device)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor


def _build_rotated_trajectory(
    grid_size: tuple[int, int],
    resolution_out: tuple[int, int],
    magnification: Tensor,
    shift: Tensor,
    angle: float,
    device: torch.device,
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
            device=device, dtype=torch.float32,
        )  # (length_out,)
        return indices.unsqueeze(0) * step.unsqueeze(-1) + offset.unsqueeze(-1)

    omega_x = axis(grid_width, out_width, magnification[:, 0], shift[:, 0])
    omega_y = axis(grid_height, out_height, magnification[:, 1], shift[:, 1])

    grid_x = omega_x[:, None, :].expand(number_of_wavelengths, out_height, out_width)
    grid_y = omega_y[:, :, None].expand(number_of_wavelengths, out_height, out_width)

    cos_angle, sin_angle = math.cos(angle), math.sin(angle)
    sample_x = grid_x * cos_angle - grid_y * sin_angle
    sample_y = grid_x * sin_angle + grid_y * cos_angle

    return torch.stack(
        (
            sample_x.reshape(number_of_wavelengths, -1),
            sample_y.reshape(number_of_wavelengths, -1),
        ),
        dim=0,
    )


class KbNufftPartialAffine(FourierBase):
    """A partial affine of the spectrum via the Kaiser-Bessel NUFFT (``torchkbnufft``).

    The sample points come from :func:`_build_rotated_trajectory` -- the same
    scaled + shifted window the chirp-z zoom uses, with an extra rotation that,
    because the NUFFT evaluates arbitrary k-space points, is simply a rotation of
    the trajectory (no shears). ``magnification`` and ``shift`` may carry a
    leading wavelength axis ``(n_wl, 2)`` so the trajectory differs per
    wavelength; ``angle`` is a scalar in radians. The points are evaluated by the
    general (interpolating) non-uniform FFT, so ``is_gridded`` is ``False`` and
    the output amplitude carries the KbNufft normalization (not the exact DFT).

    ``forward``/``adjoint`` operate on a flattened ``(n_images, n_wl, H, W)``
    field (the leading batch already collapsed by the caller); the per-wavelength
    trajectory is tiled across the image axis. ``adjoint`` is the conjugate
    transpose (``KbNufftAdjoint``) on the same trajectory.
    """

    def __init__(
        self,
        resolution: tuple[int, int],
        resolution_out: tuple[int, int],
        magnification: tuple[float, float] | Tensor,
        shift: tuple[float, float] | Tensor = (0.0, 0.0),
        angle: float = 0.0,
        grid_size: tuple[int, int] | None = None,
        dtype: torch.dtype = torch.float32,
        device: torch.device = "cpu",
        norm: str | None = None,
        **nufft_kwargs: Any,
    ) -> None:
        if tkbn is None:
            raise ImportError(NUFFT_INSTALL_HINT)

        if grid_size is None:
            grid_size = resolution

        magnification = _as_per_wavelength(magnification, device)
        shift = _as_per_wavelength(shift, device)
        frequencies = _build_rotated_trajectory(
            grid_size, resolution_out, magnification, shift, angle, device
        )

        super().__init__(
            resolution,
            frequencies=frequencies,
            is_gridded=False,
            resolution_out=resolution_out,
            device=device,
        )

        self.angle = angle
        self.norm = norm
        self._kbnufft = tkbn.KbNufft(
            im_size=list(resolution),
            grid_size=list(grid_size),
            dtype=dtype,
            device=device,
            **nufft_kwargs,
        ).to(device)
        self._kbnufft_adjoint = tkbn.KbNufftAdjoint(
            im_size=list(resolution),
            grid_size=list(grid_size),
            dtype=dtype,
            device=device,
            **nufft_kwargs,
        ).to(device)

    def _batched_trajectory(
        self, number_of_images: int, number_of_wavelengths: int
    ) -> Tensor:
        """Tile the stored ``(2, n_wl, hw)`` trajectory across the image batch to
        ``(n_images * n_wl, 2, hw)``, matching the row-major ``(image,
        wavelength)`` flattening of the field.
        """
        trajectory = self.frequencies.moveaxis(0, 1)  # (n_wl, 2, hw): [:, 0] = x
        # torchkbnufft maps omega[0] onto the first image axis (rows / height) and
        # omega[1] onto columns / width, so hand it (omega_y, omega_x). Without
        # this swap the focal field comes out transposed relative to a plain FFT.
        trajectory = trajectory.flip(1)
        trajectory = trajectory.unsqueeze(0).expand(number_of_images, -1, -1, -1)
        return trajectory.reshape(number_of_images * number_of_wavelengths, 2, -1)

    def forward(self, input: Tensor) -> Tensor:
        """``input``: ``(n_images, n_wl, H, W)`` -> ``(n_images, n_wl, H_out,
        W_out)``.
        """
        number_of_images, number_of_wavelengths = input.shape[0], input.shape[1]
        field = input.reshape(
            number_of_images * number_of_wavelengths, 1, *self.resolution
        )
        trajectory = self._batched_trajectory(number_of_images, number_of_wavelengths)
        output = self._kbnufft(field, trajectory, norm=self.norm)
        return output.reshape(
            number_of_images, number_of_wavelengths, *self.resolution_out
        )

    def adjoint(self, samples: Tensor) -> Tensor:
        """``samples``: ``(n_images, n_wl, H_out, W_out)`` -> ``(n_images, n_wl,
        H, W)``.
        """
        number_of_images, number_of_wavelengths = samples.shape[0], samples.shape[1]
        flat = samples.reshape(number_of_images * number_of_wavelengths, 1, -1)
        trajectory = self._batched_trajectory(number_of_images, number_of_wavelengths)
        image = self._kbnufft_adjoint(flat, trajectory, norm=self.norm)
        return image.reshape(
            number_of_images, number_of_wavelengths, *self.resolution
        )
