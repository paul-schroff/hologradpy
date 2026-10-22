from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import torch
from torch import Tensor

from ..utils import resolve_device
from .abstract import FourierBase
from .sampling import get_zoom_frequency_grid, rotated_zoom_trajectory

if TYPE_CHECKING:
    from ..roi import ROI

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
    tensor in ``(column, row)`` order. Accepts a 2-tuple ``(column, row)`` (treated as
    a single wavelength) or an already-batched ``(n_wl, 2)`` tensor/sequence.
    """
    tensor = torch.as_tensor(value, dtype=dtype, device=device)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor


class NUFFTPartialAffine(FourierBase):
    """A partial affine of the spectrum via the non-uniform FFT (``FINUFFT``).

    The sample points are the window of :func:`get_zoom_frequency_grid` turned by
    :func:`rotated_zoom_trajectory`, the points the chirp-z zoom samples. The NUFFT
    evaluates arbitrary k-space points, so the rotation needs no shear of the field.
    ``magnification`` and ``shift`` may carry a leading wavelength axis ``(n_wl, 2)``,
    giving one trajectory per wavelength. ``angle`` is a scalar in radians, in the pixel
    frame from +column towards +row, as for
    :class:`~hologradpy.fourier_transforms.czt.ChirpZPartialAffine`. The NUFFT
    interpolates, so the amplitude matches the exact DFT sum to within the requested
    ``eps``. FINUFFT defaults ``eps`` to ``1e-6``.

    With ``output_roi``, only that rectangle of the ``resolution_out`` grid is
    evaluated.

    ``forward`` and ``adjoint`` take a ``(n_images, n_wl, H, W)`` field, with any
    leading batch axes collapsed into ``n_images`` by the caller. FINUFFT takes one
    trajectory per call and broadcasts it over the image axis, so only the wavelength
    axis is looped over.
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
        device: torch.device | None = None,
        norm: str | None = None,
        output_roi: ROI | None = None,
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

        device = resolve_device(device)
        magnification = _as_per_wavelength(magnification, device, dtype)
        shift = _as_per_wavelength(shift, device, dtype)
        omega_x, omega_y = get_zoom_frequency_grid(
            grid_size,
            resolution_out,
            (magnification[:, 0], magnification[:, 1]),
            (shift[:, 0], shift[:, 1]),
            device,
            output_roi,
        )
        frequencies = rotated_zoom_trajectory(omega_x, omega_y, angle)
        if output_roi is not None:
            resolution_out = (output_roi.height, output_roi.width)

        super().__init__(
            resolution,
            frequencies=frequencies,
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
