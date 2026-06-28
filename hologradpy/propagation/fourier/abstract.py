from __future__ import annotations

import torch
from torch import Tensor
import torch.nn as nn


class FourierBase(nn.Module):
    """Base class for 2D Fourier transforms -- a linear map between two conjugate
    sample sets, evaluated at a set of k-space sample points.

    The transform is described in **unit-free** terms only: the input sample
    counts ``resolution``, the output sample counts ``resolution_out``, and
    :attr:`frequencies` -- the angular frequencies in rad/sample at which the DFT
    is evaluated, laid out as ``(2, n_points)`` with ``[0] = omega_x`` and
    ``[1] = omega_y``. It makes **no assumption about the physical domains** (it
    is not specifically "spatial -> frequency"): it works equally for the forward
    DFT, the inverse, an optical Fourier lens (spatial -> focal plane), etc. The
    physical interpretation -- which domain is which, and the spacing relationship
    (``2*pi/(N*dx)`` for a plain DFT, ``lambda*f/(N*dx)`` for a Fourier lens) --
    belongs to the wrapping layer (e.g. a ``FourierLens`` ``OpticsModule`` that
    carries the focal length and the field geometry).

    Subclasses implement :meth:`forward` and :meth:`adjoint` (the conjugate
    transpose, not necessarily the inverse). :attr:`is_gridded` is ``True`` when
    the transform exploits a regular grid of sample points (FFT, chirp-z) and
    ``False`` for the general non-uniform method (NUFFT); generic code and fast
    paths can branch on it.
    """

    def __init__(
        self,
        resolution: tuple[int, int],
        frequencies: Tensor,
        is_gridded: bool,
        resolution_out: tuple[int, int] | None = None,
        device: torch.device = "cpu",
    ) -> None:
        super().__init__()
        self.resolution = resolution
        self.resolution_out = resolution if resolution_out is None else resolution_out
        self.frequencies = frequencies
        self.is_gridded = is_gridded
        self.device = device

    def forward(self, input: Tensor) -> Tensor:
        raise NotImplementedError(
            "Subclasses of FourierBase must implement forward()."
        )

    def adjoint(self, input: Tensor) -> Tensor:
        raise NotImplementedError(
            "Subclasses of FourierBase must implement adjoint()."
        )
