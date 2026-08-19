"""slmsuite to native conversions: units and window of interest.

slmsuite exposes geometry as ``pitch_um`` ``(x, y)`` in micrometres and wavelengths as
``wav_um`` in micrometres, and describes a camera readout window as
``(x0, width, y0, height)``. These helpers are the single place those device
conventions meet the native SI, ``(y, x)`` and :class:`ROI` representations.
"""

from __future__ import annotations

from typing import Sequence, TypeVar

import numpy as np
import torch
from numpy.typing import NDArray

from array_api_compat import array_namespace

from ...roi import ROI

ArrayLike = TypeVar("ArrayLike", torch.Tensor, NDArray)


def pixel_size_from_pitch_um(
    pitch_um: Sequence[float] | NDArray,
) -> NDArray[np.float64]:
    """``pitch_um`` (x, y) um -> ``pixel_size`` (y, x) m."""
    return np.asarray(pitch_um, dtype=np.float64)[::-1] * 1e-6


def pitch_um_from_pixel_size(pixel_size: ArrayLike) -> ArrayLike:
    """``pixel_size`` (y, x) m -> ``pitch_um`` (x, y) um.

    Backend-agnostic (numpy or torch) so a caller holding a torch tensor gets a tensor
    back with its dtype preserved. The inverse :func:`pixel_size_from_pitch_um` stays
    numpy-only because it reads slmsuite's plain-sequence ``pitch_um``.
    """
    xp = array_namespace(pixel_size)
    return xp.flip(pixel_size, axis=-1) * 1e6


def wavelength_from_wav_um(wav_um: float) -> float:
    """``wav_um`` [um] -> HoloGradPy ``wavelength`` [m]."""
    return float(wav_um) * 1e-6


def wav_um_from_wavelength(wavelength: float) -> float:
    """HoloGradPy ``wavelength`` [m] -> slmsuite ``wav_um`` [um]."""
    return float(wavelength) * 1e6


def roi_from_woi(woi: tuple[int, int, int, int]) -> ROI:
    """From an slmsuite readout window ``(x0, width, y0, height)``."""
    x0, width, y0, height = woi
    return ROI(int(y0), int(x0), int(height), int(width))


def roi_to_woi(roi: ROI) -> tuple[int, int, int, int]:
    """To an slmsuite readout window ``(x0, width, y0, height)``."""
    return (roi.left_column, roi.width, roi.top_row, roi.height)
