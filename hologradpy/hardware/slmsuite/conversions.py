"""slmsuite to native unit conversions.

slmsuite exposes geometry as ``pitch_um`` ``(x, y)`` in micrometres and wavelengths as
``wav_um`` in micrometres. These helpers convert them to the native SI units and
``(y, x)`` order.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
from numpy.typing import NDArray


def pixel_size_from_pitch_um(
    pitch_um: Sequence[float] | NDArray | None,
) -> NDArray[np.float64]:
    """``pitch_um`` (x, y) um -> ``pixel_size`` (y, x) m.

    Raises:
        ValueError: ``pitch_um`` is None, which slmsuite stores for a device opened
            without a pixel pitch.
    """
    if pitch_um is None:
        raise ValueError(
            "The device does not state its pixel pitch. Set pitch_um = (x_um, y_um) "
            "on it."
        )
    return np.asarray(pitch_um, dtype=np.float64)[::-1] * 1e-6


def wavelength_from_wav_um(wav_um: float) -> float:
    """``wav_um`` [um] -> HoloGradPy ``wavelength`` [m]."""
    return float(wav_um) * 1e-6
