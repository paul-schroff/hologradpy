"""Backgrounds to subtract from camera frames.

Subtracting a background is an optional step of the speckle calibrations and of camera
feedback. A background is the part of a frame that carries no signal with the beam 
blocked.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray

#: A background in counts, as one level for every pixel, a whole-sensor frame, or a
#: function of the exposure in seconds returning either.
Background = float | NDArray | Callable[[float], float | NDArray]


def background_at(
    background: Background, exposure: float | None, frame_shape: tuple[int, int]
) -> float | NDArray[np.float64]:
    """The background in counts of a frame taken at ``exposure``.

    Args:
        background: One level for every pixel, a whole-sensor frame, or a function of
            the exposure in seconds returning either.
        exposure: The exposure of the frame in seconds, or None where it is unknown.
        frame_shape: The ``(height, width)`` of the frame to subtract the background
            from.

    Returns:
        float | NDArray[np.float64]: The level, or a frame of ``frame_shape``.

    Raises:
        ValueError: The background is a function and ``exposure`` is None, it holds a
            count that is not finite, or it is a frame of another shape.
    """
    if callable(background):
        if exposure is None:
            raise ValueError(
                "The background is a function of the exposure, and the frames record "
                "no exposure. Pass a level or a whole-sensor frame."
            )
        background = background(float(exposure))
    counts = np.asarray(background, dtype=np.float64)
    if not np.all(np.isfinite(counts)):
        raise ValueError("A background holds finite counts only.")
    if counts.ndim == 0:
        return float(counts)
    if counts.shape != tuple(frame_shape):
        raise ValueError(
            f"The background frame has shape {counts.shape}, and the frames it is "
            f"taken from are {tuple(frame_shape)}. Measure it on the whole sensor, "
            "with the camera oriented as it is now."
        )
    return counts
