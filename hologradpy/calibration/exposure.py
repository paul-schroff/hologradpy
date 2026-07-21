"""Spot-seeking camera-exposure helper for the calibrators.

The general camera-exposure operations (frame averaging and peak-driven exposure
search) now live on :class:`~hologradpy.hardware.camera.Camera` itself, as
:meth:`~hologradpy.hardware.camera.Camera.get_averaged_image` and
:meth:`~hologradpy.hardware.camera.Camera.autoexpose`. What remains here is the
spot-aware search, which depends on the calibration-level spot detection.
"""

import numpy as np
from numpy.typing import NDArray

from ..hardware import Camera

from .spot_detection import detect_spot


def expose_until_spot(
    camera: Camera,
    spot_radius: float,
    *,
    max_steps: int = 4,
    dark_threshold_fraction: float = 0.05,
    saturation_step_fraction: float = 0.05,
) -> NDArray | None:
    """Capture at the current tilt. If no spot is detected, jump the exposure once and
    retry, up to ``max_steps`` times.

    Steps the exposure down when the frame is saturated and up when it is dark. Returns
    the frame in which :func:`~hologradpy.calibration.spot_detection.detect_spot` found
    a spot, or ``None`` when the frame is well exposed but holds no spot (or the search
    is exhausted).
    """
    full_scale = float(camera.adu_levels)
    bounds = camera.exposure_bounds
    max_exposure_s = float(bounds[1]) if bounds is not None else 1.0
    exposure = float(camera.get_exposure())
    for _ in range(max_steps):
        image = np.asarray(camera.get_image())
        if detect_spot(image, spot_radius, camera):
            return image
        peak = float(image.max())
        if peak >= full_scale - 1:  # Sensor saturated, decrease exposure
            exposure *= saturation_step_fraction
        elif (
            peak < dark_threshold_fraction * full_scale
            and exposure < max_exposure_s
        ):
            exposure = min(exposure / saturation_step_fraction, max_exposure_s)
        else:
            return None  # well exposed but no spot found
        camera.set_exposure(exposure)
    return None
