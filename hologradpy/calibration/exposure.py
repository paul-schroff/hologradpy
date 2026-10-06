"""Spot-seeking camera-exposure helper for the calibrators.

The general exposure operations are
:meth:`~hologradpy.hardware.camera.Camera.get_averaged_image` and
:meth:`~hologradpy.hardware.camera.Camera.autoexpose`. The search here depends on the
calibration-level spot detection, so it lives with the calibrators.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from ..hardware import Camera, as_camera

from .spot_detection import detect_spot


def expose_until_spot(
    camera: Camera,
    spot_radius: float,
    *,
    max_steps: int = 4,
    dark_threshold_fraction: float = 0.05,
    saturation_step_fraction: float = 0.05,
) -> NDArray | None:
    """Capture at the current tilt and step the exposure until a spot is detected, over
    at most ``max_steps`` captures.

    The exposure is multiplied by ``saturation_step_fraction`` after an overexposed
    frame. It is divided by this factor after a dark frame, which has its peak below
    ``dark_threshold_fraction`` of full scale. Each new exposure is calculated from the
    camera's applied exposure and clamped into
    :attr:`~hologradpy.hardware.camera.Camera.exposure_search_bounds`. A new exposure is
    set only when another capture follows, so the camera is left at the exposure of the
    last capture.

    Args:
        camera: The camera to capture from, or a driver that
            :func:`~hologradpy.hardware.as_native.as_camera` wraps.
        spot_radius: The diffraction-limited focal-spot radius (1/e^2 intensity) in
            metres.
        max_steps: The largest number of captures. Defaults to 4.
        dark_threshold_fraction: A frame counts as dark when its peak lies below this
            fraction of full scale. Defaults to 0.05.
        saturation_step_fraction: The factor applied to the exposure after an
            overexposed frame. Defaults to 0.05.

    Returns:
        NDArray | None: The frame in which
        :func:`~hologradpy.calibration.spot_detection.detect_spot` found a spot, or None
        when a frame is exposed within range but holds no spot, when the exposure is at
        a search bound and has to pass it, or when the captures run out.
    """
    camera = as_camera(camera)
    full_scale = float(camera.max_pixel_value)
    lowest, highest = (float(bound) for bound in camera.exposure_search_bounds)
    for step in range(max_steps):
        image = np.asarray(camera.get_image())
        if detect_spot(image, spot_radius, camera):
            return image
        peak = float(image.max())
        exposure = float(camera.get_exposure())
        if peak >= camera.saturation_level:
            desired = exposure * saturation_step_fraction
        elif peak < dark_threshold_fraction * full_scale:
            desired = exposure / saturation_step_fraction
        else:
            return None  # Exposed within range, and no spot on the frame.
        requested = float(np.clip(desired, lowest, highest))
        if requested == exposure or step == max_steps - 1:
            return None
        camera.set_exposure(requested)
    return None
