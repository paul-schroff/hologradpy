"""Camera-exposure helpers for the calibrators.

The general exposure operations are
:meth:`~hologradpy.hardware.camera.Camera.get_averaged_image` and
:meth:`~hologradpy.hardware.camera.Camera.autoexpose`. The spot search here depends on
the calibration-level spot detection, so it is kept with the calibrators.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from ..hardware import Camera, as_camera
from ..roi import ROI

from .spot_detection import detect_spot


def capture_autoexposed_frame(
    camera: Camera, roi: ROI, set_fraction: float
) -> NDArray[np.float64]:
    """One frame, captured after autoexposing on ``roi``
    (:meth:`~hologradpy.hardware.camera.Camera.autoexpose`). The camera's exposure,
    gain and region of interest are put back afterwards.

    Args:
        camera: The camera to capture from.
        roi: The region the autoexposure measures.
        set_fraction: The fraction of full scale the brightest pixel of ``roi`` is
            exposed to.

    Returns:
        NDArray: The frame.
    """
    with camera.preserve_exposure_gain_and_roi():
        camera.autoexpose(set_fraction=set_fraction, roi=roi)
        return np.asarray(camera.get_image(), dtype=float)


def expose_until_spot(
    camera: Camera,
    spot_radius: float,
    *,
    max_steps: int = 4,
    dark_threshold_fraction: float = 0.05,
    saturation_step_fraction: float = 0.05,
) -> NDArray | None:
    """Capture at the current tilt and step the exposure until a spot is detected.

    The equivalent exposure
    (:meth:`~hologradpy.hardware.camera.Camera.get_equivalent_exposure`) is multiplied
    by ``saturation_step_fraction`` after an overexposed frame and divided by it after
    a dark frame. The peak leaves out the excluded pixels
    (:attr:`~hologradpy.hardware.camera.Camera.excluded_pixels`). Each new exposure is
    set through :meth:`~hologradpy.hardware.camera.Camera.set_equivalent_exposure`, so
    the gain rises beyond the gain onset. A new exposure is set only when another
    capture follows, so the camera is left at the exposure of the last capture.

    Args:
        camera: The camera to capture from, or a driver that
            :func:`~hologradpy.hardware.as_native.as_camera` wraps.
        spot_radius: The diffraction-limited focal-spot radius (1/e^2 intensity) in
            metres.
        max_steps: The largest number of captures. Defaults to 4.
        dark_threshold_fraction: A frame counts as dark when its peak lies below this
            fraction of full scale. Defaults to 0.05.
        saturation_step_fraction: The factor of each exposure step. Defaults to 0.05.

    Returns:
        NDArray | None: The frame in which
        :func:`~hologradpy.calibration.spot_detection.detect_spot` found a spot, or None
        when a frame is exposed within range but holds no spot, when the exposure is at
        a search bound and has to pass it, or when the captures run out.
    """
    camera = as_camera(camera)
    full_scale = float(camera.max_pixel_value)
    for step in range(max_steps):
        image = np.asarray(camera.get_image())
        if detect_spot(image, spot_radius, camera):
            return image
        peak = float(image[~camera.excluded_mask(camera.roi)].max())
        exposure = camera.get_equivalent_exposure()
        if peak >= camera.saturation_level:
            desired = exposure * saturation_step_fraction
        elif peak < dark_threshold_fraction * full_scale:
            desired = exposure / saturation_step_fraction
        else:
            return None  # Exposed within range, and no spot on the frame.
        if step == max_steps - 1:
            return None
        camera.set_equivalent_exposure(desired)
        if camera.get_equivalent_exposure() == exposure:
            return None  # At a search bound, which the exposure has to pass.
    return None
