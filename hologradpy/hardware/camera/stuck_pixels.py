"""Stuck-pixel detection from frames captured across an exposure sweep.

A working pixel counts in proportion to the exposure, and a stuck pixel keeps its count.
:func:`stuck_pixel_mask` compares frames taken at different exposures and marks the
pixels that do not respond. :meth:`~hologradpy.hardware.camera.Camera.find_stuck_pixels`
captures a sweep for it, and :meth:`~hologradpy.hardware.camera.Camera.autoexpose`
passes it the frames of its own search.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import binary_erosion, label

# The relative tolerance of the response test in stuck-pixel detection. A working
# pixel rises by the ratio of two exposures within this tolerance.
STUCK_PIXEL_RESPONSE_TOLERANCE = 0.2


def stuck_pixel_mask(
    frames: NDArray | Sequence[NDArray],
    exposures: Sequence[float] | NDArray,
    max_pixel_value: int,
    saturation_level: float,
    *,
    lower_threshold: float = 0.1,
    tolerance: float = STUCK_PIXEL_RESPONSE_TOLERANCE,
    blob_min_size: int = 4,
) -> NDArray[np.bool_]:
    """True at the hot and dead pixels of frames captured at different exposures.

    Each frame is compared with the frame at the next longer exposure. A pixel is
    tested when it carries signal and stays below full scale at the longer exposure. A
    working pixel then rises by the exposure ratio within ``tolerance``. A pixel that
    climbs to full scale from below also responds. A stuck pixel does neither. A pixel
    stuck above ``lower_threshold`` of full scale is flagged anywhere. A pixel stuck
    lower (dead) is flagged only where its neighbours respond, since elsewhere it
    cannot be told from an unilluminated pixel.

    The comparison uses only exposures far enough apart to tell a working pixel from a
    stuck one (:func:`_select_distinct_exposures`), so a repeated exposure is dropped.
    A region that counts as overexposure (``blob_min_size``) is not flagged. A
    ``UserWarning`` then reports that the camera is overexposed.

    Args:
        frames: A stack ``(n, height, width)`` of whole frames, or a list of them,
            such as the sweep of
            :meth:`~hologradpy.hardware.camera.Camera.find_stuck_pixels`.
        exposures: The ``n`` exposures in seconds. Frames taken at different gains are
            given their equivalent exposures
            (:meth:`~hologradpy.hardware.camera.Camera.get_equivalent_exposure`).
        max_pixel_value: The largest count a pixel can report, which is full scale.
        saturation_level: The count from which a pixel counts as overexposed.
        lower_threshold: The fraction of full scale above which a pixel carries
            signal.
        tolerance: The relative tolerance of the response test.
        blob_min_size: A region saturated in every frame counts as overexposure when
            it holds at least this many pixels.

    Returns:
        NDArray: True at the stuck pixels, in the ``(height, width)`` shape of a
        frame.

    Raises:
        ValueError: The frames are not a stack of at least two, the exposures do not
            match them, or fewer than two exposures lie far enough apart.
    """
    frames = np.asarray(frames, dtype=float)
    exposures = np.asarray(exposures, dtype=float)
    if frames.ndim != 3 or frames.shape[0] < 2:
        raise ValueError(
            "frames must be a stack of at least two (height, width) images."
        )
    if exposures.shape != (frames.shape[0],):
        raise ValueError("exposures must give one exposure time for each frame.")

    # The indices of the kept frames, in increasing order of exposure.
    distinct = _select_distinct_exposures(exposures, tolerance)
    if len(distinct) < 2:
        raise ValueError(
            "Stuck-pixel detection needs frames at two or more exposures, each "
            f"longer than the one before by more than a factor 1 / (1 - "
            f"{tolerance}). Got the exposures {exposures.tolist()} s."
        )
    exposures = exposures[distinct]
    frames = frames[distinct]

    full_scale = max_pixel_value
    saturation = saturation_level
    frames_min = frames.min(axis=0)
    frames_max = frames.max(axis=0)

    # Neighbouring exposures are compared on readings below saturation, so the
    # comparison holds over a wide sweep.
    responding = (frames_max >= saturation) & (frames_min < saturation)
    for shorter, longer, exposure_short, exposure_long in zip(
        frames[:-1], frames[1:], exposures[:-1], exposures[1:]
    ):
        ratio = exposure_long / exposure_short
        testable = (shorter > lower_threshold * full_scale) & (
            shorter < saturation / ratio
        )
        expected = shorter * ratio
        rose_as_expected = np.abs(longer - expected) <= tolerance * expected
        responding |= testable & rose_as_expected
    non_responding = ~responding

    # The pixels saturated in every frame.
    saturated = frames_min >= saturation

    # The connected components of the saturated pixels, and their sizes in pixels.
    components, count = label(saturated)
    sizes = np.bincount(components.ravel())

    # A component of blob_min_size pixels or more is overexposure, and a smaller one
    # is a group of hot pixels.
    overexposed_blob = np.zeros_like(saturated)
    for component in range(1, count + 1):
        if sizes[component] >= blob_min_size:
            overexposed_blob |= components == component
    if overexposed_blob.any():
        warnings.warn(
            f"Camera is overexposed: a blob of >= {blob_min_size} pixels is "
            "saturated across the whole exposure sweep.",
            stacklevel=2,
        )

    # A hot pixel is flagged in a dark corner too, since it still rails the exposure
    # there. Only pixels with signal count as responding, so noise in a dark
    # background never marks the neighbours of a dark pixel as responding.
    neighbours = np.ones((3, 3), dtype=bool)
    neighbours[1, 1] = False
    in_illuminated_region = binary_erosion(
        responding, structure=neighbours, border_value=False
    )
    return (
        non_responding
        & ~overexposed_blob
        & ((frames_min > lower_threshold * full_scale) | in_illuminated_region)
    )


def _select_distinct_exposures(
    exposures: NDArray[np.float64], tolerance: float
) -> NDArray[np.intp]:
    """The indices of the frames that stuck-pixel detection can compare.

    The shortest exposure is kept first. Each further exposure is kept when it exceeds
    the last kept exposure by more than a factor ``1 / (1 - tolerance)``. A pixel that
    does not respond at all passes the response test on a pair of exposures closer
    than that factor.

    Args:
        exposures: The exposure of each frame in seconds.
        tolerance: The relative tolerance of the response test.

    Returns:
        NDArray[np.intp]: Indices into ``exposures``, ordered by increasing exposure.
    """
    exposures = np.asarray(exposures, dtype=np.float64)
    kept: list[int] = []
    for index in np.argsort(exposures, kind="stable"):
        if not kept or exposures[index] * (1.0 - tolerance) > exposures[kept[-1]]:
            kept.append(int(index))
    return np.asarray(kept, dtype=np.intp)
