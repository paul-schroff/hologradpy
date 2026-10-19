"""Adapters wrapping slmsuite ``Camera`` / ``SLM`` devices in the native device
interface (see ``Camera`` / ``SLM`` templates in :mod:`hologradpy.hardware.camera` and
:mod:`hologradpy.hardware.slm`).

Geometry and units are converted here, and the camera adapter crops its frames to the
region of interest in software. A driver setting outside the native interface is set on
the wrapped device. These adapters are registered onto the backend-agnostic dispatch
(:mod:`hologradpy.hardware.as_native`) by this package's ``__init__``.
"""

from __future__ import annotations

import math
import operator
import warnings
from collections.abc import Callable
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray

from slmsuite.hardware.cameras.camera import Camera as SLMSuiteCamera
from slmsuite.hardware.slms.slm import SLM as SLMSuiteSLM

from ...phase_levels import LinearResponse, PhaseResponse
from ...roi import ROI
from ..camera import Camera, CameraOrientation
from ..slm import SLM
from .conversions import pixel_size_from_pitch_um, wavelength_from_wav_um


def _delegated_property(wrapped: str, key: str, default: Any = None) -> property:
    """A property whose value is kept on the wrapped device under ``key``.

    Every adapter of one device reads and writes the same value, since a consumer wraps
    the device it is handed in an adapter of its own. The device holds values only,
    never an adapter, so dropping the last reference to it closes it at once.

    Args:
        wrapped: The adapter attribute that holds the wrapped device.
        key: The attribute name the value is stored under on the device.
        default: The value read before one is set. Every device shares it, so it is
            immutable.

    Example:
        The camera adapter keeps its region of interest on the slmsuite camera, so two
        adapters of one camera read the same region::

            class SLMSuiteCameraAdapter(Camera):
                _roi = _delegated_property("_camera", "_hologradpy_roi")

            first = SLMSuiteCameraAdapter(raw_camera)
            first._roi = ROI(0, 0, 10, 10)  # sets raw_camera._hologradpy_roi
            second = SLMSuiteCameraAdapter(raw_camera)
            second._roi  # ROI(0, 0, 10, 10), read from raw_camera
    """

    def read(adapter: object) -> Any:
        return getattr(getattr(adapter, wrapped), key, default)

    def write(adapter: object, value: Any) -> None:
        setattr(getattr(adapter, wrapped), key, value)

    return property(read, write)


class SLMSuiteCameraAdapter(Camera):
    """A real slmsuite camera behind the HoloGradPy-native camera interface.

    :meth:`get_image` takes single untransformed frames from the wrapped camera, so
    slmsuite's own ``averaging`` and ``hdr`` settings do not apply. The region of
    interest is applied in software on the displayed frame, so the wrapped camera reads
    out its whole sensor.

    The region of interest, the excluded pixels, :attr:`frame_timeout` and
    :attr:`overexposed` are kept on the wrapped camera, so every adapter of one device
    shares them.
    """

    # The device holds the region of interest, the excluded pixels, the frame timeout
    # and whether the last frame was overexposed, so every adapter of it shares them.
    _roi = _delegated_property("_camera", "_hologradpy_roi")
    _excluded_pixels = _delegated_property("_camera", "_hologradpy_excluded_pixels")
    _overexposed = _delegated_property("_camera", "_hologradpy_overexposed", False)
    #: The seconds to wait for each frame beyond its exposure, 1 s unless set.
    frame_timeout = _delegated_property("_camera", "_hologradpy_frame_timeout", 1.0)

    def __init__(self, camera: SLMSuiteCamera) -> None:
        self._camera = camera

    @property
    def name(self) -> str:
        """The wrapped camera's name."""
        return str(self._camera.name)

    @property
    def transform(self) -> Callable[[NDArray], NDArray]:
        """The wrapped camera's frame transform, from the raw sensor frame to the
        displayed frame.
        """
        return self._camera.transform

    @property
    def pixel_size(self) -> NDArray[np.float64]:
        """Pixel pitch ``(y, x)`` of the displayed frame in metres.

        A quarter turn of the :attr:`orientation` exchanges the sensor's axes in the
        displayed frame, so the pitch is exchanged with them.

        Raises:
            ValueError: The wrapped camera does not state its pixel pitch.
        """
        pixel_size = pixel_size_from_pitch_um(self._camera.pitch_um)
        orientation = self.orientation
        if orientation is not None and orientation.swaps_axes():
            return pixel_size[::-1].copy()
        return pixel_size

    @property
    def max_pixel_value(self) -> int:
        """The largest count of a pixel in one frame, ``2 ** bitdepth - 1``.

        slmsuite's ``bitresolution`` grows with its ``averaging`` setting. The frames of
        this adapter do not use that setting, so the ceiling is read from ``bitdepth``.
        """
        return 2 ** int(self._camera.bitdepth) - 1

    @property
    def exposure_bounds(self) -> tuple[float, float] | None:
        """The ``(min, max)`` exposure time in seconds, from the wrapped camera's
        ``exposure_bounds_s``, or None when it does not state them.
        """
        bounds = self._camera.exposure_bounds_s
        return None if bounds is None else (float(bounds[0]), float(bounds[1]))

    @property
    def sensor_resolution(self) -> tuple[int, int]:
        """The whole sensor's ``(height, width)`` in the displayed frame, whatever the
        region of interest.

        It is the wrapped camera's ``default_shape``, which slmsuite gives in displayed
        axes.
        """
        height, width = self._camera.default_shape
        return (int(height), int(width))

    @property
    def roi(self) -> ROI:
        """The region of interest in displayed ``(row, col)`` coordinates, or the whole
        frame when none is set.
        """
        roi = self._roi
        return ROI(0, 0, *self.sensor_resolution) if roi is None else roi

    def _set_roi(self, roi: ROI | None) -> None:
        """Crop every frame to ``roi``, or return to the whole frame with None."""
        self._roi = roi

    def _set_orientation(
        self, orientation: CameraOrientation, raw_shape: tuple[int, int]
    ) -> None:
        """Set the wrapped camera's transform, and its ``shape`` and ``default_shape``
        to the displayed frame, which exchanges height and width under a quarter turn.
        """
        shown = (
            (raw_shape[1], raw_shape[0])
            if orientation.swaps_axes()
            else (raw_shape[0], raw_shape[1])
        )
        self._camera.transform = orientation.transformation()
        self._camera.shape = shown
        self._camera.default_shape = shown

    def get_exposure(self) -> float:
        """The current exposure time in seconds."""
        return float(self._camera.get_exposure())

    def _set_exposure(self, exposure: float) -> None:
        """Set the exposure time in seconds on the wrapped camera."""
        self._camera.set_exposure(exposure)

    def _prepare_capture(self) -> None:
        """Drop the pending frames of the wrapped camera with the driver's ``flush``.

        Every frame of the next :meth:`get_image` call is therefore exposed after the
        call. slmsuite's default ``flush`` captures and discards two frames. The ThorCam
        driver drains the pending frames of its SDK without triggering.
        """
        self._camera.flush()

    def _get_image(self) -> NDArray:
        """One frame from the wrapped camera, reoriented with its transform and
        cropped to :attr:`roi`.

        Raises:
            TimeoutError: The wrapped camera returned no frame in any attempt.
            RuntimeError: The reoriented frame does not match the whole-frame shape
                stated by the wrapped camera.
        """
        shown = self._camera.transform(self._capture_frame())
        expected = self.sensor_resolution
        if tuple(np.shape(shown)) != expected:
            raise RuntimeError(
                f"'{self._camera.name}' returned a {tuple(np.shape(shown))} frame, and "
                f"its whole frame is {expected}. Binning or a readout window set on "
                "the driver directly is not followed by this adapter, so reset them "
                "with set_binning() and set_woi(None) on the driver."
            )
        return self.roi.crop(shown)

    def _capture_frame(self) -> NDArray:
        """One frame from the wrapped camera, before the transform and the crop.

        slmsuite hands back None for a frame that did not arrive in time. A missing
        frame is captured again, up to the wrapped camera's ``capture_attempts``. Each
        attempt waits :attr:`frame_timeout` beyond the exposure.

        Raises:
            TimeoutError: No attempt returned a frame.
        """
        name = self._camera.name
        attempts = max(1, int(self._camera.capture_attempts))
        timeout = float(self.frame_timeout)
        for attempt in range(attempts):
            frame = self._camera.get_image(
                timeout_s=timeout, transform=False, hdr=False, averaging=False
            )
            if frame is not None:
                if attempt > 0:
                    warnings.warn(
                        f"'{name}' returned no frame {attempt} time(s) before one "
                        "arrived.",
                        stacklevel=3,
                    )
                return np.asarray(frame)
        raise TimeoutError(
            f"'{name}' returned no frame in {attempts} attempt(s), each waiting "
            f"{timeout:.3g} s beyond the exposure. Check the trigger profile and the "
            "cable, and that no other program holds the camera."
        )

    def close(self, *args: Any, **kwargs: Any) -> None:
        """Close the wrapped camera, passing on the driver's own arguments."""
        self._camera.close(*args, **kwargs)


class SLMSuiteSLMAdapter(SLM):
    """A real slmsuite SLM behind the HoloGradPy-native SLM interface.

    Levels reach the SLM through slmsuite's integer path, so slmsuite's own
    ``phase_correct`` and ``source["phase"]`` are not applied. A vendor correction is
    loaded with :meth:`load_vendor_correction`. Every write waits :attr:`settle_time`
    regardless of the wrapped SLM's ``settle`` flag, and zero skips the wait.

    The phase response is the one loaded with :meth:`load_phase_response`, or else a
    straight line reaching ``wav_design_um / wav_um`` cycles at full scale. For this
    nominal response, :attr:`full_scale_cycles` is the reciprocal of slmsuite's
    ``phase_scaling``. ``wav_um`` is the operating wavelength. slmsuite defaults it
    to 1 um, so it is given when the device is opened.

    :meth:`set_resolution` makes the SLM the top-left part of the wrapped SLM. This
    suits an SLM driven as a screen in a display mode larger than the SLM.

    The loaded corrections, the loaded response and the set resolution are kept on the
    wrapped SLM, so every adapter of one device shares them. A driver setting outside
    the native interface is set on the wrapped SLM.
    """

    # Loaded corrections, the loaded response and the set resolution live on the
    # device, so every adapter of it shares them.
    _phase_correction = _delegated_property("_slm", "_hologradpy_phase_correction")
    _vendor_correction = _delegated_property("_slm", "_hologradpy_vendor_correction")
    _phase_response = _delegated_property("_slm", "_hologradpy_phase_response")
    _resolution = _delegated_property("_slm", "_hologradpy_resolution")

    def __init__(self, slm: SLMSuiteSLM) -> None:
        self._slm = slm

    @property
    def name(self) -> str:
        """The wrapped SLM's name."""
        return str(self._slm.name)

    def close(self) -> None:
        """Close the wrapped SLM."""
        self._slm.close()

    @property
    def pixel_size(self) -> NDArray[np.float64]:
        """Pixel pitch ``(y, x)`` in metres."""
        return pixel_size_from_pitch_um(self._slm.pitch_um)

    @property
    def resolution(self) -> tuple[int, int]:
        """SLM resolution ``(height, width)`` in pixels.

        The resolution given to :meth:`set_resolution`, or else the wrapped SLM's whole
        shape.
        """
        if self._resolution is None:
            return self._wrapped_resolution
        return self._resolution

    @property
    def _wrapped_resolution(self) -> tuple[int, int]:
        """The wrapped slmsuite SLM's own resolution ``(height, width)`` in pixels."""
        return (int(self._slm.shape[0]), int(self._slm.shape[1]))

    def set_resolution(self, resolution: tuple[int, int] | None) -> None:
        """Use the top-left ``resolution`` pixels of the wrapped SLM's frame as the SLM,
        and pad every pattern with level zero to the frame.

        This suits an SLM driven as a screen in a display mode larger than the SLM. One
        example is a Hamamatsu X13138 (1024 x 1272) in a 1024 x 1280 mode. None returns
        to the whole frame.

        The resolution is kept on the wrapped SLM, so every adapter of it uses the same
        one. It is set before a correction is loaded, a model is built or the SLM is
        recorded, since each of those takes the resolution at the time.

        Args:
            resolution: The SLM resolution ``(height, width)`` in pixels, or None for
                the wrapped SLM's whole frame.

        Raises:
            TypeError: ``resolution`` holds a number that is not a whole number.
            ValueError: ``resolution`` is not two positive sizes, exceeds the frame, or
                differs from the shape of a loaded phase or vendor correction.
        """
        wrapped = self._wrapped_resolution
        if resolution is None:
            checked = None
        else:
            checked = tuple(operator.index(size) for size in resolution)
            if len(checked) != 2 or min(checked) < 1:
                raise ValueError(
                    "A resolution is two positive sizes (height, width) in pixels, got "
                    f"{tuple(resolution)}."
                )
            if checked[0] > wrapped[0] or checked[1] > wrapped[1]:
                raise ValueError(
                    f"The resolution {checked} does not fit in the wrapped "
                    f"{type(self._slm).__name__}, which is {wrapped}."
                )

        new_resolution = wrapped if checked is None else checked
        for kind, correction in (
            ("phase", self._phase_correction),
            ("vendor", self._vendor_correction),
        ):
            if correction is not None and tuple(correction.shape) != new_resolution:
                raise ValueError(
                    f"The loaded {kind} correction is {tuple(correction.shape)}, which "
                    f"does not match the resolution {new_resolution}. Set the "
                    "resolution before loading a correction."
                )
        self._resolution = checked

    @property
    def wavelength(self) -> float:
        """Operating wavelength in metres.

        The displayed phase is meant for this wavelength.
        """
        return wavelength_from_wav_um(self._slm.wav_um)

    @property
    def bitdepth(self) -> int | None:
        """Bits per pixel, from the wrapped SLM when it says."""
        return getattr(self._slm, "bitdepth", None)

    @property
    def _nominal_phase_response(self) -> PhaseResponse | None:
        """A straight line reaching ``wav_design_um / wav_um`` cycles at full scale.

        Its ``full_scale_cycles`` is the reciprocal of slmsuite's ``phase_scaling``.
        """
        bitdepth = self.bitdepth
        if bitdepth is None:
            return None
        wav_um = float(self._slm.wav_um)
        wav_design_um = float(getattr(self._slm, "wav_design_um", wav_um))
        return LinearResponse(
            bitdepth=int(bitdepth), full_scale_cycles=wav_design_um / wav_um
        )

    @property
    def settle_time(self) -> float:
        """The settling time of the SLM in seconds, held by the wrapped SLM as
        ``settle_time_s``.

        Each write waits this long, and zero skips the wait. Setting a negative or
        non-finite value raises ``ValueError`` and leaves the wait as it was.
        """
        return float(self._slm.settle_time_s)

    @settle_time.setter
    def settle_time(self, seconds: float) -> None:
        seconds = float(seconds)
        # slmsuite sleeps after the write, so a bad value there raises only once the
        # SLM has already changed. The value is therefore checked before it reaches
        # slmsuite.
        if not (math.isfinite(seconds) and seconds >= 0):
            raise ValueError(
                "settle_time is a wait in seconds, so it has to be a finite number "
                f"of zero or more. Got {seconds}."
            )
        self._slm.settle_time_s = seconds

    @property
    def display(self) -> NDArray:
        """The displayed gray levels.

        This is the wrapped SLM's own buffer, cropped to :attr:`resolution` after
        :meth:`set_resolution`. The next write overwrites it in place, so it is copied
        to be kept.
        """
        if self._resolution is None:
            return self._slm.display
        height, width = self._resolution
        return self._slm.display[:height, :width]

    def set_levels(self, levels: NDArray | torch.Tensor) -> None:
        """Display gray levels on the wrapped slmsuite SLM, then wait
        :attr:`settle_time` for the SLM to settle.

        After :meth:`set_resolution`, the levels fill the top left of the wrapped SLM,
        and the rest of it shows level zero.

        Args:
            levels: Whole gray levels at the SLM resolution, each from 0 to
                ``2**bitdepth - 1``. They are shown as given, without either
                correction.

        Raises:
            TypeError: ``levels`` are not integers.
            ValueError: ``levels`` are not the SLM's shape, or lie outside the range
                from 0 to ``2**bitdepth - 1``.
        """
        levels = self._checked_levels(levels)
        wrapped = self._wrapped_resolution
        if levels.shape != wrapped:
            height, width = levels.shape
            padded = np.zeros(wrapped, dtype=levels.dtype)
            padded[:height, :width] = levels
            levels = padded
        self._slm.set_phase(levels, settle=True)
