"""Adapters wrapping slmsuite ``Camera`` / ``SLM`` devices in the native device
interface (see ``Camera`` / ``SLM`` templates in :mod:`hologradpy.hardware.camera` and
:mod:`hologradpy.hardware.slm`).

Geometry, units and ROI are converted here. Every other attribute or method is delegated
to the wrapped device unchanged. These adapters are registered onto the backend-agnostic
dispatch (:mod:`hologradpy.hardware.as_native`) by this package's ``__init__``.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from slmsuite.hardware.cameras.camera import Camera as SLMSuiteCamera
from slmsuite.hardware.slms.slm import SLM as SLMSuiteSLM

from ...roi import ROI
from ..camera import Camera, CameraOrientation
from ..slm import SLM
from .conversions import (
    pixel_size_from_pitch_um,
    roi_from_woi,
    roi_to_woi,
    wavelength_from_wav_um,
)


class SLMSuiteCameraAdapter(Camera):
    """A real slmsuite camera behind the HoloGradPy-native camera interface.

    Geometry / units / ROI are converted here, at the single boundary. The native
    geometry / exposure / capture members delegate to the wrapped camera, converting
    where the conventions differ. The template methods (``autoexpose``,
    ``get_averaged_image``, ``get_spatial_grid``, ``flush``) are
    inherited from :class:`~hologradpy.hardware.camera.Camera` and run on those
    delegated members, so a real camera and the simulator autoexpose identically. Any
    other attribute is delegated unchanged via ``__getattr__``.
    """

    def __init__(self, camera: SLMSuiteCamera) -> None:
        self._camera = camera

    @property
    def pixel_size(self) -> NDArray[np.float64]:
        """Pixel pitch ``(y, x)`` in metres."""
        return pixel_size_from_pitch_um(self._camera.pitch_um)

    @property
    def max_pixel_value(self) -> int:
        """The largest count a pixel can report.

        slmsuite states the level count as ``bitresolution``, so the step from a count
        of levels to the largest one is made here.
        """
        return int(self._camera.bitresolution) - 1

    @property
    def exposure_bounds(self) -> tuple[float, float] | None:
        """The ``(min, max)`` exposure time in seconds, or ``None`` if unbounded."""
        bounds = self._camera.exposure_bounds_s
        return None if bounds is None else (float(bounds[0]), float(bounds[1]))

    @property
    def roi(self) -> ROI:
        """The current region of interest.

        A device that has not been given a window yet reports none, which means the
        whole sensor.
        """
        woi = getattr(self._camera, "woi", None)
        if woi is None:
            height, width = self._camera.shape
            return ROI(0, 0, int(height), int(width))
        return roi_from_woi(woi)

    def set_roi(self, roi: ROI | None) -> None:
        """Set the region of interest (``None`` resets to the full sensor)."""
        self._camera.set_woi(None if roi is None else roi_to_woi(roi))

    def set_orientation(self, orientation: CameraOrientation) -> None:
        """Remount the sensor, reorienting every frame the wrapped camera returns."""
        current = self.orientation
        if current is None:
            raise NotImplementedError(
                f"{type(self._camera).__name__} applies a frame transform that is not "
                "one of the eight orientations, so the shape it would display cannot "
                "be worked out. Set its transform directly."
            )
        # default_shape names the displayed frame, so undo the current quarter turn to
        # get back to the sensor before applying the new one.
        sensor = tuple(int(size) for size in self._camera.default_shape)
        if current.swaps_axes():
            sensor = (sensor[1], sensor[0])
        if orientation.swaps_axes():
            sensor = (sensor[1], sensor[0])

        self._camera.transform = orientation.transformation()
        self._camera.default_shape = sensor
        # set_woi(None) is what brings the wrapped camera's shape back in line.
        self.set_roi(None)

    def get_exposure(self) -> float:
        """The current exposure time in seconds."""
        return float(self._camera.get_exposure())

    def set_exposure(self, exposure_s: float) -> None:
        """Set the exposure time in seconds."""
        self._camera.set_exposure(exposure_s)

    def get_image(
        self, exposure_s: float | None = None, averaging: int = 1
    ) -> NDArray:
        """Capture a frame (``exposure_s`` sets the exposure first, ``averaging``
        sums that many frames)."""
        if exposure_s is not None:
            self._camera.set_exposure(exposure_s)
        return self._camera.get_image(averaging=averaging)

    def __getattr__(self, name: str):
        # Delegate anything not overridden above to the wrapped slmsuite camera. Guard
        # ``_camera`` so a half-built instance (copy / unpickle, before __init__) raises
        # AttributeError instead of recursing.
        if name == "_camera":
            raise AttributeError(name)
        return getattr(self._camera, name)


class SLMSuiteSLMAdapter(SLM):
    """A real slmsuite SLM behind the HoloGradPy-native SLM interface."""

    def __init__(self, slm: SLMSuiteSLM) -> None:
        self._slm = slm

    @property
    def pixel_size(self) -> NDArray[np.float64]:
        """Pixel pitch ``(y, x)`` in metres."""
        return pixel_size_from_pitch_um(self._slm.pitch_um)

    @property
    def resolution(self) -> tuple[int, int]:
        """SLM resolution ``(height, width)`` in pixels."""
        return (int(self._slm.shape[0]), int(self._slm.shape[1]))

    @property
    def wavelength(self) -> float:
        """Design wavelength in metres."""
        return wavelength_from_wav_um(self._slm.wav_um)

    @property
    def bitdepth(self) -> int | None:
        """Bits per pixel, from the wrapped SLM when it says."""
        return getattr(self._slm, "bitdepth", None)

    def set_levels(self, levels) -> None:
        """Display gray levels directly on the wrapped slmsuite SLM."""
        levels = np.asarray(levels)
        if not np.issubdtype(levels.dtype, np.integer):
            raise TypeError(
                f"set_levels takes integer gray levels, got {levels.dtype}. An SLM "
                "displays whole levels, and slmsuite reads a float array as a phase in "
                "radians instead."
            )
        self._slm.set_phase(levels)

    def __getattr__(self, name: str):
        # Delegate anything not overridden above to the wrapped slmsuite SLM.
        if name == "_slm":
            raise AttributeError(name)
        return getattr(self._slm, name)
