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
from ..camera import Camera
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
    def resolution(self) -> tuple[int, int]:
        """Sensor resolution ``(height, width)`` in pixels."""
        return (int(self._camera.shape[0]), int(self._camera.shape[1]))

    @property
    def adu_levels(self) -> int:
        """Number of digital levels (``2 ** bitdepth``). The max pixel value is one
        less."""
        return int(self._camera.bitresolution)

    @property
    def exposure_bounds(self) -> tuple[float, float] | None:
        """The ``(min, max)`` exposure time in seconds, or ``None`` if unbounded."""
        bounds = self._camera.exposure_bounds_s
        return None if bounds is None else (float(bounds[0]), float(bounds[1]))

    @property
    def roi(self) -> ROI:
        """The current region of interest."""
        return roi_from_woi(self._camera.woi)

    def set_roi(self, roi: ROI | None) -> None:
        """Set the region of interest (``None`` resets to the full sensor)."""
        self._camera.set_woi(None if roi is None else roi_to_woi(roi))

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

    def set_phase(self, phase) -> None:
        """Display the desired optical phase (delegated to the wrapped slmsuite SLM)."""
        self._slm.set_phase(phase)

    def __getattr__(self, name: str):
        # Delegate anything not overridden above to the wrapped slmsuite SLM. Guard
        # ``_slm`` so a half-built instance (copy / unpickle, before __init__) raises
        # AttributeError instead of recursing.
        if name == "_slm":
            raise AttributeError(name)
        return getattr(self._slm, name)
