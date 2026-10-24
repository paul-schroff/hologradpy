"""What a zeroth-order capture and a zeroth-order calibration are, on disk."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import datetime

import numpy as np
from numpy.typing import NDArray

from ..camera_mapping import CameraMapping
from ...hardware.camera import CameraData
from ...hardware.slm import SLMData
from ...serialization import SaveableRecord, record_type
from ...visualizer import VisualizationData

#: The replica with the beam's phase taken out, which focuses to the tightest spot.
FLAT_REPLICA = 0
#: The replica with the phase of the backplane, the beam's phase without the SLM's
#: vendor correction, which focuses as the light at the wraps does.
BACKPLANE_REPLICA = 1


@record_type("zeroth_order_capture")
@dataclass(frozen=True)
class ZerothOrderCaptureData(SaveableRecord):
    """Everything about a captured zeroth-order dataset except the frames themselves.

    Every probe is captured twice, back to back. The first frame is exposed on the
    zeroth order's disc and measures it, and the second is exposed on the whole sensor
    and registers the probe on its grating's spot. The frames of the store alternate in
    that order, through the probes in the order :attr:`probes` lists them.

    Attributes:
        timestamp: When the capture started.
        slm_data: The SLM that displayed the probes.
        camera_data: The camera that captured them, at the exposure of the first
            frames.
        camera_mapping: The map from camera pixels to the model's output plane, which
            places the zeroth order on the sensor.
        region: The zeroth order's disc, True on the sensor pixels it covers.
        grating_tilts: The ``(x, y)`` position in metres, from the zeroth order, of the
            spot of every grating, ``(G, 2)``.
        probes: One row of integers per probe, ``(P, 4)``. The row holds the grating,
            the replica (:data:`FLAT_REPLICA` or :data:`BACKPLANE_REPLICA`), the piston
            step and the replica's amplitude in multiples of :attr:`probe_amplitude`.
        probe_amplitude: The amplitude of the replica, as a multiple of the hologram's.
        number_of_pistons: The piston steps of a full cycle of the replica.
        metadata: The exposures, and anything else the capture measured.
    """

    timestamp: datetime
    slm_data: SLMData
    camera_data: CameraData
    camera_mapping: CameraMapping
    region: NDArray[np.bool_]
    grating_tilts: NDArray
    probes: NDArray
    probe_amplitude: float
    number_of_pistons: int
    metadata: dict = field(default_factory=dict)


@record_type("zeroth_order_calibration")
@dataclass
class ZerothOrderCalibrationData(SaveableRecord):
    """A measured zeroth order, as
    :class:`~hologradpy.optics.modules.zeroth_order.ZerothOrderField` takes it.

    Attributes:
        timestamp: When the calibration was fitted.
        name: What produced it.
        focal_length: The focal length of the Fourier lens in metres.
        kernel: The image-plane kernel of the field no hologram changes, complex,
            ``(rows, columns)``.
        kernel_pixel_size: The ``(y, x)`` pitch of the kernel in metres, the intensity
            Nyquist pitch of the SLM.
        wrap_weight: The field scattered at one wrapped edge, as a multiple of the
            beam's field on the pixel that counts the edge.
        region: The zeroth order's disc the probes measured, True on the sensor pixels
            it covers.
        added_phase: The phase in radians the SLM added to every probe for its vendor
            correction, at the SLM's resolution, or None without one. The wraps of a
            hologram are those of its sum with this phase.
        metadata: The scale of the frames, the residual of the fit, the floor, the
            registration and the settings of the capture.
        visualization_data: The panels of the fit, dropped by :meth:`lean`.
    """

    timestamp: datetime
    name: str
    focal_length: float
    kernel: NDArray
    kernel_pixel_size: tuple[float, float]
    wrap_weight: complex
    region: NDArray[np.bool_]
    added_phase: NDArray | None = None
    metadata: dict = field(default_factory=dict)
    visualization_data: VisualizationData | None = None

    def lean(self) -> ZerothOrderCalibrationData:
        """Return a copy without the visualization data, ready to save."""
        return replace(self, visualization_data=None)
