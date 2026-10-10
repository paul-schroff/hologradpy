from __future__ import annotations
from abc import abstractmethod
from datetime import datetime
from dataclasses import dataclass, field




from ...optics.complex_amplitude import ComplexAmplitude
from ...serialization import SaveableRecord, record_type
from ...visualizer import VisualizationData

from ..abstract import CalibratorBase


@record_type("wavefront_calibration", version=2)
@dataclass
class WavefrontCalibrationData(SaveableRecord):
    timestamp: datetime
    name: str
    complex_amplitude: ComplexAmplitude
    metadata: dict = field(default_factory=dict)
    visualization_data: VisualizationData | None = None

    # save / load come from SaveableRecord.

    @classmethod
    def _migrate(cls, version: int, stored: dict) -> dict:
        """Convert a calibration saved while y increased with increasing row index
        (version 1).

        The field is indexed by row and column, so it stays as it is. The y of the beam
        centre ``shift_y`` and of the raster tilts changes sign, since y now increases
        with decreasing row index.
        """
        if version < 2:
            metadata = dict(stored.get("metadata") or {})
            if metadata.get("shift_y") is not None:
                metadata["shift_y"] = -metadata["shift_y"]
            for key in ("linear_phase_tilt", "lattice_phase_tilt"):
                if metadata.get(key) is not None:
                    x, y = metadata[key]
                    metadata[key] = (x, -y)
            stored["metadata"] = metadata
        return stored


class WavefrontCalibratorBase(CalibratorBase):
    """Base for calibrators that measure the amplitude and phase at the SLM."""

    @abstractmethod
    def calibrate(self, *args, **kwargs) -> WavefrontCalibrationData:
        """Run the calibration and return the measured SLM-plane wavefront.

        The arguments are specific to each calibrator, since the measurement
        strategies have little in common. Only the return type is part of the
        contract.
        """
