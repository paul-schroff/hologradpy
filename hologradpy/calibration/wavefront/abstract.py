from __future__ import annotations
from abc import abstractmethod
from datetime import datetime
from dataclasses import dataclass, field




from ...optics.complex_amplitude import ComplexAmplitude
from ...serialization import SaveableRecord, record_type
from ...visualizer import VisualizationData

from ..abstract import CalibratorBase


@record_type("wavefront_calibration")
@dataclass
class WavefrontCalibrationData(SaveableRecord):
    timestamp: datetime
    name: str
    complex_amplitude: ComplexAmplitude
    metadata: dict = field(default_factory=dict)
    visualization_data: VisualizationData | None = None

    # save / load come from SaveableRecord.


class WavefrontCalibratorBase(CalibratorBase):
    """Base for calibrators that measure the amplitude and phase at the SLM."""

    @abstractmethod
    def calibrate(self, *args, **kwargs) -> WavefrontCalibrationData:
        """Run the calibration and return the measured SLM-plane wavefront.

        The arguments are specific to each calibrator, since the measurement
        strategies have little in common. Only the return type is part of the
        contract.
        """
