from __future__ import annotations

from datetime import datetime
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ..abstract import WavefrontCalibrationData

from ...camera_mapping import CameraMapping

from ....hardware.camera import CameraData
from ....hardware.slm import SLMData
from ....serialization import SaveableRecord, record_type


@record_type("speckle_capture")
@dataclass(frozen=True)
class SpeckleCaptureData(SaveableRecord):
    """Everything about a captured speckle dataset except the samples themselves."""
    timestamp: datetime
    phase_pattern_type: str
    slm_data: SLMData
    camera_data: CameraData
    camera_mapping: CameraMapping
    roi_mask: NDArray[np.bool_]
    benchmark_calibration: WavefrontCalibrationData | None
    metadata: dict
