"""What a fitted pixel-crosstalk calibration is, on disk."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
from numpy.typing import NDArray

from ....serialization import SaveableRecord, record_type
from ....visualizer import VisualizationData


@record_type("pixel_crosstalk_calibration")
@dataclass
class PixelCrosstalkCalibrationData(SaveableRecord):
    """A fitted crosstalk model."""

    timestamp: datetime
    name: str
    model: str
    upscale_factor: int
    extent: int
    kernel: NDArray
    parameters: dict = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)
    visualization_data: VisualizationData | None = None

    @property
    def central_pixel_weight(self) -> float:
        """The fraction of the kernel inside the pixel it belongs to."""
        kernel = np.asarray(self.kernel)
        factor = int(self.upscale_factor)
        start = (kernel.shape[0] - factor) // 2
        centre = kernel[start : start + factor, start : start + factor]
        return float(centre.sum() / kernel.sum())
