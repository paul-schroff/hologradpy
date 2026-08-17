"""Panels for a speckle calibration that fitted a point spread function."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .abstract import SpeckleCalibratorVisualizer, SpeckleVisualizationData

from .....visualizer import INTENSITY_CMAP, PHASE_CMAP, GridCell, Panel, PlotLayout
from .....serialization import record_type


@record_type("psf_speckle_visualization")
@dataclass
class PSFSpeckleVisualizationData(SpeckleVisualizationData):
    """A speckle calibration that fitted a point spread function.

    Everything the shared panels need, plus the fitted kernel, which is the thing that
    was actually optimised where the SLM-plane field is derived from it.
    """

    # Defaulted because the base class already has a defaulted field, and a dataclass
    # cannot take a non-default one after that. The panels handle None.
    psf_kernel: NDArray | None = None

    def visualizer(self) -> SpeckleCalibratorVisualizer:
        return PSFCalibratorVisualizer(self)


class PSFCalibratorVisualizer(SpeckleCalibratorVisualizer):
    """Render a PSF-parameterised speckle calibration.

    The shared panels plus the fitted kernel a PSF calibration optimised. Its phase is
    the interesting half: a kernel that picks up structure there is carrying the
    aberration the SLM-plane field is derived from.
    """

    def __init__(self, data: PSFSpeckleVisualizationData) -> None:
        self.data = data

    def default_layout(self) -> PlotLayout:
        kernel = self.data.psf_kernel
        if kernel is None:
            return super().default_layout()

        kernel_shape = np.asarray(kernel).shape
        kernel_aspect = kernel_shape[0] / kernel_shape[1]
        return super().default_layout().add_row(
            [
                GridCell("psf_amplitude", aspect=kernel_aspect, colorbar=True),
                GridCell("psf_phase", aspect=kernel_aspect, colorbar=True),
            ]
        )

    def panels(self) -> dict[str, Panel]:
        kernel = self.data.psf_kernel
        if kernel is None:
            # No cells were added for it either, so the figure is just the shared one.
            return super().panels()

        kernel = np.asarray(kernel)
        return {
            **super().panels(),
            "psf_amplitude": self._image_panel(
                np.abs(kernel), "fitted PSF amplitude", cmap=INTENSITY_CMAP
            ),
            "psf_phase": self._image_panel(
                np.angle(kernel), "fitted PSF phase [rad]", cmap=PHASE_CMAP
            ),
        }
