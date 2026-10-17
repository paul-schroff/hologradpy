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

    Everything the shared panels need, plus the fitted kernel and its pitch.
    """

    # Defaulted because the base class already has a defaulted field, and a dataclass
    # cannot take a non-default one after that. The panels handle None.
    psf_kernel: NDArray | None = None
    #: The ``(y, x)`` pitch of the kernel in metres of the image plane.
    psf_pixel_size: tuple[float, float] | None = None

    def visualizer(self) -> SpeckleCalibratorVisualizer:
        return PSFCalibratorVisualizer(self)


class PSFCalibratorVisualizer(SpeckleCalibratorVisualizer):
    """Render a PSF-parameterized speckle calibration.

    The figure holds the shared panels and the amplitude and phase of the fitted
    kernel. The phase is the interesting half, since phase structure in the kernel
    carries the aberration. The SLM-plane field is derived from the kernel. The
    convergence panel shows the cost before the fit and after each optimization
    step.
    """

    loss_entry_name = "Levenberg-Marquardt step"
    first_loss_entry = 0

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
            "psf_amplitude": self._kernel_panel(
                np.abs(kernel), "fitted PSF amplitude", INTENSITY_CMAP
            ),
            "psf_phase": self._kernel_panel(
                np.angle(kernel), "fitted PSF phase [rad]", PHASE_CMAP
            ),
        }

    def _kernel_panel(self, values: NDArray, title: str, cmap: str) -> Panel:
        """One kernel cell, on axes in micrometres of the image plane when the payload
        carries the kernel's pitch.
        """
        pixel_size = self.data.psf_pixel_size
        if pixel_size is None:
            return self._image_panel(values, title, cmap=cmap)
        rows, columns = values.shape
        half_height = 0.5 * rows * pixel_size[0] * 1e6
        half_width = 0.5 * columns * pixel_size[1] * 1e6
        return lambda axs: self.draw_image(
            axs,
            values,
            cmap=cmap,
            title=title,
            extent=(-half_width, half_width, half_height, -half_height),
            xlabel="x [$\\mu$m]",
            ylabel="y [$\\mu$m]",
        )
