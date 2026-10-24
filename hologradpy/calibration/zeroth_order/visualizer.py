"""Rendering a fitted zeroth order."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from ...serialization import record_type
from ...visualizer import (
    INTENSITY_CMAP,
    BaseVisualizer,
    GridCell,
    Panel,
    PlotLayout,
    VisualizationData,
)

if TYPE_CHECKING:
    from matplotlib.axes import Axes


@record_type("zeroth_order_visualization")
@dataclass
class ZerothOrderVisualizationData(VisualizationData):
    """What :class:`~.calibrator.ZerothOrderCalibrator` records for visualization.

    Attributes:
        measured_intensity: The intensity of the missed field measured with every
            grating, on the box of the region, ``(G, h, w)``. It is zero where the
            probes did not resolve the field.
        fitted_intensity: The intensity of the fitted kernel and light at the wraps
            for every grating, on the same pixels, ``(G, h, w)``.
        kernel_intensity: The intensity of the fitted kernel, ``(rows, columns)``.
        wrap_weight: The fitted weight of the light at the wraps.
        relative_residual: The norm of what the fit leaves of the measured fields, over
            the norm of the measured fields.
    """

    measured_intensity: NDArray
    fitted_intensity: NDArray
    kernel_intensity: NDArray
    wrap_weight: complex = 0j
    relative_residual: float = float("nan")

    def visualizer(self) -> ZerothOrderVisualizer:
        """The visualizer that renders this payload."""
        return ZerothOrderVisualizer(self)


class ZerothOrderVisualizer(BaseVisualizer):
    """Render a zeroth-order calibration from its :class:`ZerothOrderVisualizationData`.

    The first row shows the measured intensity of every grating, the second the fitted
    intensity on the same scale, and the third the kernel.
    """

    def __init__(self, data: ZerothOrderVisualizationData) -> None:
        self.data: ZerothOrderVisualizationData = data

    def default_layout(self) -> PlotLayout:
        measured = np.asarray(self.data.measured_intensity)
        kernel = np.asarray(self.data.kernel_intensity)
        aspect = measured.shape[-2] / measured.shape[-1]
        layout = PlotLayout()
        layout.add_row(
            [
                GridCell(f"measured_{index}", aspect=aspect, colorbar=True)
                for index in range(len(measured))
            ]
        )
        layout.add_row(
            [
                GridCell(f"fitted_{index}", aspect=aspect, colorbar=True)
                for index in range(len(measured))
            ]
        )
        layout.add_row(
            [
                GridCell(
                    "kernel", aspect=kernel.shape[0] / kernel.shape[1], colorbar=True
                )
            ]
        )
        return layout

    def panels(self) -> dict[str, Panel]:
        measured = np.asarray(self.data.measured_intensity, dtype=float)
        fitted = np.asarray(self.data.fitted_intensity, dtype=float)
        panels: dict[str, Panel] = {}
        for index in range(len(measured)):
            vmax = max(float(measured[index].max()), float(fitted[index].max()))
            panels[f"measured_{index}"] = self._image_panel(
                measured[index], f"grating {index}, measured", vmax
            )
            panels[f"fitted_{index}"] = self._image_panel(
                fitted[index], f"grating {index}, fitted", vmax
            )
        weight = complex(self.data.wrap_weight)
        panels["kernel"] = self._image_panel(
            np.asarray(self.data.kernel_intensity, dtype=float),
            f"kernel, w = {weight.real:.3g}{weight.imag:+.3g}i, "
            f"residual {self.data.relative_residual:.1%}",
            None,
        )
        return panels

    def _image_panel(self, image: NDArray, title: str, vmax: float | None) -> Panel:
        def panel(axs: Axes):
            return self.draw_image(
                axs, image, cmap=INTENSITY_CMAP, vmin=0.0, vmax=vmax, title=title
            )

        return panel
