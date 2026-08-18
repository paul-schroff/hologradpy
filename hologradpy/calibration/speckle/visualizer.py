"""Rendering shared by every speckle calibration, whatever it fitted."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from matplotlib.axes import Axes
from matplotlib.cm import ScalarMappable
from matplotlib.figure import Figure

from ...visualizer import (
    BaseVisualizer,
    DIFFERENCE_CMAP,
    GridCell,
    INTENSITY_CMAP,
    PHASE_CMAP,
    Panel,
    PlotLayout,
    foreground_color,
)


class SpeckleVisualizerBase(BaseVisualizer):
    """The panels any speckle fit can draw.

    What the dataset alone shows (the pattern displayed and the frame it produced), how
    the fit converged, and the measured speckle beside the predicted one. What was
    recovered is the subclass's business, since that is the only part that differs
    between fitting a wavefront and fitting a crosstalk kernel.

    A subclass supplies :attr:`data` carrying at least ``camera_image``, ``roi_mask``,
    ``slm_pattern``, ``measured_roi``, ``predicted_roi``, ``loss_history`` and
    ``loss_component_history``.
    """

    def __init__(self, data) -> None:
        self.data = data

    def _aspect(self, image: NDArray) -> float:
        shape = np.asarray(image).shape
        return shape[0] / shape[1]

    def _require(self, name: str) -> NDArray:
        """One of the fitted arrays, or a message naming what has not happened yet."""
        value = getattr(self.data, name)
        if value is None:
            raise RuntimeError(
                f"This payload carries no {name}, so it was recorded from a captured "
                "dataset rather than a finished fit. Run fit_wavefront and read the "
                "calibration's visualization_data, or call render_dataset() for the "
                "panels a dataset alone can fill."
            )
        return value

    def _dataset_layout(self) -> PlotLayout:
        layout = PlotLayout(column_width=3.6, margins=(1.0, 0.15, 0.5, 0.5))
        layout.add_row(self._dataset_cells())
        return layout

    def _dataset_cells(self) -> list[GridCell]:
        """The pattern that was displayed and the frame it produced, side by side.

        The pattern cell is dropped when the payload has none, so a record written
        before it was carried still renders, one cell narrower.
        """
        cells = [
            GridCell(
                "camera", aspect=self._aspect(self.data.camera_image), colorbar=True
            )
        ]
        if self.data.slm_pattern is None:
            return cells

        return [
            GridCell(
                "pattern",
                aspect=self._aspect(self.data.slm_pattern),
                colorbar=True,
            ),
            *cells,
        ]

    def _dataset_panels(self) -> dict[str, Panel]:
        panels: dict[str, Panel] = {"camera": self._camera_panel}
        if self.data.slm_pattern is None:
            return panels

        return {
            "pattern": self._image_panel(
                self.data.slm_pattern, "SLM pattern [levels]", cmap=PHASE_CMAP
            ),
            **panels,
        }

    def render_dataset(self, **kwargs) -> Figure:
        """Draw the captured dataset alone: one SLM pattern and the frame it produced.

        Available as soon as a dataset exists, so the capture can be checked before
        spending a fit on it. The region of interest should sit on the speckle and
        exclude the zeroth order, and the frame should be exposed rather than saturated.

        Raises:
            RuntimeError: If the payload carries no phase pattern, since the figure
                would then be one cell the full diagnostics already draws.
        """
        self._require("slm_pattern")
        return self.render(
            layout=self._dataset_layout(), panels=self._dataset_panels(), **kwargs
        )

    def _difference_panel(
        self, difference: NDArray, title: str, unit: str, *, residual_rms: float
    ) -> Panel:
        """A residual cell: diverging, symmetric about zero, its error in the title.

        Symmetric limits are what make the neutral color mean agreement rather than
        some arbitrary offset, and the residual is small enough next to the field it
        came from that it needs its own scale to be visible at all.
        """
        limit = float(np.nanmax(np.abs(difference)))
        suffix = f" {unit}" if unit else ""
        return self._image_panel(
            difference,
            f"{title} (rms {residual_rms:.3f}{suffix})",
            cmap=DIFFERENCE_CMAP,
            vmin=-limit,
            vmax=limit,
        )

    def _image_panel(
        self,
        image: NDArray,
        title: str,
        *,
        cmap: str = INTENSITY_CMAP,
        vmin: float | None = None,
        vmax: float | None = None,
    ) -> Panel:
        """One image cell, bound to its data and title."""
        if vmin is None and vmax is not None:
            vmin = 0.0
        return lambda axs: self.draw_image(
            axs, image, cmap=cmap, vmin=vmin, vmax=vmax, title=title
        )

    def _camera_panel(self, axs: Axes) -> ScalarMappable:
        mappable = self.draw_image(
            axs, self.data.camera_image, cmap=INTENSITY_CMAP, title="camera + ROI"
        )
        axs.contour(
            np.asarray(self.data.roi_mask, dtype=float),
            levels=[0.5],
            colors="white",
            linewidths=0.8,
        )
        return mappable

    @staticmethod
    def _normalized(image: NDArray) -> NDArray:
        array = np.asarray(image, dtype=float)
        total = array.sum()
        return array / total if total > 0 else array

    def _loss_panel(self, axs: Axes) -> None:
        history = list(self.data.loss_history)
        if not history:
            axs.set_axis_off()
            axs.set_title("loss (not recorded)")
            return

        components = {
            label: list(values)
            for label, values in self.data.loss_component_history.items()
            if values
        }
        if len(components) < 2:
            components = {}

        epochs = np.arange(1, len(history) + 1)
        curves = [
            {
                "x": epochs,
                "y": history,
                "color": foreground_color(),
                "label": "total",
            }
        ]
        curves += [
            {
                "x": np.arange(1, len(values) + 1),
                "y": values,
                "style": "--",
                "label": label,
            }
            for label, values in components.items()
        ]

        self.draw_line(
            axs,
            curves,
            xlabel="epoch",
            ylabel="loss",
            title="convergence",
            yscale="log" if min(history) > 0 else "linear",
            legend=bool(components),
        )
