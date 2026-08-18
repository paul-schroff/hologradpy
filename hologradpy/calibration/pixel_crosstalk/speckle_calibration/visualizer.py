"""Rendering a fitted pixel-crosstalk kernel."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from ....serialization import record_type
from ....visualizer import (
    GridCell,
    INTENSITY_CMAP,
    Panel,
    PlotLayout,
    VisualizationData,
    foreground_color,
)
from ...speckle.visualizer import SpeckleVisualizerBase

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure


@record_type("pixel_crosstalk_visualization")
@dataclass
class CrosstalkVisualizationData(VisualizationData):
    """Everything :class:`CrosstalkSpeckleCalibrator` records for visualization.

    ``camera_image``, ``roi_mask``, ``slm_pattern``, ``measured_roi``,
    ``predicted_roi``, ``loss_history`` and ``loss_component_history`` are the shared
    speckle panels, drawn by :class:`SpeckleVisualizerBase`.

    ``kernel`` is what this fit recovered, on the sub-pixel grid, and
    ``upscale_factor`` says how many of its samples span one SLM pixel, which is what
    turns a sample index into a position. ``injected_kernel`` is the kernel the bench
    was built with, recorded only when the calibration ran against simulated hardware.
    """

    camera_image: NDArray
    roi_mask: NDArray
    upscale_factor: int = 1
    # Defaulted so a payload can be built straight after the capture, with nothing
    # fitted yet. render() says which of them is missing rather than drawing empty axes.
    slm_pattern: NDArray | None = None
    measured_roi: NDArray | None = None
    predicted_roi: NDArray | None = None
    kernel: NDArray | None = None
    injected_kernel: NDArray | None = None
    loss_history: list[float] = field(default_factory=list)
    loss_component_history: dict[str, list[float]] = field(default_factory=dict)

    def visualizer(self) -> CrosstalkVisualizer:
        """The visualizer that renders this payload."""
        return CrosstalkVisualizer(self)


class CrosstalkVisualizer(SpeckleVisualizerBase):
    """Render a pixel-crosstalk calibration from its
    :class:`CrosstalkVisualizationData`.
    """

    def default_layout(self) -> PlotLayout:
        roi_aspect = self._aspect(self._require("measured_roi"))

        # The default left / bottom margins assume image cells, which carry no tick
        # labels. Widen them so the line axes' labels fit.
        layout = PlotLayout(column_width=3.6, margins=(1.0, 0.15, 0.5, 0.5))
        layout.add_row(self._dataset_cells())
        layout.add_row(
            [
                GridCell("measured", aspect=roi_aspect, colorbar=True),
                GridCell("predicted", aspect=roi_aspect, colorbar=True),
                GridCell("residual", aspect=roi_aspect, colorbar=True),
            ]
        )
        layout.add_row(
            [
                GridCell("kernel", aspect="equal", colorbar=True),
                GridCell("profile", aspect="auto", height=2.6),
            ]
        )
        layout.add_row([GridCell("loss", colspan=3, aspect="auto", height=2.2)])
        return layout

    def panels(self) -> dict[str, Panel]:
        measured = self._normalized(self._require("measured_roi"))
        predicted = self._normalized(self._require("predicted_roi"))
        roi_vmax = max(float(np.nanmax(measured)), float(np.nanmax(predicted)))
        residual = measured - predicted

        return {
            **self._dataset_panels(),
            "measured": self._image_panel(
                measured, "measured speckle (ROI)", vmax=roi_vmax
            ),
            "predicted": self._image_panel(
                predicted, "predicted speckle (ROI)", vmax=roi_vmax
            ),
            # Both are normalized to unit sum, exactly as the loss compares them, so
            # this is the mismatch the fit is still paying for.
            "residual": self._difference_panel(
                residual,
                "measured - predicted",
                "",
                residual_rms=float(np.sqrt(np.mean(residual**2))),
            ),
            "kernel": self._kernel_panel,
            "profile": self._profile_panel,
            "loss": self._loss_panel,
        }

    def _positions(self, kernel: NDArray) -> NDArray:
        """Where each kernel sample sits, in SLM pixels from the center."""
        size = kernel.shape[0]
        factor = max(int(self.data.upscale_factor), 1)
        return (np.arange(size) - (size - 1) / 2) / factor

    def _kernel_panel(self, axs: Axes):
        kernel = np.asarray(self._require("kernel"), dtype=float)
        weight = self._central_weight(kernel)
        return self.draw_image(
            axs,
            kernel,
            cmap=INTENSITY_CMAP,
            title=f"recovered kernel ({weight:.1%} in pixel)",
            interpolation="nearest",
        )

    def _central_weight(self, kernel: NDArray) -> float:
        """The fraction of the kernel inside the SLM pixel it belongs to."""
        factor = max(int(self.data.upscale_factor), 1)
        start = (kernel.shape[0] - factor) // 2
        center = kernel[start : start + factor, start : start + factor]
        total = kernel.sum()
        return float(center.sum() / total) if total else float("nan")

    def _profile_panel(self, axs: Axes) -> None:
        """Cuts through the center of the kernel, along each axis.

        Plotted in SLM pixels so the reach can be read straight off, with the injected
        kernel behind it when the bench had one. The two cuts differ only for a model
        that lets them, so a symmetric fit draws them on top of each other.
        """
        kernel = np.asarray(self._require("kernel"), dtype=float)
        positions = self._positions(kernel)
        middle = kernel.shape[0] // 2

        curves = [
            {
                "x": positions,
                "y": kernel[middle, :],
                "color": foreground_color(),
                "label": "recovered, horizontal",
            },
            {
                "x": positions,
                "y": kernel[:, middle],
                "style": "--",
                "color": foreground_color(),
                "label": "recovered, vertical",
            },
        ]

        injected = self.data.injected_kernel
        if injected is not None:
            injected = np.asarray(injected, dtype=float)
            curves += [
                {
                    "x": self._positions(injected),
                    "y": injected[injected.shape[0] // 2, :],
                    "style": ":",
                    "label": "injected, horizontal",
                },
                {
                    "x": self._positions(injected),
                    "y": injected[:, injected.shape[1] // 2],
                    "style": ":",
                    "label": "injected, vertical",
                },
            ]

        self.draw_line(
            axs,
            curves,
            xlabel="position [SLM pixels]",
            ylabel="weight",
            title="kernel profile",
            legend=True,
        )

    def render_comparison(self, **kwargs) -> Figure:
        """Draw the recovered kernel against the one that was injected. Only possible
        for a calibration run against simulated hardware.

        Raises:
            RuntimeError: If the payload carries no injected kernel, which means the
                calibration ran against a camera that could not supply one.
        """
        return self.render(
            layout=self._comparison_layout(),
            panels=self._comparison_panels(),
            **kwargs,
        )

    def _comparison(self) -> tuple[NDArray, NDArray]:
        """The recovered and injected kernels, on a shared grid."""
        if self.data.injected_kernel is None:
            raise RuntimeError(
                "This calibration carries no injected kernel to compare against, so it "
                "did not run on simulated hardware. Only a camera exposing "
                "'static_crosstalk_kernel' can supply one."
            )

        recovered = np.asarray(self._require("kernel"), dtype=float)
        injected = np.asarray(self.data.injected_kernel, dtype=float)
        if recovered.shape != injected.shape:
            raise RuntimeError(
                f"The recovered kernel is {recovered.shape} and the injected one "
                f"{injected.shape}, so they were built on different grids and cannot "
                "be compared. Fit with the upscale_factor and extent the bench uses."
            )

        # Both to unit sum, since only the shape of the kernel is recoverable: a scaled
        # kernel is the same optics with a different overall phase offset.
        return self._normalized(recovered), self._normalized(injected)

    def _comparison_layout(self) -> PlotLayout:
        layout = PlotLayout(column_width=3.6, margins=(1.0, 0.15, 0.5, 0.5))
        layout.add_row(
            [
                GridCell("injected", aspect="equal", colorbar=True),
                GridCell("recovered", aspect="equal", colorbar=True),
                GridCell("difference", aspect="equal", colorbar=True),
            ]
        )
        layout.add_row([GridCell("profile", colspan=3, aspect="auto", height=2.6)])
        return layout

    def _comparison_panels(self) -> dict[str, Panel]:
        recovered, injected = self._comparison()
        difference = recovered - injected
        limit = float(np.nanmax(injected))

        return {
            "injected": self._kernel_image_panel(
                injected, "injected kernel", vmax=limit
            ),
            "recovered": self._kernel_image_panel(
                recovered, "recovered kernel", vmax=limit
            ),
            "difference": self._difference_panel(
                difference,
                "recovered - injected",
                "",
                residual_rms=float(np.sqrt(np.mean(difference**2))),
            ),
            "profile": self._profile_panel,
        }

    def _kernel_image_panel(
        self, kernel: NDArray, title: str, *, vmax: float | None = None
    ) -> Panel:
        """One kernel cell, drawn sample by sample so the sub-pixel grid stays
        readable.
        """
        return lambda axs: self.draw_image(
            axs,
            kernel,
            cmap=INTENSITY_CMAP,
            vmin=None if vmax is None else 0.0,
            vmax=vmax,
            title=title,
            interpolation="nearest",
        )
