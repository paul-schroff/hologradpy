from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from .....analysis.error_metrics import wavefront_rms
from .....analysis.fitting import remove_tilt
from .....analysis.unwrapping import unwrap_2d_poisson
from .....visualizer import (
    INTENSITY_CMAP,
    PHASE_CMAP,
    GridCell,
    Panel,
    PlotLayout,
    VisualizationData,
)
from ....speckle.visualizer import SpeckleVisualizerBase
from .....serialization import record_type

if TYPE_CHECKING:
    from matplotlib.figure import Figure


@record_type("speckle_visualization")
@dataclass
class SpeckleVisualizationData(VisualizationData):
    """Everything ``SpeckleCalibrator`` records for visualization.

    ``slm_pattern`` is the SLM pattern that produced ``camera_image``, one full-sensor
    capture from the dataset, and ``roi_mask`` is the full-frame region-of-interest
    mask, so the last two overlay. Those three are all a dataset carries, and
    :meth:`SpeckleCalibratorVisualizer.render_dataset` draws them on their own, before
    anything has been fitted.

    The rest is what the fit adds. ``measured_roi`` / ``predicted_roi`` are cropped to
    the region's bounding box and masked, as the loss sees them. ``loss_history`` is the
    mean loss per epoch, and ``loss_component_history`` the same average for each term
    of it, empty when the cost held only one.

    ``injected_field`` is the SLM-plane field the bench was built with, recorded only
    when the calibration ran against simulated hardware, and ``beam_mask`` the region
    the comparison covers: the injected beam when there is one, so the region does not
    move with the fit's own errors, and the recovered beam otherwise.
    """

    camera_image: NDArray
    roi_mask: NDArray
    # Defaulted so a payload can be built straight after the capture, with nothing
    # fitted yet. render() says which of them is missing rather than drawing empty axes.
    slm_pattern: NDArray | None = None
    measured_roi: NDArray | None = None
    predicted_roi: NDArray | None = None
    recovered_amplitude: NDArray | None = None
    recovered_phase: NDArray | None = None
    loss_history: list[float] = field(default_factory=list)
    loss_component_history: dict[str, list[float]] = field(default_factory=dict)
    injected_field: NDArray | None = None
    beam_mask: NDArray | None = None

    def visualizer(self) -> SpeckleCalibratorVisualizer:
        """The visualizer that renders this payload."""
        return SpeckleCalibratorVisualizer(self)


class SpeckleCalibratorVisualizer(SpeckleVisualizerBase):
    """Render a speckle calibration from its :class:`SpeckleVisualizationData`."""

    def default_layout(self) -> PlotLayout:
        roi_aspect = self._aspect(self._require("measured_roi"))
        slm_aspect = self._aspect(self._require("recovered_phase"))

        # The default left / bottom margins assume image cells, which carry no tick
        # labels. Widen them so the loss curve's axis labels fit.
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
                GridCell("amplitude", aspect=slm_aspect, colorbar=True),
                GridCell("phase", aspect=slm_aspect, colorbar=True),
            ]
        )
        # The loss curve gets its own row: sharing one with the image cells puts its y
        # tick labels hard against the neighbouring colorbar.
        layout.add_row(
            [GridCell("loss", colspan=3, aspect="auto", height=2.2)]
        )
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
            "amplitude": self._image_panel(
                self._require("recovered_amplitude"), "recovered amplitude",
                cmap=INTENSITY_CMAP,
            ),
            "phase": self._image_panel(
                self._require("recovered_phase"), "recovered phase", cmap=PHASE_CMAP
            ),
            "loss": self._loss_panel,
        }

    def render_comparison(self, **kwargs) -> Figure:
        """Draw the recovered field against the one that was injected. Only possible for
        a calibration run against simulated hardware.

        Both phases are unwrapped and have piston and tilt removed, by
        :func:`~hologradpy.analysis.fitting.remove_tilt`, since a constant or a ramp
        displaces the focus rather than aberrating it.

        Raises:
            RuntimeError: If the payload carries no injected field, which means the
                calibration ran against a camera that could not supply one.
        """
        return self.render(
            layout=self._comparison_layout(),
            panels=self._comparison_panels(),
            **kwargs,
        )

    def _comparison(self) -> tuple[NDArray, NDArray, NDArray, NDArray, NDArray]:
        """The five aligned arrays the comparison panels share."""
        if self.data.injected_field is None:
            raise RuntimeError(
                "This calibration carries no injected field to compare against, so it "
                "did not run on simulated hardware. Only a camera exposing "
                "'static_slm_field' can supply one."
            )

        injected = np.asarray(self.data.injected_field)
        mask = (
            None if self.data.beam_mask is None
            else np.asarray(self.data.beam_mask, dtype=bool)
        )

        unwrap_mask = np.ones(injected.shape, dtype=bool) if mask is None else mask

        wrapped_recovered = np.asarray(self._require("recovered_phase"))
        recovered_phase = remove_tilt(
            unwrap_2d_poisson(wrapped_recovered, unwrap_mask), mask=mask
        )
        injected_phase = remove_tilt(
            unwrap_2d_poisson(np.angle(injected), unwrap_mask), mask=mask
        )

        recovered_amplitude = np.asarray(
            self._require("recovered_amplitude"), dtype=float
        )
        injected_amplitude = np.abs(injected)
        recovered_amplitude = recovered_amplitude / recovered_amplitude.max()
        injected_amplitude = injected_amplitude / injected_amplitude.max()

        if mask is not None:
            recovered_phase = np.where(mask, recovered_phase, np.nan)
            injected_phase = np.where(mask, injected_phase, np.nan)

        return (
            injected_phase,
            recovered_phase,
            injected_amplitude,
            recovered_amplitude,
            np.ones_like(injected_amplitude, dtype=bool) if mask is None else mask,
        )

    def _comparison_layout(self) -> PlotLayout:
        shape = np.asarray(self._require("recovered_phase")).shape
        aspect = shape[0] / shape[1]

        layout = PlotLayout(column_width=3.6, margins=(1.0, 0.15, 0.5, 0.5))
        for row in ("phase", "amplitude"):
            layout.add_row(
                [
                    GridCell(f"injected_{row}", aspect=aspect, colorbar=True),
                    GridCell(f"recovered_{row}", aspect=aspect, colorbar=True),
                    GridCell(f"difference_{row}", aspect=aspect, colorbar=True),
                ]
            )
        return layout

    def _comparison_panels(self) -> dict[str, Panel]:
        (
            injected_phase,
            recovered_phase,
            injected_amplitude,
            recovered_amplitude,
            mask,
        ) = self._comparison()

        phase_limit = float(np.nanmax(np.abs(injected_phase)))
        phase_difference = recovered_phase - injected_phase
        amplitude_difference = recovered_amplitude - injected_amplitude

        return {
            "injected_phase": self._image_panel(
                injected_phase, "injected phase [rad]", cmap=PHASE_CMAP,
                vmin=-phase_limit, vmax=phase_limit,
            ),
            "recovered_phase": self._image_panel(
                recovered_phase, "recovered phase [rad]", cmap=PHASE_CMAP,
                vmin=-phase_limit, vmax=phase_limit,
            ),
            "difference_phase": self._difference_panel(
                phase_difference,
                "phase difference",
                "rad",
                residual_rms=wavefront_rms(phase_difference, mask),
            ),
            "injected_amplitude": self._image_panel(
                injected_amplitude, "injected amplitude", cmap=INTENSITY_CMAP,
                vmin=0.0, vmax=1.0,
            ),
            "recovered_amplitude": self._image_panel(
                recovered_amplitude, "recovered amplitude", cmap=INTENSITY_CMAP,
                vmin=0.0, vmax=1.0,
            ),
            "difference_amplitude": self._difference_panel(
                amplitude_difference,
                "amplitude difference",
                "",
                residual_rms=float(np.sqrt(np.mean(amplitude_difference[mask] ** 2))),
            ),
        }
