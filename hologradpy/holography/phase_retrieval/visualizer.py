"""Rendering the result of a phase retrieval."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from ...serialization import record_type
from ...visualizer import (
    DIFFERENCE_CMAP,
    INTENSITY_CMAP,
    PHASE_CMAP,
    BaseVisualizer,
    GridCell,
    Panel,
    PlotLayout,
    VisualizationData,
    foreground_color,
    region_bounding_box,
)

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    from .abstract import PhaseRetrievalData

TWO_PI = 2 * np.pi

#: Fraction of the peak intensity below which the image-plane phase is left undrawn.
PHASE_VISIBLE_FRACTION = 0.05


@record_type("phase_retrieval_visualization")
@dataclass
class PhaseRetrievalVisualizationData(VisualizationData):
    """What a retrieval records for plotting, beyond the fields of its own record.

    ``PhaseRetrievalData`` already carries the SLM phase, the target, the signal region
    and the loss history, so this holds only what has to be computed by running the
    model: the intensity and image-plane phase the retrieved SLM phase produces, and
    optionally the intensity the starting phase produced.

    ``retrieved_phase`` is the phase in the image plane, which is what an intensity-only
    cost leaves free. It is distinct from ``PhaseRetrievalData.phase``, the pattern on
    the SLM.

    ``target_phase`` is the image-plane phase the cost asked for, and is only present
    when the cost constrained phase at all. Its presence is what adds the phase row to
    default layout.


    ``metric_history`` holds each metric recorded at every objective evaluation, on the
    same footing as the loss, so convergence can be read in the units the result is
    judged in as well as in the cost.
    """

    retrieved_intensity: NDArray
    retrieved_phase: NDArray | None = None
    target_phase: NDArray | None = None
    metric_history: dict[str, list[float]] = field(default_factory=dict)
    initial_intensity: NDArray | None = None

    def visualizer(self, retrieval: PhaseRetrievalData) -> PhaseRetrievalVisualizer:
        """The visualizer that renders this payload, bound to its record."""
        return PhaseRetrievalVisualizer(retrieval)


class PhaseRetrievalVisualizer(BaseVisualizer):
    """Render a phase retrieval: what was asked for, what came out, and the cost.

    Takes the :class:`PhaseRetrievalData` record rather than a payload on its own,
    since the record holds most of what is drawn. The predicted intensity comes from
    the record's ``visualization_data``.
    """

    def __init__(self, retrieval: PhaseRetrievalData) -> None:
        """
        Args:
            retrieval: The finished retrieval to draw.
        """
        self.retrieval = retrieval

    @property
    def extras(self) -> PhaseRetrievalVisualizationData:
        """The plotting payload, or a message naming what is missing.

        Raises:
            RuntimeError: The record carries no payload, so the retrieval was run
                without one and the predicted intensity was never computed.
        """
        extras = self.retrieval.visualization_data
        if not isinstance(extras, PhaseRetrievalVisualizationData):
            raise RuntimeError(
                "This retrieval carries no PhaseRetrievalVisualizationData, so the "
                "intensity its phase produces was never recorded. Run the retriever "
                "through run_and_record, which computes it."
            )
        return extras

    def _aspect(self, image: NDArray) -> float:
        shape = np.asarray(image).shape
        return shape[0] / shape[1]

    def _region(self) -> NDArray:
        """The signal region as a boolean mask, or the whole plane if there is none."""
        target = np.asarray(self.retrieval.target)
        if self.retrieval.signal_region is None:
            return np.ones(target.shape, dtype=bool)
        return np.asarray(self.retrieval.signal_region, dtype=bool)

    def _crop(self) -> tuple[slice, slice]:
        """The signal region, with a margin, as row and column slices."""
        return region_bounding_box(self._region())

    def _constrains_phase(self) -> bool:
        """Whether the cost asked for an image-plane phase to compare against."""
        return (
            self.extras.target_phase is not None
            and self.extras.retrieved_phase is not None
        )

    def default_layout(self) -> PlotLayout:
        rows, columns = self._crop()
        target = np.asarray(self.retrieval.target)
        close_aspect = self._aspect(target[rows, columns])
        plane_aspect = self._aspect(target)
        slm_aspect = self._aspect(self.retrieval.phase)

        # Six columns, so a row of three and a row of two both divide it evenly and
        # line up. The default left / bottom margins assume image cells, which carry no
        # tick labels, so widen them for the loss curve axis labels.
        layout = PlotLayout(column_width=1.8, margins=(1.0, 0.15, 0.5, 0.5))
        # The close-up: what was asked for, what came out, and the error between them,
        # over the region the cost constrained.
        layout.add_row(
            [
                GridCell("target", colspan=2, aspect=close_aspect, colorbar=True),
                GridCell("retrieved", colspan=2, aspect=close_aspect, colorbar=True),
                GridCell("residual", colspan=2, aspect=close_aspect, colorbar=True),
            ]
        )
        # The same three for phase, but only when a cost constrained it. An
        # cost leaves the image-plane phase free, so there is nothing to compare to.
        if self._constrains_phase():
            layout.add_row(
                [
                    GridCell(
                        "target_phase", colspan=2, aspect=close_aspect, colorbar=True
                    ),
                    GridCell(
                        "output_phase", colspan=2, aspect=close_aspect, colorbar=True
                    ),
                    GridCell(
                        "phase_error", colspan=2, aspect=close_aspect, colorbar=True
                    ),
                ]
            )
        # The two full planes, where the light that missed the region shows up.
        layout.add_row(
            [
                GridCell("slm_phase", colspan=3, aspect=slm_aspect, colorbar=True),
                GridCell("output", colspan=3, aspect=plane_aspect, colorbar=True),
            ]
        )
        layout.add_row([GridCell("loss", colspan=6, aspect="auto", height=2.2)])
        # The same run in the units it is judged in. The cost is what was minimized,
        # which is not always what the result is measured by.
        tracked = self._tracked_metrics()
        if tracked:
            span = max(1, 6 // len(tracked))
            layout.add_row(
                [
                    GridCell(
                        f"metric_{index}", colspan=span, aspect="auto", height=2.0,
                        sharex="loss",
                    )
                    for index in range(len(tracked))
                ]
            )
        return layout

    def _tracked_metrics(self) -> list[str]:
        """The metrics recorded often enough to draw a curve from."""
        history = self.extras.metric_history or {}
        return [name for name, values in history.items() if len(values) > 1]

    def panels(self) -> dict[str, Panel]:
        target, retrieved, residual, region = self._compared()
        rows, columns = self._crop()
        limit = float(np.nanmax(target))

        cells = {
            "target": self._image_panel(target[rows, columns], "target", vmax=limit),
            "retrieved": self._image_panel(
                retrieved[rows, columns], "retrieved", vmax=limit
            ),
            "residual": self._residual_panel(
                residual[rows, columns], region[rows, columns]
            ),
            "slm_phase": self._image_panel(
                np.asarray(self.retrieval.phase) % TWO_PI,
                "retrieved SLM phase [rad]",
                cmap=PHASE_CMAP,
            ),
            "output": self._image_panel(retrieved, "full output plane", vmax=limit),
            "loss": self._loss_panel,
        }
        if self._constrains_phase():
            cells.update(self._phase_panels(rows, columns, region, retrieved))
        for index, name in enumerate(self._tracked_metrics()):
            cells[f"metric_{index}"] = self._metric_panel(name)
        return cells

    def _metric_panel(self, name: str) -> Panel:
        """One metric against the objective evaluation it was recorded at.

        Linear, since a metric already in decibels is logarithmic and plotting it on a
        log axis twice over says nothing.
        """
        values = list(self.extras.metric_history[name])

        def panel(axs: Axes) -> None:
            self.draw_line(
                axs,
                [
                    {
                        "x": np.arange(1, len(values) + 1),
                        "y": values,
                        "color": foreground_color(),
                        "label": name,
                    }
                ],
                xlabel="iteration",
                ylabel=name,
                title=f"{name}: {values[0]:.4g} to {values[-1]:.4g}",
            )

        return panel

    def _phase_panels(
        self, rows: slice, columns: slice, region: NDArray, intensity: NDArray
    ) -> dict[str, Panel]:
        """The phase asked for, the phase produced, and how far apart they are.

        All three are masked to the pixels holding at least
        :data:`PHASE_VISIBLE_FRACTION` of the peak intensity. The phase of a dark pixel
        is numerical noise, and a signal region that is mostly dark would otherwise fill
        the panels with it.
        """
        peak = float(np.nanmax(intensity[region])) if region.any() else 0.0
        lit = region & (intensity >= PHASE_VISIBLE_FRACTION * peak)

        wanted = _wrapped(np.asarray(self.extras.target_phase))
        produced = _wrapped(np.asarray(self.extras.retrieved_phase))
        # Wrapped, so a pair straddling the cut reads as the small difference it is
        # rather than a whole cycle.
        error = _wrapped(produced - wanted)

        inside = np.nanmax(np.abs(np.where(lit, error, np.nan))) if lit.any() else 1.0
        spread = float(inside) if np.isfinite(inside) else 1.0

        def shown(image: NDArray) -> NDArray:
            return np.where(lit, image, np.nan)[rows, columns]

        return {
            "target_phase": self._image_panel(
                shown(wanted), "target phase [rad]", cmap=PHASE_CMAP,
                vmin=-np.pi, vmax=np.pi,
            ),
            "output_phase": self._image_panel(
                shown(produced), "image-plane phase [rad]", cmap=PHASE_CMAP,
                vmin=-np.pi, vmax=np.pi,
            ),
            "phase_error": self._image_panel(
                shown(error), f"phase error [rad] (max {spread:.2f})",
                cmap=DIFFERENCE_CMAP, vmin=-spread, vmax=spread,
            ),
        }

    def _compared(self) -> tuple[NDArray, NDArray, NDArray, NDArray]:
        """Target and retrieved intensity on one scale, their difference, the region.

        Both are normalized over the signal region, which is the only place the cost
        constrained them, so the residual shows the error the retrieval is still
        paying rather than an overall scale difference.
        """
        target = np.asarray(self.retrieval.target, dtype=float)
        retrieved = np.asarray(self.extras.retrieved_intensity, dtype=float)
        region = self._region()

        target = _normalized_in(target, region)
        retrieved = _normalized_in(retrieved, region)
        return target, retrieved, retrieved - target, region

    def _image_panel(
        self,
        image: NDArray,
        title: str,
        *,
        cmap: str = INTENSITY_CMAP,
        vmin: float | None = None,
        vmax: float | None = None,
    ) -> Panel:
        """One image cell, bound to its data and title.

        Given only a `vmax`, the scale starts at zero, which is what an intensity
        wants. Phase panels pass both ends.
        """
        if vmin is None and vmax is not None:
            vmin = 0.0
        return lambda axs: self.draw_image(
            axs, image, cmap=cmap, vmin=vmin, vmax=vmax, title=title
        )

    def _residual_panel(self, residual: NDArray, region: NDArray) -> Panel:
        """The error inside the signal region, symmetric about zero.

        Symmetric limits are what make the neutral colour mean agreement, and the
        residual is small next to the intensity it came from, so it needs its own
        scale to be visible at all.
        """
        inside = residual[region]
        rms = float(np.sqrt(np.mean(inside**2))) if inside.size else float("nan")
        limit = float(np.nanmax(np.abs(inside))) if inside.size else 1.0
        return lambda axs: self.draw_image(
            axs,
            np.where(region, residual, np.nan),
            cmap=DIFFERENCE_CMAP,
            vmin=-limit,
            vmax=limit,
            title=f"retrieved - target (rms {rms:.3g})",
        )

    def _loss_panel(self, axs: Axes) -> None:
        history = list(self.retrieval.loss_history)
        if not history:
            axs.set_axis_off()
            axs.set_title("loss (not recorded)")
            return

        self.draw_line(
            axs,
            [
                {
                    "x": np.arange(1, len(history) + 1),
                    "y": history,
                    "color": foreground_color(),
                    "label": "total",
                }
            ],
            # The metric panels below carry the shared axis label when they exist.
            xlabel=None if self._tracked_metrics() else "iteration",
            ylabel="loss",
            title="convergence",
            yscale="log" if min(history) > 0 else "linear",
        )

    def render_initial(self, **kwargs) -> Figure:
        """Draw the starting intensity beside the target and the retrieved one.

        Raises:
            RuntimeError: The retrieval recorded no initial intensity.
        """
        if self.extras.initial_intensity is None:
            raise RuntimeError(
                "This retrieval recorded no initial intensity, so there is nothing to "
                "compare the starting point against."
            )
        target, retrieved, _, region = self._compared()
        initial = _normalized_in(
            np.asarray(self.extras.initial_intensity, dtype=float), region
        )
        limit = float(np.nanmax(target))

        rows, columns = self._crop()
        aspect = self._aspect(target[rows, columns])
        layout = PlotLayout(column_width=3.6, margins=(1.0, 0.15, 0.5, 0.5))
        layout.add_row(
            [
                GridCell("initial", aspect=aspect, colorbar=True),
                GridCell("target", aspect=aspect, colorbar=True),
                GridCell("retrieved", aspect=aspect, colorbar=True),
            ]
        )
        panels = {
            "initial": self._image_panel(
                initial[rows, columns], "initial", vmax=limit
            ),
            "target": self._image_panel(target[rows, columns], "target", vmax=limit),
            "retrieved": self._image_panel(
                retrieved[rows, columns], "retrieved", vmax=limit
            ),
        }
        return self.render(layout=layout, panels=panels, **kwargs)


def _wrapped(phase: NDArray) -> NDArray:
    """``phase`` folded into (-pi, pi], where a flat phase sits at zero."""
    return (phase + np.pi) % TWO_PI - np.pi


def _normalized_in(image: NDArray, region: NDArray) -> NDArray:
    """``image`` scaled so its signal region sums to one, as the cost saw it."""
    total = float(image[region].sum())
    return image / total if total > 0 else image
