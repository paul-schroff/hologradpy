from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from ...analysis.error_metrics import normalize
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
)
from ...serialization import record_type

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.cm import ScalarMappable
    from matplotlib.figure import Figure

    from .abstract import CameraFeedbackData


@record_type("target_placement")
@dataclass
class TargetPlacementData(VisualizationData):
    """Where the target patch will sit on the sensor, before anything is run.

    Produced by
    :meth:`~hologradpy.holography.camera_feedback.FeedbackCorrectorBase.placement_data`
    once the camera mapping is known, which is the earliest the answer exists: the
    patch is positioned relative to the zeroth order, and only the mapping knows where
    that is.

    All pixel coordinates are ``(row, column)`` on the sensor. ``target_position`` is
    the request that produced them, as ``(x, y)`` metres in the optical plane.
    """

    target: NDArray
    signal_region: NDArray
    zeroth_order: tuple[float, float]
    target_center: tuple[float, float]
    patch_shape: tuple[int, int]
    addressable_corners: NDArray
    target_position: tuple[float, float] = (0.0, 0.0)
    overshoot: float = 0.0

    @property
    def is_addressable(self) -> bool:
        """True when the whole patch is inside the region the SLM can reach.

        A patch beyond it cannot be produced at all: the grating that would put light
        there aliases past the SLM's Nyquist frequency.
        """
        return self.overshoot <= 0.0

    def visualizer(self) -> TargetPlacementVisualizer:
        return TargetPlacementVisualizer(self)


class TargetPlacementVisualizer(BaseVisualizer):
    """Draw the placed target, the zeroth order, and the addressable limit."""

    def __init__(self, data: TargetPlacementData) -> None:
        self.data = data

    def default_layout(self) -> PlotLayout:
        height, width = np.asarray(self.data.target).shape
        layout = PlotLayout(column_width=7.0, margins=(0.9, 0.15, 0.5, 0.5))
        layout.add_row([GridCell("placement", aspect=height / width, colorbar=True)])
        return layout

    def panels(self) -> dict[str, Panel]:
        return {"placement": self._placement_panel}

    def _placement_panel(self, axs: Axes) -> ScalarMappable:
        from matplotlib.patches import Polygon, Rectangle

        data = self.data
        target = np.asarray(data.target)
        height, width = target.shape
        center_row, center_column = data.target_center
        zeroth_row, zeroth_column = data.zeroth_order
        patch_height, patch_width = data.patch_shape

        # Returned at the end: compose() only fills a cell's colorbar when the panel
        # hands back a mappable.
        mappable = self.draw_image(axs, target, cmap=INTENSITY_CMAP, title="")
        # draw_image clears the ticks, which suits a bare image. Here the pixel
        # coordinates are the point, since the whole panel is about where things sit.
        axs.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
        axs.set_xticks(np.linspace(0, width - 1, 5).astype(int))
        axs.set_yticks(np.linspace(0, height - 1, 5).astype(int))

        axs.add_patch(
            Rectangle(
                (center_column - patch_width / 2, center_row - patch_height / 2),
                patch_width, patch_height,
                fill=False, edgecolor="tab:cyan", linewidth=1.2, label="target patch",
            )
        )
        axs.plot(center_column, center_row, "+", color="tab:cyan", markersize=10)
        axs.plot(
            zeroth_column, zeroth_row, "o", markerfacecolor="none",
            color="tab:red", markersize=9, label="zeroth order",
        )
        # A rectangle rather than a circle: the x and y grating frequencies are limited
        # independently, so the corners are reachable too. Rotated, because it is a
        # rectangle in the optical plane rather than on the sensor.
        axs.add_patch(
            Polygon(
                np.asarray(data.addressable_corners)[:, ::-1],
                closed=True, fill=False, edgecolor="tab:red", linestyle="--",
                label="addressable (Nyquist) limit",
            )
        )
        # The addressable region is usually far larger than the sensor, so the view
        # stays on the sensor rather than letting it set the scale.
        axs.set_xlim(-0.5, width - 0.5)
        axs.set_ylim(height - 0.5, -0.5)

        offset_x, offset_y = data.target_position
        verdict = (
            "within the addressable region"
            if data.is_addressable
            else f"REACHES PAST the addressable limit by {data.overshoot * 1e6:.0f} um"
        )
        axs.set_title(
            f"target at ({offset_x * 1e6:+.0f}, {offset_y * 1e6:+.0f}) um from the "
            f"zeroth order\ncenter (row {center_row:.0f}, column {center_column:.0f}), "
            f"{verdict}"
        )
        axs.set_xlabel("camera column (+x)")
        axs.set_ylabel("camera row (+y)")
        axs.legend(loc="upper right", fontsize=8)
        return mappable


class CameraFeedbackVisualizer(BaseVisualizer):
    """Render a feedback run: what was asked for, what was measured, and the gap."""

    def __init__(self, data: CameraFeedbackData, iteration: int = -1) -> None:
        self.data = data
        self.iteration = iteration

    def _aspect(self, image: NDArray) -> float:
        shape = np.asarray(image).shape
        return shape[0] / shape[1]

    def _cropped_target(self) -> NDArray:
        return self.data.signal_roi.crop(np.asarray(self.data.target))

    def _cropped_region(self) -> NDArray:
        return self.data.signal_roi.crop(
            np.asarray(self.data.signal_region, dtype=bool)
        )

    def _measured(self, iteration: int | None = None) -> NDArray:
        """A measured frame, cropped to the region and on the target's scale."""
        index = self.iteration if iteration is None else iteration
        return normalize(
            np.asarray(self.data.measured_images[index]), self._cropped_region()
        )

    def best_iteration(self) -> int:
        """The iteration that scored best on the run's first metric, indexed from zero.

        The loop need not end on its best result: too high a gain for the last few
        corrections makes it overshoot, so the final frame can be worse than an earlier
        one.

        Keyed on the run's first metric, in the direction it recorded, so a figure of
        merit works as well as an error. Falls back to the iteration being shown when
        the run scored nothing.
        """
        primary = self._primary_metric()
        if primary is None:
            return self.iteration
        return self._best_index(*primary)

    def _primary_metric_name(self) -> str:
        """The primary metric's name for a title, or a generic word if there is none."""
        primary = self._primary_metric()
        return primary[0] if primary is not None else "score"

    def _primary_metric(self) -> tuple[str, list[float]] | None:
        """The first metric the run recorded any values for, name and history."""
        for name, history in self.data.metrics.items():
            if history:
                return name, history
        return None

    def _best_index(self, name: str, history: list[float]) -> int:
        """Where a metric reached its best value.

        Which end counts as best belongs to the metric: rmse wants its minimum, psnr and
        efficiency their maximum. Read from the direction the run recorded, and assumed
        to be the minimum for a name the record carries no flag for.
        """
        values = np.asarray(history, dtype=float)
        if self.data.lower_is_better.get(name, True):
            return int(np.argmin(values))
        return int(np.argmax(values))

    def _difference(self, iteration: int | None = None) -> NDArray:
        """Measured minus target over the signal region.

        Masked as well as cropped, since a region need not fill its own bounding box.
        """
        inside = self._cropped_region()
        difference = self._measured(iteration) - self._cropped_target()
        return np.where(inside, difference, np.nan)

    # --- Shared layout settings ------------------------------------------------

    @staticmethod
    def _layout() -> PlotLayout:
        """An empty layout with this figure's margins, for one row or for all four.

        Wide left and bottom margins so the convergence row's axis labels fit.
        """
        return PlotLayout(column_width=3.6, margins=(1.0, 0.15, 0.5, 0.5))

    @staticmethod
    def _metric_key(name: str) -> str:
        """Cell key for a metric, kept distinct from the image cells."""
        return f"metric:{name}"

    def _inside_fraction(self, image: NDArray) -> str:
        """How much of an image's light lands inside the signal region, as a title tag.

        The number that says whether a starting guess is usable.
        """
        inside = np.asarray(self.data.signal_region, dtype=bool)
        total = float(np.nansum(image))
        if total <= 0:
            return ""
        fraction = float(np.nansum(image[inside])) / total
        return f" (fraction inside the region {fraction:.3f})"

    # --- Row 1: what was asked for, and where the search started ------------------

    def _target_cells(self) -> list[GridCell]:
        camera_aspect = self._aspect(self.data.target)
        cells = [GridCell("target", aspect=camera_aspect, colorbar=True)]
        if self.data.initial_guess is not None:
            cells.append(GridCell("initial", aspect=camera_aspect, colorbar=True))
        return cells

    def _target_panels(self) -> dict[str, Panel]:
        target = np.asarray(self.data.target)
        panels = {"target": self._image_panel(target, "target on the camera grid")}
        if self.data.initial_guess is None:
            return panels

        guess = np.asarray(self.data.initial_guess)
        # Its own color scale: the guess is the model's prediction in physical units
        # and the target a normalized profile. What is compared is where the light sits.
        panels["initial"] = self._image_panel(
            guess, f"initial guess{self._inside_fraction(guess)}"
        )
        return panels

    def _target_layout(self) -> PlotLayout:
        return self._layout().add_row(self._target_cells())

    def render_target(self, **kwargs) -> Figure:
        """Draw the target beside the potential the starting phase alone produces."""
        return self.render(
            layout=self._target_layout(), panels=self._target_panels(), **kwargs
        )

    # --- Row 2: the hologram and the frame it produced ----------------------------

    def _hologram_cells(self) -> list[GridCell]:
        cells = [
            GridCell(
                "phase",
                aspect=self._aspect(self.data.retrievals[self.iteration].phase),
                colorbar=True,
            )
        ]
        if self.data.final_camera_image is not None:
            cells.append(
                GridCell(
                    "camera",
                    aspect=self._aspect(self.data.final_camera_image),
                    colorbar=True,
                )
            )
        return cells

    def _hologram_panels(self) -> dict[str, Panel]:
        panels = {
            # Wrapped, because that is what the SLM displays.
            "phase": self._image_panel(
                np.asarray(self.data.retrievals[self.iteration].phase) % (2 * np.pi),
                f"SLM phase [rad], iteration {self._number()}",
                cmap=PHASE_CMAP,
            )
        }
        if self.data.final_camera_image is not None:
            # The whole sensor in counts, which is where light outside the signal
            # region shows up.
            panels["camera"] = self._image_panel(
                np.asarray(self.data.final_camera_image), "camera image [counts]"
            )
        return panels

    def _hologram_layout(self) -> PlotLayout:
        return self._layout().add_row(self._hologram_cells())

    def render_hologram(self, **kwargs) -> Figure:
        """Draw the hologram and the camera frame it produced, side by side."""
        return self.render(
            layout=self._hologram_layout(), panels=self._hologram_panels(), **kwargs
        )

    # --- Row 3: the signal region, before and after -------------------------------

    def _region_cells(self) -> list[GridCell]:
        aspect = self._aspect(self._difference())
        return [
            GridCell("first", aspect=aspect, colorbar=True),
            GridCell("lowest", aspect=aspect, colorbar=True),
            GridCell("difference", aspect=aspect, colorbar=True),
        ]

    def _region_panels(self) -> dict[str, Panel]:
        target = self._cropped_target()
        inside = self._cropped_region()

        best = self.best_iteration()
        first = self._measured(0)
        lowest = self._measured(best)
        difference = self._difference(best)

        residual = float(np.sqrt(np.mean((lowest - target)[inside] ** 2)))
        peak = max(float(np.nanmax(first)), float(np.nanmax(lowest)))
        limit = float(np.nanmax(np.abs(difference))) if inside.any() else 1.0

        return {
            # One scale across both, so the improvement shows.
            "first": self._image_panel(first, "iteration 1", vmax=peak),
            "lowest": self._image_panel(
                lowest,
                f"iteration {best + 1}, best {self._primary_metric_name()}",
                vmax=peak,
            ),
            "difference": self._image_panel(
                difference,
                f"measured - target (rms {residual:.3f})",
                cmap=DIFFERENCE_CMAP,
                vmin=-limit,
                vmax=limit,
            ),
        }

    def _region_layout(self) -> PlotLayout:
        return self._layout().add_row(self._region_cells())

    def render_signal_region(self, **kwargs) -> Figure:
        """Draw the signal region before and after feedback, and what is left over."""
        return self.render(
            layout=self._region_layout(), panels=self._region_panels(), **kwargs
        )

    # --- Row 4: convergence -------------------------------------------------------

    def _convergence_cells(self) -> list[GridCell]:
        return [
            GridCell(self._metric_key(name), aspect="auto", height=2.2)
            for name in self.data.metrics
        ]

    def _convergence_panels(self) -> dict[str, Panel]:
        return {
            self._metric_key(name): self._history_panel(history, name)
            for name, history in self.data.metrics.items()
        }

    def _convergence_layout(self) -> PlotLayout:
        return self._layout().add_row(self._convergence_cells())

    def render_convergence(self, **kwargs) -> Figure:
        """Draw one curve per metric against feedback iteration."""
        return self.render(
            layout=self._convergence_layout(),
            panels=self._convergence_panels(),
            **kwargs,
        )

    # --- All four together --------------------------------------------------------

    def default_layout(self) -> PlotLayout:
        layout = self._layout()
        for cells in (
            self._target_cells(),
            self._hologram_cells(),
            self._region_cells(),
            self._convergence_cells(),
        ):
            layout.add_row(cells)
        return layout

    def panels(self) -> dict[str, Panel]:
        return {
            **self._target_panels(),
            **self._hologram_panels(),
            **self._region_panels(),
            **self._convergence_panels(),
        }

    def _number(self) -> int:
        """The iteration being shown, counting from one, whichever way it was asked
        for.
        """
        return self.iteration + 1 if self.iteration >= 0 else (
            self.data.number_of_iterations + self.iteration + 1
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
        if vmin is None and vmax is not None:
            vmin = 0.0
        return lambda axs: self.draw_image(
            axs, image, cmap=cmap, vmin=vmin, vmax=vmax, title=title
        )

    def _history_panel(self, history: list[float], label: str) -> Panel:
        """One metric against iteration, with its best value starred.

        Discrete markers, since the iterations are separate measurements.
        """

        def panel(axs: Axes) -> None:
            if not history:
                axs.set_axis_off()
                axs.set_title(f"{label} (not recorded)")
                return

            iterations = np.arange(1, len(history) + 1)
            self.draw_line(
                axs,
                [
                    {
                        "x": iterations,
                        "y": list(history),
                        "style": "o",
                        "color": foreground_color(),
                    }
                ],
                xlabel="feedback iteration",
                title=label,
            )
            best = self._best_index(label, history)
            axs.plot(best + 1, history[best], "*", color="tab:red", markersize=14)

        return panel

    def render_iteration(self, iteration: int, **kwargs) -> Figure:
        """Render a different iteration of the same run."""
        return CameraFeedbackVisualizer(self.data, iteration).render(**kwargs)
