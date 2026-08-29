"""Rendering a vortex annihilation run."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from ...serialization import record_type
from ...visualizer import (
    INTENSITY_CMAP,
    PHASE_CMAP,
    BaseVisualizer,
    GridCell,
    Panel,
    PlotLayout,
    foreground_color,
    region_bounding_box,
)
from ...visualizer import VisualizationData

if TYPE_CHECKING:
    from matplotlib.axes import Axes

TWO_PI = 2 * np.pi

#: Marker colours for a positive and a negative topological charge.
POSITIVE_COLOR = "#ff3b30"
NEGATIVE_COLOR = "#00e5ff"


@record_type("vortex_annihilation")
@dataclass
class VortexAnnihilationData(VisualizationData):
    """What an annihilation run records, before and after.

    A phase vortex is a point where the image-plane phase winds by a whole cycle, so
    the intensity there is forced to zero. Annihilation finds them, multiplies in a
    field of opposite charge, and retrieves again.

    ``counts`` holds the number found at each round, so the last entry is zero when the
    run converged and non-zero when it stopped at its iteration limit.

    ``signal_region`` is what the retrieval was evaluated over, and is used to crop the
    panels down to it. Vortices are only hunted inside it, so a full-frame view spends
    most of its area on places none were ever looked for.
    """

    counts: list[int] = field(default_factory=list)
    signal_region: NDArray | None = None
    initial_intensity: NDArray | None = None
    initial_phase: NDArray | None = None
    initial_positions: NDArray | None = None
    initial_charges: NDArray | None = None
    final_intensity: NDArray | None = None
    final_phase: NDArray | None = None
    final_positions: NDArray | None = None
    final_charges: NDArray | None = None

    @property
    def converged(self) -> bool:
        """Whether the run ended because nothing was left to annihilate."""
        return bool(self.counts) and self.counts[-1] == 0

    def visualizer(self) -> VortexAnnihilationVisualizer:
        """The visualizer that draws this run."""
        return VortexAnnihilationVisualizer(self)


class VortexAnnihilationVisualizer(BaseVisualizer):
    """Draw where the vortices were, where they went, and how fast they went.

    The intensity and the phase are shown for the same field, since a vortex is only
    obvious in both at once: a null in the intensity sitting on a point the phase winds
    around.
    """

    def __init__(self, data: VortexAnnihilationData) -> None:
        """
        Args:
            data: The recorded run.
        """
        self.data = data

    def _crop(self) -> tuple[slice, slice]:
        """The signal region, with a margin, as row and column slices."""
        if self.data.signal_region is None:
            return slice(None), slice(None)
        return region_bounding_box(self.data.signal_region)

    def _aspect(self) -> float:
        rows, columns = self._crop()
        shape = np.asarray(self.data.initial_intensity)[rows, columns].shape
        return shape[0] / shape[1]

    def default_layout(self) -> PlotLayout:
        aspect = self._aspect()
        layout = PlotLayout(column_width=3.6, margins=(1.0, 0.15, 0.5, 0.5))
        layout.add_row(
            [
                GridCell("before_intensity", aspect=aspect, colorbar=True),
                GridCell("before_phase", aspect=aspect, colorbar=True),
            ]
        )
        layout.add_row(
            [
                GridCell("after_intensity", aspect=aspect, colorbar=True),
                GridCell("after_phase", aspect=aspect, colorbar=True),
            ]
        )
        layout.add_row([GridCell("counts", colspan=2, aspect="auto", height=2.2)])
        return layout

    def panels(self) -> dict[str, Panel]:
        found = _count(self.data.initial_positions)
        left = _count(self.data.final_positions)
        return {
            "before_intensity": self._marked(
                self.data.initial_intensity,
                self.data.initial_positions,
                self.data.initial_charges,
                f"intensity, {found} vortices",
                INTENSITY_CMAP,
            ),
            "before_phase": self._marked(
                self.data.initial_phase,
                self.data.initial_positions,
                self.data.initial_charges,
                "phase [rad]",
                PHASE_CMAP,
            ),
            "after_intensity": self._marked(
                self.data.final_intensity,
                self.data.final_positions,
                self.data.final_charges,
                f"intensity, {left} left",
                INTENSITY_CMAP,
            ),
            "after_phase": self._marked(
                self.data.final_phase,
                self.data.final_positions,
                self.data.final_charges,
                "phase [rad]",
                PHASE_CMAP,
            ),
            "counts": self._counts_panel,
        }

    def _marked(
        self,
        image: NDArray | None,
        positions: NDArray | None,
        charges: NDArray | None,
        title: str,
        cmap: str,
    ) -> Panel:
        """One image with its vortices marked, coloured by the sign of their charge.

        The image is cropped to the signal region, so the marker coordinates are
        shifted to match and any vortex outside the crop is dropped.
        """
        crop_rows, crop_columns = self._crop()

        def panel(axs: Axes):
            cropped = np.asarray(image)[crop_rows, crop_columns]
            mappable = self.draw_image(axs, cropped, cmap=cmap, title=title)
            if positions is None or len(positions) == 0:
                return mappable

            rows, columns = np.asarray(positions).T
            rows = rows - (crop_rows.start or 0)
            columns = columns - (crop_columns.start or 0)
            visible = (
                (rows >= 0)
                & (rows < cropped.shape[0])
                & (columns >= 0)
                & (columns < cropped.shape[1])
            )
            signs = (
                np.ones(len(rows))
                if charges is None
                else np.sign(np.asarray(charges).reshape(-1))
            )
            for sign, colour, label in (
                (1, POSITIVE_COLOR, "charge +1"),
                (-1, NEGATIVE_COLOR, "charge -1"),
            ):
                keep = (signs == sign) & visible
                if not keep.any():
                    continue
                # Hollow, and small enough to sit on a single vortex. A retrieval can
                # leave hundreds of them a few pixels apart, and a filled marker wide
                # enough to see hides both its own vortex and its neighbours.
                self.draw_points(
                    axs,
                    columns[keep],
                    rows[keep],
                    marker="o",
                    color=colour,
                    size=4,
                    edgecolor=colour,
                    label=label,
                    legend=True,
                    markerfacecolor="none",
                    markeredgewidth=0.9,
                )
            return mappable

        return panel

    def _counts_panel(self, axs: Axes) -> None:
        counts = list(self.data.counts)
        if not counts:
            axs.set_axis_off()
            axs.set_title("vortex count (not recorded)")
            return

        rounds = np.arange(1, len(counts) + 1)
        self.draw_line(
            axs,
            [
                {
                    "x": rounds,
                    "y": counts,
                    "style": "-o",
                    "color": foreground_color(),
                    "label": "detected",
                }
            ],
            xlabel="annihilation round",
            ylabel="vortices detected",
            title=(
                "converged"
                if self.data.converged
                else "stopped at the iteration limit"
            ),
        )
        axs.set_xticks(rounds)


def _count(positions: NDArray | None) -> int:
    """How many vortices a recorded position array holds."""
    return 0 if positions is None else len(positions)
