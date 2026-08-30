"""Visualize a coarse camera-mapping search: a :class:`BaseVisualizer` over its
:class:`CoarseVisualizationData`.

Follows the same ``VisualizationData`` pattern as the raster calibrator
(``RasterVisualizationData`` / ``RasterCalibratorVisualizer``): ``CoarseMapper``
records a self-contained :class:`CoarseVisualizationData`, attaches it to
``CameraMapping.visualization_data``, and this visualizer renders it -- four
panels: a schematic of the SLM Nyquist output plane (the addressable zone, the
candidate spiral spots, the zeroth order and where the camera sensor sits) and
the camera captures of the three search stages (the full calibration spot array,
the spot-walking trail, and the final four affine probes).
"""

from __future__ import annotations

from dataclasses import dataclass

from ....grids import plane_center
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from ....visualizer import (
    INTENSITY_CMAP,
    BaseVisualizer,
    GridCell,
    Panel,
    PlotLayout,
)
from ....serialization import record_type
from ..visualizer import CameraMappingVisualizationData

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.cm import ScalarMappable
    from matplotlib.figure import Figure

# TODO: Sanity check and tidy up
@record_type("coarse_visualization")
@dataclass(kw_only=True)
class CoarseVisualizationData(CameraMappingVisualizationData):
    """Everything ``CoarseMapper.map_camera`` records for visualization.

    The shared frames plus the coarse search's own captures, attached to
    ``CameraMapping.visualization_data``. Pixel coordinates are output-plane (model)
    pixels in ``(x, y)`` order. ``output_resolution`` is ``(height, width)``.
    """

    #: Camera capture of the full calibration spot array, or None when the zeroth order
    #: was on the sensor, in which case no array was displayed.
    array_image: NDArray | None
    #: Max-projection of the center-search captures, the probe spot's trail as it walks
    #: towards the sensor center, or None.
    walk_image: NDArray | None
    #: Max-projection of the four affine-probe captures.
    probe_image: NDArray
    #: Fitted probe positions in camera pixels, ``(N, 2)``.
    detected_points: NDArray
    #: Candidate spiral spot positions, ``(N, 2)``.
    array_spot_positions: NDArray
    #: The four affine probes in output-plane pixels.
    affine_probe_positions: NDArray
    #: SLM Nyquist half-extent ``(x, y)``.
    nyquist_half_extent_px: tuple[float, float]
    #: Output-plane resolution ``(height, width)``.
    output_resolution: tuple[int, int]
    #: The four camera-sensor corners, ``(4, 2)``.
    sensor_rectangle: NDArray


class CoarseMapperVisualizer(BaseVisualizer):
    """Render a coarse camera-mapping search from its
    :class:`CoarseVisualizationData`.

    >>> coarse = CoarseMapper(slm, camera, model).map_camera()
    >>> CoarseMapperVisualizer(coarse.visualization_data).render()

    Four panels: the Nyquist output-plane schematic plus the camera captures of
    the full spot array, the spot-walking trail, and the final four probes.
    """

    def __init__(self, data: CoarseVisualizationData) -> None:
        self.data = data

    def default_layout(self) -> PlotLayout:
        # All camera captures share the sensor shape; the 4-probe composite is
        # always present, so use it to size the image cells.
        camera_shape = np.asarray(self.data.probe_image).shape
        aspect = camera_shape[0] / camera_shape[1]
        layout = PlotLayout(column_width=3.6)
        layout.add_row([GridCell("plane", colspan=3, aspect="auto", height=4.5)])
        layout.add_row(
            [
                GridCell("array", aspect=aspect, colorbar=True),
                GridCell("walk", aspect=aspect, colorbar=True),
                GridCell("probes", aspect=aspect, colorbar=True),
            ]
        )
        return layout

    def _plane_panel(self, axs: Axes) -> None:
        """Schematic of the output plane: Nyquist zone, candidate spots, zeroth
        order and camera-sensor footprint (all in output-plane pixels).
        """
        import matplotlib.patches as patches

        data = self.data
        half_x, half_y = data.nyquist_half_extent_px
        center = plane_center(data.output_resolution)

        axs.add_patch(
            patches.Rectangle(
                (center[0] - half_x, center[1] - half_y),
                2 * half_x, 2 * half_y, fill=False, edgecolor="crimson",
                linestyle="--", linewidth=1.5, label="SLM Nyquist zone",
            )
        )
        spots = np.asarray(data.array_spot_positions, dtype=float)
        axs.plot(
            spots[:, 0], spots[:, 1], "o", color="0.6", markersize=3,
            markeredgecolor="none", label="candidate spots",
        )
        # Number the candidate spots in the order they are searched/displayed.
        for order, (spot_x, spot_y) in enumerate(spots, start=1):
            axs.annotate(
                str(order), (spot_x, spot_y), textcoords="offset points",
                xytext=(3, 3), fontsize=6, color="0.3", ha="left", va="bottom",
            )
        probes = np.asarray(data.affine_probe_positions, dtype=float)
        if probes.size:
            axs.plot(
                probes[:, 0], probes[:, 1], "o", color="C0", markersize=7,
                markeredgecolor="white", label="affine probes",
            )
        # The zeroth order (undiffracted DC, tilt = 0) sits at the output-plane
        # center by definition.
        axs.plot(
            [center[0]], [center[1]], "+", color="black", markersize=12,
            markeredgewidth=2, label="zeroth order",
        )
        polygon = np.asarray(data.sensor_rectangle, dtype=float)
        axs.add_patch(
            patches.Polygon(
                polygon, closed=True, facecolor="C1", alpha=0.25,
                edgecolor="C1", linewidth=1.5, label="camera sensor",
            )
        )

        xs = np.concatenate([[center[0] - half_x, center[0] + half_x], polygon[:, 0]])
        ys = np.concatenate([[center[1] - half_y, center[1] + half_y], polygon[:, 1]])
        pad = 0.05 * max(float(np.ptp(xs)), float(np.ptp(ys)), 1.0)
        axs.set_xlim(xs.min() - pad, xs.max() + pad)
        axs.set_ylim(ys.min() - pad, ys.max() + pad)
        axs.set_aspect("equal")
        axs.invert_yaxis()  # image coordinates (y grows downward)
        axs.set_xlabel("output-plane x [px]")
        axs.set_ylabel("output-plane y [px]")
        axs.set_title("Nyquist output plane, spots and camera sensor")
        axs.legend(loc="upper right", fontsize=7, framealpha=0.9)
        return None

    @staticmethod
    def _image_panel(image: NDArray | None, title: str, placeholder: str) -> Panel:
        """Panel that shows ``image`` (or ``placeholder`` text when None)."""
        def panel(axs: Axes) -> ScalarMappable | None:
            if image is None:
                axs.set_xticks([])
                axs.set_yticks([])
                axs.text(
                    0.5, 0.5, placeholder, ha="center", va="center",
                    transform=axs.transAxes, fontsize=9, color="0.4",
                )
                axs.set_title(title)
                return None
            return BaseVisualizer.draw_image(
                axs, image, cmap=INTENSITY_CMAP, title=title
            )

        return panel

    def _probes_panel(self, axs: Axes) -> ScalarMappable:
        """The 4-probe composite with the detected spot positions overlaid."""
        mappable = BaseVisualizer.draw_image(
            axs, np.asarray(self.data.probe_image), cmap=INTENSITY_CMAP,
            title="Final 4 affine probes",
        )
        detected = np.asarray(self.data.detected_points, dtype=float)
        if detected.size:
            BaseVisualizer.draw_points(
                axs, detected[:, 0], detected[:, 1], color="red", size=6,
            )
        return mappable

    def panels(self) -> dict[str, Panel]:
        data = self.data
        return {
            "plane": self._plane_panel,
            "array": self._image_panel(
                data.array_image, "Full spot array",
                "zeroth order on sensor:\nno array displayed",
            ),
            "walk": self._image_panel(
                data.walk_image, "Spot-walking (max-projection)",
                "no walk captured",
            ),
            "probes": self._probes_panel,
        }

    def render(self, **kwargs) -> Figure:
        kwargs.setdefault("suptitle", "Coarse camera mapping: search stages")
        return super().render(**kwargs)
