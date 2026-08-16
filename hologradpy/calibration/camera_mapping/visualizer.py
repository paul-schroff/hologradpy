"""Visualize a :class:`CameraMapping` (from any camera mapper)."""

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
    PlotBuilder,
    PlotLayout,
    VisualizationData,
)

if TYPE_CHECKING:
    from matplotlib.figure import Figure

    from .mapping import CameraMapping


@record_type("camera_mapping_visualization")
@dataclass
class CameraMappingVisualizationData(VisualizationData):
    """The frames a mapping was fit from.

    What every mapper records, and all :class:`CameraMapperVisualizer` needs. A mapper
    with more to show subclasses this and its own visualizer draws the rest.

    Attributes:
        camera_image: What the camera saw.
        simulated_image: What the model predicted for the same pattern.
        zeroth_order_mask: True over the camera pixels a mapper left out of both the
            exposure and the detection, None when it masked nothing.
    """

    camera_image: NDArray
    simulated_image: NDArray
    zeroth_order_mask: NDArray | None = None


# TODO: Sanity check and tidy up
class CameraMapperVisualizer(BaseVisualizer):
    """A general visualizer over a :class:`CameraMapping`.

    Renders the camera and simulated images with their point correspondences, the
    reprojection residuals of the fitted affine transform. When the mapping
    carries per-spot Gaussian fits (``SpotArrayMapper``), renders the fitted waist of
    each spot with the uncertainty-weighted mean. Works for any mapper's mapping (the
    spot-fit panel is only drawn when the fits are present).

    Needs the frames, so a mapping that has been through
    :meth:`~hologradpy.calibration.camera_mapping.CameraMapping.lean` cannot be drawn.
    """

    def __init__(self, mapping: CameraMapping) -> None:
        if mapping.visualization_data is None:
            raise ValueError(
                f"The {mapping.name!r} mapping carries no frames to draw. A mapping "
                "embedded in another record has been through lean(); plot the one the "
                "mapper returned instead."
            )
        self.mapping = mapping
        self.frames: CameraMappingVisualizationData = mapping.visualization_data

    def default_layout(self) -> PlotLayout:
        camera_shape = np.asarray(self.frames.camera_image).shape
        simulated_shape = np.asarray(self.frames.simulated_image).shape

        layout = PlotLayout(column_width=4.0)
        layout.add_row(
            [
                GridCell("camera", aspect=camera_shape[0] / camera_shape[1],
                         colorbar=True),
                GridCell("simulated", aspect=simulated_shape[0] / simulated_shape[1],
                         colorbar=True),
            ]
        )
        layout.add_row([GridCell("residual", colspan=2, aspect="auto", height=3.0)])
        if self.mapping.spot_fit.parameters is not None:
            layout.add_row([GridCell("waist", colspan=2, aspect="auto", height=2.5)])
        return layout

    def render(self) -> Figure:
        """Build the figure for this mapping."""
        mapping = self.mapping
        detected = np.asarray(mapping.detected_points, dtype=float)
        calculated = np.asarray(mapping.calculated_points, dtype=float)

        residual_vectors = np.asarray(mapping.fit.reprojection_errors, dtype=float)
        rms = float(mapping.fit.reprojection_rms)
        residuals = np.linalg.norm(residual_vectors, axis=1)

        # Magnify the (sub-pixel) residual arrows to ~10% of the array span so
        # their structure is visible; the factor is stated in the title.
        span = max(np.ptp(calculated[:, 0]), np.ptp(calculated[:, 1]), 1.0)
        magnification = 0.1 * span / max(float(residuals.max()), 1e-12)

        builder = (
            PlotBuilder(self.default_layout())
            .draw_image(
                "camera", np.asarray(self.frames.camera_image), cmap=INTENSITY_CMAP,
                title="Camera image + detected spots",
            )
            .draw_points(
                "camera", detected[:, 0], detected[:, 1], color="red", size=5,
                label="matched spots",
            )
            .draw_image(
                "simulated", np.asarray(self.frames.simulated_image),
                cmap=INTENSITY_CMAP,
                title="Simulated image + calculated spots",
            )
            .draw_points(
                "simulated", calculated[:, 0], calculated[:, 1], color="cyan", size=5
            )
            .draw_quiver(
                "residual",
                calculated[:, 0],
                calculated[:, 1],
                residual_vectors[:, 0],
                residual_vectors[:, 1],
                scale=1.0 / magnification,
                xlabel="x [simulated px]",
                ylabel="y [simulated px]",
                title=(
                    f"Reprojection residuals (RMS = {rms:.2f} px, "
                    f"arrows x{magnification:.0f})"
                ),
                invert_y=True,
            )
        )

        # Detections that were found but excluded from the transform (poor fit,
        # no target match, or affine outlier) -- marked so no spot disappears
        # silently.
        excluded = mapping.fit.excluded_points
        if excluded is not None and len(excluded) > 0:
            excluded = np.asarray(excluded, dtype=float)
            builder.draw_points(
                "camera",
                excluded[:, 0],
                excluded[:, 1],
                marker="x",
                color="lightgray",
                edgecolor="lightgray",
                size=6,
                label="detected, excluded",
                legend=True,
            )

        spot_fit = mapping.spot_fit
        if spot_fit.parameters is not None:
            waists = np.array([p[0] for p in spot_fit.parameters]) * 1e6
            errors = np.array(
                [np.sqrt(c[0, 0]) for c in spot_fit.covariances]
            ) * 1e6
            average = spot_fit.waist * 1e6
            builder.draw_line(
                "waist",
                [{"x": np.arange(len(waists)), "y": waists, "yerr": errors,
                  "style": "o", "color": "C1", "label": "fitted waist"}],
                hlines=(average,),
                xlabel="Spot index",
                ylabel="Waist [um]",
                title=f"Per-spot waist (weighted mean = {average:.2f} um)",
                legend=True,
            )

        return builder.build(suptitle=f"CameraMapping: {mapping.name}")

