"""Visualize a raster phase scan: a :class:`BaseVisualizer` over the data.

Consumes a :class:`RasterVisualizationData` (recorded by
``RasterCalibrator.measure_phase`` and carried on
``WavefrontCalibrationData.visualization_data``) and renders the per-superpixel scan as
an animation / GIF, plus a static drift-tracking figure. All layout and GIF machinery
lives in :class:`~hologradpy.visualizer.BaseVisualizer`; this class only supplies the
content draw methods and the default layout.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from ....visualizer import (
    INTENSITY_CMAP,
    PHASE_CMAP,
    AnimatedVisualizer,
    GridCell,
    Panel,
    PlotBuilder,
    PlotLayout,
    VisualizationData,
)
from ....serialization import record_type

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.cm import ScalarMappable
    from matplotlib.figure import Figure


@record_type("raster_visualization")
@dataclass
class RasterVisualizationData(VisualizationData):
    """Everything ``RasterCalibrator.measure_phase`` records for visualization.

    A concrete :class:`~hologradpy.visualizer.VisualizationData`. Produced by the
    calibrator, consumed by ``RasterCalibratorVisualizer`` and attached to
    ``WavefrontCalibrationData.visualization_data`` by ``calibrate``. The images and
    per-superpixel series are in scan order (aligned with ``superpixel_coordinates``).
    Lattice fields are ``None`` unless pointing compensation was on;
    ``displayed_slm_phases`` is ``None`` unless ``record_displayed_phases`` was
    requested (it is one full-resolution frame per superpixel). ``full_frame_image``
    is a single full-sensor snapshot of the first scan frame (interference pattern,
    optical lattice and zeroth order together), also recorded only when
    ``record_displayed_phases`` was requested. ``full_frame_marker_positions`` maps
    each feature label to its ``(x, y)`` camera pixel (``None`` for a feature absent
    from the scan, e.g. the lattice without pointing compensation).
    """

    camera_images: NDArray
    fitted_images: NDArray
    measured_phase: NDArray
    superpixel_coordinates: NDArray
    lattice_images: NDArray | None = None
    fitted_lattice_images: NDArray | None = None
    lattice_shift_x: NDArray | None = None
    lattice_shift_y: NDArray | None = None
    lattice_shift_x_err: NDArray | None = None
    lattice_shift_y_err: NDArray | None = None
    displayed_slm_phases: NDArray | None = None
    full_frame_image: NDArray | None = None
    full_frame_marker_positions: dict[str, tuple[float, float] | None] | None = None


class RasterCalibratorVisualizer(AnimatedVisualizer):
    """Render a raster phase scan from its :class:`RasterVisualizationData`.

    >>> phase, _, _ = calibrator.measure_phase(
    ...     20, 16, 32, 32, (500e-6, 500e-6), measured_intensity=intensity,
    ...     compensate_pointing=True, lattice_phase_tilt=(-800e-6, -800e-6),
    ...     record_displayed_phases=True,
    ... )
    >>> viz = RasterCalibratorVisualizer(calibrator.visualization_data)
    >>> viz.save_gif("scan.gif")
    >>> viz.plot_drift_tracking()

    (``calibrator.visualization_data`` and a saved
    ``WavefrontCalibrationData.visualization_data`` are the same object.)

    Phase maps render at full resolution; set ``phase_downsample`` > 1 to subsample them
    when rendering if the frames are too large.
    """

    def __init__(
        self,
        data: RasterVisualizationData,
        phase_downsample: int = 1,
    ) -> None:
        self.data = data
        self.phase_downsample = phase_downsample

    @property
    def _has_lattice(self) -> bool:
        return self.data.lattice_images is not None

    def _phase_frame(self, array) -> np.ndarray:
        step = self.phase_downsample
        return np.asarray(array)[::step, ::step]

    def frame_count(self) -> int:
        return self.data.camera_images.shape[0]

    def frame_suptitle(self, frame: int) -> str:
        return f"Superpixel {frame + 1}/{self.frame_count()}"

    def panels_for_frame(self, frame: int) -> dict[str, Panel]:
        data = self.data
        panels: dict[str, Panel] = {
            "displayed_phase": lambda axs: self._draw_displayed_phase(axs, frame),
            "measured_phase": lambda axs: self._draw_measured_phase(axs, frame),
            "interference": lambda axs: self._draw_camera_pair(
                axs, frame, data.camera_images, data.fitted_images,
                show_fit=False, title="interference"),
            "interference_fit": lambda axs: self._draw_camera_pair(
                axs, frame, data.camera_images, data.fitted_images,
                show_fit=True, title="interference fit"),
        }
        if self._has_lattice:
            panels["lattice"] = lambda axs: self._draw_camera_pair(
                axs, frame, data.lattice_images, data.fitted_lattice_images,
                show_fit=False, title="optical lattice")
            panels["lattice_fit"] = lambda axs: self._draw_camera_pair(
                axs, frame, data.lattice_images, data.fitted_lattice_images,
                show_fit=True, title="lattice fit")
        return panels

    def _draw_displayed_phase(self, axs: Axes, frame: int) -> ScalarMappable:
        return self.draw_image(
            axs,
            self._phase_frame(self.data.displayed_slm_phases[frame]),
            cmap=PHASE_CMAP,
            title="displayed SLM phase",
        )

    def _draw_measured_phase(self, axs: Axes, frame: int) -> ScalarMappable:
        mappable = self.draw_image(
            axs,
            self._phase_frame(self.data.measured_phase),
            cmap=PHASE_CMAP,
            title="measured SLM phase",
        )
        coordinates = self.data.superpixel_coordinates
        if coordinates is not None:
            self.draw_points(
                axs,
                coordinates[0, frame] / self.phase_downsample,
                coordinates[1, frame] / self.phase_downsample,
            )
        return mappable

    def _draw_camera_pair(self, axs, frame, images, fits, *, show_fit, title):
        shared_max = max(
            float(images[frame].max()), float(fits[frame].max()), 1.0
        )
        array = fits[frame] if show_fit else images[frame]
        return self.draw_image(
            axs, array, cmap=INTENSITY_CMAP, vmin=0, vmax=shared_max, title=title
        )

    def default_layout(self) -> PlotLayout:
        if self.data.displayed_slm_phases is None:
            raise RuntimeError(
                "No displayed SLM phases recorded; run measure_phase(..., "
                "record_displayed_phases=True) before visualizing the scan."
            )
        # Phase maps keep the SLM's true aspect ratio (height / width) and each spans
        # the width of the camera panels beneath them.
        height, width = np.asarray(self.data.measured_phase).shape
        ratio = height / width
        span = 2 if self._has_lattice else 1

        layout = PlotLayout()
        layout.add_row(
            [
                GridCell("displayed_phase", colspan=span, aspect=ratio),
                GridCell("measured_phase", colspan=span, aspect=ratio),
            ]
        )
        camera = [GridCell("interference"), GridCell("interference_fit")]
        if self._has_lattice:
            camera += [GridCell("lattice"), GridCell("lattice_fit")]
        layout.add_row(camera)
        return layout

    def plot_drift_tracking(
        self, injected_x=None, injected_y=None
    ) -> Figure:
        """Plot the lattice-tracked pointing drift [um] over the scan.

        If ``injected_x``/``injected_y`` (metres, per superpixel) are given, the
        injected drift is overlaid (dashed) and a residual panel is added.
        """
        data = self.data
        if data.lattice_shift_x is None:
            raise RuntimeError(
                "No lattice drift recorded; run measure_phase with "
                "compensate_pointing=True."
            )

        frames = np.arange(len(data.lattice_shift_x))
        shift_x = data.lattice_shift_x * 1e6
        shift_y = data.lattice_shift_y * 1e6
        shift_x_err = data.lattice_shift_x_err * 1e6
        shift_y_err = data.lattice_shift_y_err * 1e6
        has_injected = injected_x is not None and injected_y is not None

        layout = PlotLayout(column_width=7.0, margins=(0.7, 0.2, 0.4, 0.6))
        layout.add_row([GridCell("track", aspect="auto", height=4.0)])
        if has_injected:
            layout.add_row(
                [GridCell("resid", aspect="auto", height=2.0, sharex="track")]
            )

        track_curves = [
            {"x": frames, "y": shift_x, "yerr": shift_x_err,
             "color": "C0", "label": "tracked x"},
            {"x": frames, "y": shift_y, "yerr": shift_y_err,
             "color": "C1", "label": "tracked y"},
        ]
        if has_injected:
            track_curves += [
                {"x": frames, "y": injected_x * 1e6, "style": "--",
                 "color": "C0", "label": "injected x"},
                {"x": frames, "y": injected_y * 1e6, "style": "--",
                 "color": "C1", "label": "injected y"},
            ]

        builder = PlotBuilder(layout).draw_line(
            "track", track_curves,
            ylabel="Pointing drift [um]",
            title="Optical lattice tracks pointing drift",
            legend=True,
        )
        if has_injected:
            builder.draw_line(
                "resid",
                [
                    {"x": frames, "y": shift_x - injected_x * 1e6,
                     "yerr": shift_x_err, "color": "C0", "label": "x"},
                    {"x": frames, "y": shift_y - injected_y * 1e6,
                     "yerr": shift_y_err, "color": "C1", "label": "y"},
                ],
                hlines=(0.0,),
                xlabel="Superpixel index",
                ylabel="Residual [um]",
                legend=True,
            )
        return builder.build()

    def plot_full_frame(
        self, cmap: str = INTENSITY_CMAP, vmax: float | None = None
    ) -> Figure:
        """Plot the full-sensor snapshot of the first scan frame.

        The snapshot shows the interference pattern, the optical lattice and the
        zeroth order together on the camera, with a linear colorbar in raw camera
        counts. Each recorded feature is marked and labeled, so the zeroth order and
        the (dim, edge-lit) lattice can be located even when they sit near the noise
        floor. By default the color scale is clipped to a small fraction of the peak
        so the bright interference spot saturates and those faint features come up;
        pass an explicit ``vmax`` (for example the image maximum) to override.
        """
        import matplotlib.pyplot as plt

        if self.data.full_frame_image is None:
            raise RuntimeError(
                "No full-frame snapshot recorded; run measure_phase with "
                "record_displayed_phases=True."
            )
        image = np.asarray(self.data.full_frame_image, dtype=float)
        if vmax is None:
            # A few percent of the peak. The interference spot and its fringes cover
            # many pixels at high counts, so a percentile clip stays too bright to
            # reveal the faint lattice / zeroth order (a small fraction of the peak);
            # clip to that fraction instead, so the bright spot saturates.
            vmax = max(0.05 * float(image.max()), float(np.median(image)) + 1.0)
        figure, axs = plt.subplots()
        axs.set_xticks([])
        axs.set_yticks([])
        mappable = axs.imshow(image, cmap=cmap, vmin=0.0, vmax=float(vmax))

        marker_styles = {
            "interference pattern": ("o", "white"),
            "optical lattice": ("s", "white"),
            "zeroth order": ("x", "red"),
        }
        positions = self.data.full_frame_marker_positions or {}
        labeled = False
        for label, position in positions.items():
            if position is None:
                continue
            marker, color = marker_styles.get(label, ("+", "white"))
            axs.plot(
                position[0],
                position[1],
                marker=marker,
                markerfacecolor="none",
                markeredgecolor=color,
                color=color,
                markersize=13,
                linestyle="none",
                label=label,
            )
            labeled = True
        if labeled:
            axs.legend(loc="upper right", fontsize=8)

        axs.set_title("Full-frame snapshot")
        figure.colorbar(mappable, ax=axs, label="camera counts")
        return figure
