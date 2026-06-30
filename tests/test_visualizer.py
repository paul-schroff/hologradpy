"""Tests for the visualization framework (PlotLayout + BaseVisualizer)."""

import dataclasses

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from PIL import Image  # noqa: E402

from hologradpy.visualizer import (  # noqa: E402
    BaseVisualizer,
    GridCell,
    PlotBuilder,
    PlotLayout,
    VisualizationData,
)
from hologradpy.calibration import RasterCalibratorVisualizer  # noqa: E402
from hologradpy.calibration.wavefront.abstract import (  # noqa: E402
    WavefrontCalibrationData,
)
from hologradpy.calibration.wavefront.raster_calibration.visualizer import (  # noqa: E402
    RasterVisualizationData,
)


def _fake_raster_data(number=3, with_lattice=True, with_phases=True):
    camera = np.random.rand(number, 12, 12)
    lattice = np.random.rand(number, 8, 8) if with_lattice else None
    coordinates = np.stack(
        [np.linspace(0, 100, number), np.linspace(0, 80, number)]
    )
    return RasterVisualizationData(
        camera_images=camera,
        fitted_images=camera.copy(),
        measured_phase=np.random.rand(64, 80),
        superpixel_coordinates=coordinates,
        lattice_images=lattice,
        fitted_lattice_images=lattice.copy() if with_lattice else None,
        lattice_shift_x=np.linspace(0, 1e-6, number),
        lattice_shift_y=np.linspace(0, -1e-6, number),
        lattice_shift_x_err=np.full(number, 1e-8),
        lattice_shift_y_err=np.full(number, 1e-8),
        displayed_slm_phases=(
            np.random.rand(number, 64, 80) if with_phases else None
        ),
    )


# --- PlotLayout --------------------------------------------------------


def test_layout_builds_named_axes():
    layout = PlotLayout()
    layout.add_row(
        [GridCell("a", colspan=2, aspect=0.8), GridCell("b", colspan=2, aspect=0.8)]
    )
    layout.add_row([GridCell("c"), GridCell("d"), GridCell("e"), GridCell("f")])
    figure = layout.build()
    assert set(layout.axes) == {"a", "b", "c", "d", "e", "f"}
    assert layout.colorbar_axes == {}
    plt.close(figure)


def test_layout_colorbar_slot_gets_its_own_axes():
    layout = PlotLayout()
    layout.add_row([GridCell("img", colorbar=True)])
    figure = layout.build()
    assert "img" in layout.axes
    assert "img" in layout.colorbar_axes
    # The colorbar axes is distinct from the image axes.
    assert layout.colorbar_axes["img"] is not layout.axes["img"]
    plt.close(figure)


def test_sharex_links_axes():
    layout = PlotLayout()
    layout.add_row([GridCell("top", aspect="auto", height=2.0)])
    layout.add_row([GridCell("bottom", aspect="auto", height=1.0, sharex="top")])
    figure = layout.build()
    assert layout.axes["bottom"] in layout.axes["top"].get_shared_x_axes().get_siblings(
        layout.axes["top"]
    )
    plt.close(figure)


# --- BaseVisualizer.compose ---------------------------------------------------


def test_compose_renders_and_fills_colorbar():
    layout = PlotLayout(column_width=3.0)
    layout.add_row(
        [GridCell("left", colorbar=True), GridCell("right", colorbar=True)]
    )
    figure = BaseVisualizer.compose(
        layout,
        {
            "left": lambda axs: BaseVisualizer.draw_image(
                axs, np.random.rand(8, 10), cmap="magma"
            ),
            "right": lambda axs: BaseVisualizer.draw_image(
                axs, np.random.rand(8, 10), cmap="seismic"
            ),
        },
    )
    # Both colorbar axes were drawn into (a colorbar solid is an image or a
    # collection depending on the matplotlib version).
    for name in ("left", "right"):
        colorbar_axs = layout.colorbar_axes[name]
        assert len(colorbar_axs.images) + len(colorbar_axs.collections) > 0
    plt.close(figure)


# --- PlotBuilder --------------------------------------------------------------


def test_plotbuilder_fills_named_cells_and_colorbar():
    layout = PlotLayout(column_width=3.0)
    layout.add_row([GridCell("a", colorbar=True), GridCell("b")])
    figure = (
        PlotBuilder(layout)
        .draw_image("a", np.random.rand(8, 10), cmap="magma")
        .draw_image("b", np.random.rand(8, 10), cmap="seismic")
        .build()
    )
    assert len(layout.axes["a"].images) == 1
    assert len(layout.axes["b"].images) == 1
    colorbar_axs = layout.colorbar_axes["a"]
    assert len(colorbar_axs.images) + len(colorbar_axs.collections) > 0
    plt.close(figure)


def test_plotbuilder_draw_points_layers_onto_cell():
    layout = PlotLayout()
    layout.add_row([GridCell("img")])
    figure = (
        PlotBuilder(layout)
        .draw_image("img", np.random.rand(8, 10), cmap="magma")
        .draw_points("img", 3, 4)
        .build()
    )
    axs = layout.axes["img"]
    assert len(axs.images) == 1  # the image
    assert len(axs.lines) == 1  # the marker, drawn via plot()
    plt.close(figure)


def test_plotbuilder_line_rows_with_sharex():
    x = np.arange(10)
    layout = PlotLayout(column_width=6.0)
    layout.add_row([GridCell("track", aspect="auto", height=3.0)])
    layout.add_row([GridCell("resid", aspect="auto", height=1.5, sharex="track")])
    figure = (
        PlotBuilder(layout)
        .draw_line("track", [{"x": x, "y": np.sin(x)}], ylabel="y")
        .draw_line("resid", [{"x": x, "y": np.cos(x)}], hlines=(0.0,), xlabel="i")
        .build()
    )
    track = layout.axes["track"]
    assert layout.axes["resid"] in track.get_shared_x_axes().get_siblings(track)
    plt.close(figure)


def test_plotlayout_copy_is_independent():
    layout = PlotLayout(column_width=4.0)
    layout.add_row([GridCell("a"), GridCell("b")])
    clone = layout.copy()
    clone.add_row([GridCell("c")])
    assert clone.column_width == 4.0  # style copied
    assert len(clone._rows) == 2 and len(layout._rows) == 1  # rows independent
    assert clone._rows[0][0] is not layout._rows[0][0]  # cells not shared


# --- VisualizationData wiring -------------------------------------------------


def test_raster_visualization_data_is_visualization_data():
    assert issubclass(RasterVisualizationData, VisualizationData)


def test_wavefront_result_visualization_field_is_optional():
    fields = {f.name: f for f in dataclasses.fields(WavefrontCalibrationData)}
    assert "visualization_data" in fields
    assert fields["visualization_data"].default is None


# --- RasterCalibratorVisualizer ----------------------------------------------


def test_raster_visualizer_default_layout_named_axes():
    visualizer = RasterCalibratorVisualizer(_fake_raster_data())
    assert visualizer.frame_count() == 3
    layout = visualizer.default_layout()
    figure = layout.build()
    assert set(layout.axes) == {
        "displayed_phase",
        "measured_phase",
        "interference",
        "interference_fit",
        "lattice",
        "lattice_fit",
    }
    plt.close(figure)


def test_raster_visualizer_layout_without_lattice():
    visualizer = RasterCalibratorVisualizer(_fake_raster_data(with_lattice=False))
    layout = visualizer.default_layout()
    figure = layout.build()
    assert set(layout.axes) == {
        "displayed_phase",
        "measured_phase",
        "interference",
        "interference_fit",
    }
    plt.close(figure)


def test_raster_visualizer_requires_displayed_phases():
    visualizer = RasterCalibratorVisualizer(_fake_raster_data(with_phases=False))
    with pytest.raises(RuntimeError):
        visualizer.default_layout()


def test_raster_visualizer_save_gif(tmp_path):
    visualizer = RasterCalibratorVisualizer(_fake_raster_data())
    out = tmp_path / "scan.gif"
    visualizer.save_gif(str(out), fps=2, dpi=50)
    with Image.open(out) as image:
        assert image.mode == "P"
        assert len(image.getpalette()) // 3 == 256
        assert getattr(image, "n_frames", 1) == 3


def test_plot_drift_tracking_with_and_without_injected():
    data = _fake_raster_data()
    visualizer = RasterCalibratorVisualizer(data)

    figure_tracked = visualizer.plot_drift_tracking()
    assert len(figure_tracked.axes) == 1  # tracked only
    plt.close(figure_tracked)

    figure_residual = visualizer.plot_drift_tracking(
        np.linspace(0, 1e-6, 3), np.linspace(0, -1e-6, 3)
    )
    assert len(figure_residual.axes) == 2  # tracked + residual
    plt.close(figure_residual)
