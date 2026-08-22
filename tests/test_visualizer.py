"""Tests for the visualization framework (PlotLayout + BaseVisualizer)."""

import dataclasses

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_hex
import pytest
from PIL import Image

from hologradpy.visualizer import (
    BaseVisualizer,
    GridCell,
    PlotBuilder,
    PlotLayout,
    VisualizationData,
    region_bounding_box,
)
from hologradpy.calibration import RasterCalibratorVisualizer
from hologradpy.calibration.wavefront.abstract import (
    WavefrontCalibrationData,
)
from hologradpy.calibration.wavefront.raster_calibration.visualizer import (
    RasterVisualizationData,
)
from hologradpy.calibration.wavefront.speckle_calibration.visualizer import (
    SpeckleVisualizationData,
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


def _cell_size(layout, name):
    """One cell's size in inches, which is what the eye actually compares.

    The cells are placed by a Divider locator, which only resolves at draw time, so
    the position before a draw is still the full figure rect.
    """
    layout.figure.canvas.draw()
    width, height = layout.figure.get_size_inches()
    box = layout.axes[name].get_position()
    return box.width * width, box.height * height


def test_cells_of_different_shapes_come_out_the_same_height():
    """A square image beside a wide one should line up along the row, with the wide one
    given the extra width it needs. Sizing both the same width instead leaves the wide
    one letterboxed inside a cell taller than itself.
    """
    layout = PlotLayout()
    layout.add_row([GridCell("square", aspect=1.0), GridCell("wide", aspect=0.5)])
    layout.build()

    square_width, square_height = _cell_size(layout, "square")
    wide_width, wide_height = _cell_size(layout, "wide")

    assert wide_height == pytest.approx(square_height, rel=1e-6)
    # Aspect ratios untouched: each cell is still the shape it asked for.
    assert square_height / square_width == pytest.approx(1.0, rel=1e-6)
    assert wide_height / wide_width == pytest.approx(0.5, rel=1e-6)
    assert wide_width == pytest.approx(2 * square_width, rel=1e-6)
    plt.close(layout.figure)


def test_cells_of_the_same_shape_stay_equally_wide():
    """The common case must not move: matching heights across a uniform row is just the
    equal-width layout it always had.
    """
    layout = PlotLayout()
    layout.add_row([GridCell("a", aspect=0.8), GridCell("b", aspect=0.8)])
    layout.build()

    assert _cell_size(layout, "a")[0] == pytest.approx(
        _cell_size(layout, "b")[0], rel=1e-6
    )
    plt.close(layout.figure)


def test_a_colorbar_gets_room_for_its_tick_labels():
    """The labels hang off the right of the bar, into whatever comes next. Without an
    allowance they run into the neighbouring panel, which is what the default col_gap
    of 0.3 inches left them doing.
    """
    layout = PlotLayout()
    layout.add_row([GridCell("left", colorbar=True), GridCell("right")])
    layout.build()
    layout.figure.canvas.draw()  # the Divider locator only resolves at draw time

    left_box = layout.axes["left"].get_position()
    bar_box = layout.colorbar_axes["left"].get_position()
    right_box = layout.axes["right"].get_position()
    figure_width = layout.figure.get_size_inches()[0]
    gap = (right_box.x0 - bar_box.x1) * figure_width

    assert bar_box.x0 > left_box.x1  # the bar sits outside its panel
    assert gap == pytest.approx(
        layout.col_gap + layout.colorbar_label_width, rel=1e-6
    )
    plt.close(layout.figure)


def test_a_line_row_keeps_equal_columns():
    """A line plot has no natural height, so there is nothing to match it against and
    the row falls back to equal columns.
    """
    layout = PlotLayout()
    layout.add_row([GridCell("image", aspect=0.5), GridCell("curve", aspect="auto")])
    layout.build()

    assert _cell_size(layout, "image")[0] == pytest.approx(
        _cell_size(layout, "curve")[0], rel=1e-6
    )
    plt.close(layout.figure)


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


# --- speckle comparison panels --------------------------------------------------


def _fake_speckle_data(
    *,
    with_truth: bool = True,
    loss_component_history: dict[str, list[float]] | None = None,
) -> SpeckleVisualizationData:
    """A payload shaped like one a speckle calibration records.

    The recovered field is the injected one plus a small ripple, so the difference
    panels have something to show and its RMS is a number the test can predict.
    """
    rows, columns = np.mgrid[0:16, 0:16] / 15.0
    amplitude = np.exp(-((rows - 0.5) ** 2 + (columns - 0.5) ** 2) / 0.1)
    injected_phase = 2.0 * rows - 1.5 * columns
    injected = amplitude * np.exp(1j * injected_phase)

    mask = amplitude > 0.1 * amplitude.max()
    return SpeckleVisualizationData(
        camera_image=np.abs(injected) ** 2,
        roi_mask=mask,
        slm_pattern=np.angle(injected),
        measured_roi=np.abs(injected) ** 2,
        predicted_roi=np.abs(injected) ** 2,
        recovered_amplitude=amplitude,
        recovered_phase=injected_phase + 0.05 * np.sin(8 * rows),
        loss_history=[1.0, 0.5],
        loss_component_history=loss_component_history or {},
        injected_field=injected if with_truth else None,
        beam_mask=mask if with_truth else None,
    )


# --- convergence panel ------------------------------------------------------------


THREE_TERMS = {
    "intensity mse": [0.7, 0.3],
    "phase smoothness": [0.2, 0.15],
    "amplitude smoothness": [0.1, 0.05],
}


def _loss_axes(visualizer):
    """The convergence panel, found by its title rather than its position."""
    figure = visualizer.render()
    axs = next(a for a in figure.axes if a.get_title() == "convergence")
    return figure, axs


def test_the_convergence_panel_draws_the_total_and_each_term():
    """The point of recording the terms: a flat total can be the mismatch still falling
    while a prior climbs to meet it, and only the separate curves show that.
    """
    visualizer = _fake_speckle_data(loss_component_history=THREE_TERMS).visualizer()

    figure, axs = _loss_axes(visualizer)

    labels = [line.get_label() for line in axs.get_lines()]
    assert labels == [
        "total",
        "intensity mse",
        "phase smoothness",
        "amplitude smoothness",
    ]
    assert axs.get_legend() is not None
    plt.close(figure)


def test_the_total_curve_follows_the_theme():
    """It used to be hard coded black, which is invisible on a dark background. A PSF
    fit draws that one curve and nothing else, so the whole panel came out empty.
    """
    with plt.style.context("dark_background"):
        visualizer = _fake_speckle_data(
            loss_component_history=THREE_TERMS
        ).visualizer()
        figure, axs = _loss_axes(visualizer)
        total = axs.get_lines()[0]

        assert to_hex(total.get_color()) == to_hex(plt.rcParams["text.color"])
        assert to_hex(total.get_color()) != to_hex(plt.rcParams["figure.facecolor"])

    plt.close(figure)


def test_the_convergence_panel_skips_a_lone_term():
    """A PSF calibration fits against the mismatch alone, so its one component curve
    would sit exactly on the total and say nothing.
    """
    visualizer = _fake_speckle_data(
        loss_component_history={"intensity mse": [1.0, 0.5]}
    ).visualizer()

    figure, axs = _loss_axes(visualizer)

    assert len(axs.get_lines()) == 1
    assert axs.get_legend() is None
    plt.close(figure)


def test_a_record_without_the_terms_still_draws_its_total():
    """Payloads pickled before the terms were recorded have to keep rendering, which is
    what the field's default is for.
    """
    visualizer = _fake_speckle_data().visualizer()

    figure, axs = _loss_axes(visualizer)

    assert len(axs.get_lines()) == 1
    assert axs.get_yscale() == "log"
    plt.close(figure)


def test_a_term_starting_at_zero_leaves_the_log_axis_alone():
    """A smoothness prior on a field that starts flat is exactly zero for the first
    epoch, every single run. Letting that pick the scale would flatten a loss that falls
    two decades, so the total governs it and the zero simply has no point.
    """
    terms = dict(THREE_TERMS)
    terms["amplitude smoothness"] = [0.0, 0.05]
    visualizer = _fake_speckle_data(loss_component_history=terms).visualizer()

    figure, axs = _loss_axes(visualizer)

    assert axs.get_yscale() == "log"
    plt.close(figure)


def test_a_non_positive_total_drops_the_panel_to_a_linear_axis():
    """A cost with a negative term, such as an efficiency reward, has no log axis to
    draw on and must not lose points to one.
    """
    data = _fake_speckle_data(loss_component_history=THREE_TERMS)
    data.loss_history = [1.0, -0.5]
    visualizer = data.visualizer()

    figure, axs = _loss_axes(visualizer)

    assert axs.get_yscale() == "linear"
    plt.close(figure)


def test_speckle_comparison_draws_six_cells():
    """Injected, recovered and difference, for phase and amplitude."""
    visualizer = _fake_speckle_data().visualizer()

    figure = visualizer.render_comparison()

    titles = {axs.get_title() for axs in figure.axes if axs.get_title()}
    assert any("injected phase" in title for title in titles)
    assert any("recovered phase" in title for title in titles)
    assert any("phase difference" in title for title in titles)
    assert any("injected amplitude" in title for title in titles)
    assert any("recovered amplitude" in title for title in titles)
    assert any("amplitude difference" in title for title in titles)
    plt.close(figure)


def test_speckle_comparison_difference_is_symmetric_about_zero():
    """The neutral color has to mean agreement, which only holds if the limits are
    symmetric. Otherwise a residual of zero reads as some arbitrary color.
    """
    visualizer = _fake_speckle_data().visualizer()

    figure = visualizer.render_comparison()

    difference_images = [
        image
        for axs in figure.axes
        if "difference" in axs.get_title()
        for image in axs.get_images()
    ]
    assert difference_images
    for image in difference_images:
        low, high = image.get_clim()
        assert low == pytest.approx(-high)
    plt.close(figure)


def test_speckle_comparison_without_truth_says_why():
    """A calibration from a real bench has nothing to compare against. Asking has to
    name what is missing rather than drawing empty axes.
    """
    data = _fake_speckle_data(with_truth=False)

    # The diagnostics figure is unaffected by the absence.
    figure = data.visualizer().render()
    plt.close(figure)

    with pytest.raises(RuntimeError, match="static_slm_field"):
        data.visualizer().render_comparison()


def test_speckle_payload_predating_the_comparison_still_renders():
    """The two fields were added with defaults so that records pickled before they
    existed keep working. Constructing without them is that case.
    """
    data = SpeckleVisualizationData(
        camera_image=np.zeros((8, 8)),
        roi_mask=np.ones((8, 8), dtype=bool),
        measured_roi=np.zeros((4, 4)),
        predicted_roi=np.zeros((4, 4)),
        recovered_amplitude=np.ones((8, 8)),
        recovered_phase=np.zeros((8, 8)),
    )

    assert data.injected_field is None
    assert data.beam_mask is None
    assert data.slm_pattern is None

    figure = data.visualizer().render()

    # The pattern cell is simply absent rather than the whole figure failing.
    titles = [axs.get_title() for axs in figure.axes if axs.get_title()]
    assert "camera + ROI" in titles
    assert "SLM phase pattern" not in titles
    plt.close(figure)


def test_the_dataset_figure_needs_a_pattern_to_be_worth_drawing():
    """Without one it would be a single cell the full diagnostics already carries, so
    it says so rather than drawing a lone camera frame.
    """
    data = SpeckleVisualizationData(
        camera_image=np.zeros((8, 8)), roi_mask=np.ones((8, 8), dtype=bool)
    )

    with pytest.raises(RuntimeError, match="no slm_pattern"):
        data.visualizer().render_dataset()


def test_the_dataset_figure_draws_before_anything_is_fitted():
    """The point of it: a capture can be checked before a fit is spent on it, so it
    must not need any of the fitted arrays.
    """
    data = SpeckleVisualizationData(
        camera_image=np.random.default_rng(0).uniform(size=(8, 8)),
        roi_mask=np.ones((8, 8), dtype=bool),
        slm_pattern=np.random.default_rng(1).uniform(size=(6, 6)),
    )

    figure = data.visualizer().render_dataset()

    titles = [axs.get_title() for axs in figure.axes if axs.get_title()]
    assert titles == ["SLM pattern [levels]", "camera + ROI"]
    plt.close(figure)


def test_the_bounding_box_is_tight_around_the_region():
    """The crop shows the region and nothing else, so no padding by default."""
    region = np.zeros((100, 80), dtype=bool)
    region[20:61, 10:51] = True

    rows, columns = region_bounding_box(region)

    assert (rows.start, rows.stop) == (20, 61)
    assert (columns.start, columns.stop) == (10, 51)
    # Every edge of the crop lands on the region rather than outside it.
    cropped = region[rows, columns]
    assert cropped.all()


def test_the_bounding_box_pads_when_asked():
    """The margin is still available, just not the default."""
    region = np.zeros((100, 80), dtype=bool)
    region[20:61, 10:51] = True

    rows, columns = region_bounding_box(region, margin_fraction=0.1)

    assert (rows.start, rows.stop) == (16, 65)
    assert (columns.start, columns.stop) == (6, 55)


def test_the_bounding_box_stays_inside_the_frame():
    """A region against the edge cannot be padded off the end of the array."""
    region = np.zeros((10, 10), dtype=bool)
    region[0:3, 7:10] = True

    rows, columns = region_bounding_box(region, margin_fraction=0.5)

    assert rows.start == 0
    assert columns.stop == 10
    assert region[rows, columns].shape[0] <= 10


def test_an_empty_region_falls_back_to_the_whole_frame():
    """Cropping to nothing would leave an empty panel, so nothing is cropped."""
    rows, columns = region_bounding_box(np.zeros((5, 5), dtype=bool))

    assert rows == slice(None)
    assert columns == slice(None)


def test_image_grid_shapes_each_cell_like_its_own_image():
    """The reason the helper exists.

    An image is drawn with square pixels, so one placed in a cell of a different shape
    is shrunk to fit while the colorbar stays pinned to the cell, and the two come
    apart. Deriving the aspect from the array is what makes that impossible.
    """
    from hologradpy.visualizer import image_grid

    images = [
        np.zeros((400, 640)),
        np.zeros((256, 256)),
        np.zeros((500, 375)),
    ]

    builder = image_grid(images)
    cells = [cell for row in builder.layout._rows for cell in row]

    assert [cell.aspect for cell in cells] == [
        image.shape[0] / image.shape[1] for image in images
    ]


def test_image_grid_takes_rows_of_differing_length():
    """Rows are given as nested sequences, and need not match in length."""
    from hologradpy.visualizer import image_grid

    builder = image_grid([[np.zeros((8, 8)), np.zeros((8, 8))], [np.zeros((8, 8))]])

    assert [len(row) for row in builder.layout._rows] == [2, 1]


def test_image_grid_puts_every_panel_on_one_scale_when_asked():
    """A shared scale is what makes two panels comparable by eye."""
    from hologradpy.visualizer import image_grid

    dim = np.full((4, 4), 1.0)
    bright = np.full((4, 4), 5.0)

    figure = image_grid([dim, bright], shared_scale=True).build()
    limits = [image.get_clim() for image in
              [axs.images[0] for axs in figure.axes if axs.images]]

    assert limits[0] == limits[1] == (1.0, 5.0)


def test_image_grid_centres_a_symmetric_scale_on_zero():
    """So the neutral colour of a diverging map means agreement, not the midpoint."""
    from hologradpy.visualizer import image_grid

    difference = np.array([[-1.0, 3.0], [0.0, 2.0]])

    figure = image_grid([difference], symmetric=True).build()
    image = next(axs.images[0] for axs in figure.axes if axs.images)

    assert image.get_clim() == (-3.0, 3.0)


def test_a_cell_shaped_unlike_its_image_warns():
    """The guard for layouts still built by hand.

    Nothing else reports this: the figure renders, and the misalignment is only
    visible by looking at it.
    """
    import warnings

    layout = PlotLayout(column_width=3.0)
    layout.add_row([GridCell("panel", aspect=1.0, colorbar=True)])

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        PlotBuilder(layout).draw_image("panel", np.zeros((400, 640))).build()

    assert any("no longer line up" in str(entry.message) for entry in caught)


def test_a_cell_matching_its_image_does_not_warn():
    """The warning must stay quiet for correct layouts, or it becomes noise."""
    import warnings

    layout = PlotLayout(column_width=3.0)
    layout.add_row([GridCell("panel", aspect=400 / 640, colorbar=True)])

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        PlotBuilder(layout).draw_image("panel", np.zeros((400, 640))).build()

    assert not any("no longer line up" in str(entry.message) for entry in caught)
