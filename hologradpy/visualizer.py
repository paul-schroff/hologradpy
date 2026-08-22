"""A small, modular visualization framework.

Two decoupled concerns:

- :class:`PlotLayout` owns the figure and a set of *named, neatly aligned axes cells*
  and knows nothing about what is drawn into them. GridCells declare only their geometry
  (column span, aspect ratio, whether they carry a colorbar) and the manager
  sizes/aligns them with ``mpl_toolkits.axes_grid1`` so images keep their aspect ratio,
  rows line up, and colorbars never squash the image.

- :class:`BaseVisualizer` provides reusable, content-only drawing helpers
  (``draw_image`` / ``draw_line`` / ``draw_points``) plus static figure composition. A
  concrete visualizer supplies a :meth:`default_layout` and a :meth:`panels` mapping of
  cell name to a ``panel(axs)`` callable, with no frame, since most plots are static.
  :class:`AnimatedVisualizer` adds the animation / GIF machinery for the few plots that
  move: an animation is just composing the static panels once per frame.

Because a panel only paints into the axes it is handed, panels from different
visualizers can be mixed and matched into one :class:`PlotLayout` and rendered together.
"""
# TODO: Sanity check and tidy up

from __future__ import annotations

from dataclasses import dataclass, replace
import warnings
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Callable

import numpy as np
from numpy.typing import ArrayLike

from .serialization import record_type

if TYPE_CHECKING:
    from matplotlib.animation import FuncAnimation
    from matplotlib.axes import Axes
    from matplotlib.cm import ScalarMappable
    from matplotlib.figure import Figure


# One colormap per kind of quantity, package wide.
INTENSITY_CMAP = "viridis"
PHASE_CMAP = "twilight"
DIFFERENCE_CMAP = "seismic"


def region_bounding_box(
    region: ArrayLike, margin_fraction: float = 0.0
) -> tuple[slice, slice]:
    """The box a boolean mask occupies, as row and column slices.

    A signal region is usually a small part of the plane, so a full-frame view spends
    most of its area on darkness. The box is tight by default, so what is drawn is the
    region and nothing else.

    Args:
        region: The mask to bound. A mask that is false everywhere gives back slices
            covering the whole array.
        margin_fraction: Extra space on each side, as a fraction of the box it bounds.
            Zero by default, for a box that ends where the region does.

    Returns:
        tuple[slice, slice]: Row and column slices, ready to index an image with.
    """
    region = np.asarray(region, dtype=bool)
    rows = np.flatnonzero(region.any(axis=1))
    columns = np.flatnonzero(region.any(axis=0))
    if rows.size == 0 or columns.size == 0:
        return slice(None), slice(None)

    def span(indices: np.ndarray, size: int) -> slice:
        first, last = int(indices[0]), int(indices[-1])
        margin = round(margin_fraction * (last - first + 1))
        return slice(max(first - margin, 0), min(last + margin + 1, size))

    return span(rows, region.shape[0]), span(columns, region.shape[1])


def foreground_color() -> str:
    # TODO: Move imports to top of file
    from matplotlib import rcParams

    return rcParams["text.color"]


if TYPE_CHECKING:
    Panel = Callable[[Axes], ScalarMappable | None]
else:
    Panel = Callable


@record_type("visualization_data")
@dataclass
class VisualizationData:
    """Base for a result's visualization payload."""


@dataclass
class GridCell:
    """Geometry of one named axes cell in a :class:`PlotLayout`.

    Args:
        name: Key used to look the axes up in ``layout.axes[name]``.
        colspan: Number of grid columns the cell spans.
        aspect: Cell height-to-width ratio. A float is the ratio
            ``height / width``, ``"equal"`` is a square cell (ratio 1.0), and
            ``"auto"`` leaves the height to ``height`` or the row default and
            suits line plots.
        colorbar: If True, a matched-height colorbar axes is appended on the
            right and exposed as ``layout.colorbar_axes[name]``.
        height: Explicit cell height [inches], overriding ``aspect`` for the row
            height. Mainly for ``aspect="auto"`` line rows.
        sharex: Name of another cell whose x-axis this cell should share.
    """

    name: str
    colspan: int = 1
    aspect: float | str = "equal"
    colorbar: bool = False
    height: float | None = None
    sharex: str | None = None


class PlotLayout:
    """Build a figure of named, neatly aligned axes cells.

    Add rows of :class:`GridCell` with :meth:`add_row`, then :meth:`build`. Columns are
    equal width, and each row's height follows from its cells' aspect ratios so images
    stay undistorted and rows line up. Built on ``mpl_toolkits.axes_grid1.Divider``
    (absolute inch sizing), with every colorbar placed in its own divider cell so it
    matches the height of the cell it belongs to.
    """

    def __init__(
        self,
        column_width: float = 3.2,
        col_gap: float = 0.3,
        row_gap: float = 0.5,
        margins: tuple[float, float, float, float] = (0.12, 0.12, 0.6, 0.18),
        colorbar_width: float = 0.15,
        colorbar_pad: float = 0.08,
        colorbar_label_width: float = 0.45,
    ) -> None:
        # margins: (left, right, top, bottom) in inches.
        self.column_width = column_width
        self.col_gap = col_gap
        self.row_gap = row_gap
        self.margins = margins
        self.colorbar_width = colorbar_width
        self.colorbar_pad = colorbar_pad
        self.colorbar_label_width = colorbar_label_width

        self._rows: list[list[GridCell]] = []
        self.figure: Figure | None = None
        self.axes: dict[str, Axes] = {}
        self.colorbar_axes: dict[str, Axes] = {}

    def add_row(self, cells: list[GridCell]) -> PlotLayout:
        """Append a row of cells (left to right). Returns self for chaining."""
        self._rows.append(list(cells))
        return self

    def cell(self, name: str) -> GridCell | None:
        """The cell of that name, or None when the layout has no such cell."""
        return next(
            (cell for row in self._rows for cell in row if cell.name == name), None
        )

    def copy(self) -> PlotLayout:
        """Return an independent clone (same style + cells) for reuse.

        A configured layout can seed many figures, and ``copy()`` lets you tweak one
        (e.g. add a row) without disturbing the original.
        """
        clone = PlotLayout(
            column_width=self.column_width,
            col_gap=self.col_gap,
            row_gap=self.row_gap,
            margins=self.margins,
            colorbar_width=self.colorbar_width,
            colorbar_pad=self.colorbar_pad,
            colorbar_label_width=self.colorbar_label_width,
        )
        clone._rows = [[replace(cell) for cell in row] for row in self._rows]
        return clone

    def _cell_height(self, cell: GridCell, column_width: float) -> float | None:
        """Natural cell height [inches], or None when sizing defers to the row."""
        if cell.height is not None:
            return cell.height
        if cell.aspect == "auto":
            return None
        width = self._cell_width(cell, column_width)
        return width * self._ratio(cell)

    def _cell_width(self, cell: GridCell, column_width: float) -> float:
        return cell.colspan * column_width + (cell.colspan - 1) * self.col_gap

    def _row_columns(self, row: list[GridCell]) -> int:
        return sum(cell.colspan for cell in row)

    def _row_gutter(self, row: list[GridCell]) -> float:
        """Width taken by colorbars and tick labels."""
        gutter = self.colorbar_pad + self.colorbar_width
        return sum(gutter for cell in row if cell.colorbar) + sum(
            self.colorbar_label_width for cell in row[:-1] if cell.colorbar
        )

    @staticmethod
    def _ratio(cell: GridCell) -> float:
        return 1.0 if cell.aspect == "equal" else float(cell.aspect)

    def _matches_heights(self, row: list[GridCell]) -> bool:
        """True if a row can be scaled to a shared cell height."""
        return bool(row) and all(
            cell.aspect != "auto" and cell.height is None for cell in row
        )

    def _row_plan(
        self, row: list[GridCell], figure_width: float, left: float, right: float
    ) -> tuple[list[float], float]:
        """Cell widths for one row, and their heights."""
        if self._matches_heights(row):
            available = (
                figure_width
                - left
                - right
                - (len(row) - 1) * self.col_gap
                - self._row_gutter(row)
            )
            weights = [cell.colspan / self._ratio(cell) for cell in row]
            widths = [available * weight / sum(weights) for weight in weights]
            return widths, max(
                width * self._ratio(cell) for width, cell in zip(widths, row)
            )

        columns = self._row_columns(row)
        column_width = (
            self.column_width
            if columns == 0
            else (
                figure_width
                - left
                - right
                - max(columns - 1, 0) * self.col_gap
                - self._row_gutter(row)
            )
            / columns
        )
        widths = [self._cell_width(cell, column_width) for cell in row]
        heights = [
            height
            for height in (self._cell_height(cell, column_width) for cell in row)
            if height is not None
        ]
        return widths, max(heights) if heights else column_width

    def build(self, suptitle: str | None = None) -> Figure:
        """Creates the figure and all cell axes.  Returns the figure."""
        # TODO: Move imports to top of the file
        import matplotlib.pyplot as plt
        from mpl_toolkits.axes_grid1 import Divider, Size

        left, right, top, bottom = self.margins
        # A colorbar at the end of a row puts its ticks in the right margin, so widen
        # it or they are clipped.
        if any(row and row[-1].colorbar for row in self._rows):
            right += self.colorbar_label_width

        def natural_width(row: list[GridCell]) -> float:
            columns = self._row_columns(row)
            return (
                left
                + right
                + columns * self.column_width
                + max(columns - 1, 0) * self.col_gap
                + self._row_gutter(row)
            )

        figure_width = max(natural_width(row) for row in self._rows)

        row_plans = [
            self._row_plan(row, figure_width, left, right) for row in self._rows
        ]
        row_heights = [height for _, height in row_plans]

        figure_height = (
            top
            + bottom
            + sum(row_heights)
            + (len(self._rows) - 1) * self.row_gap
        )

        figure = plt.figure(figsize=(figure_width, figure_height))

        # Vertical sizes are bottom-to-top: bottom margin, rows in reverse order
        # interleaved with gaps, then top margin. Shared by every row's divider, which
        # is what keeps the rows aligned even though their columns differ.
        vertical: list = [Size.Fixed(bottom)]
        for index, height in enumerate(reversed(row_heights)):
            vertical.append(Size.Fixed(height))
            if index < len(row_heights) - 1:
                vertical.append(Size.Fixed(self.row_gap))
        vertical.append(Size.Fixed(top))

        number_of_rows = len(self._rows)
        for row_index, (row, (widths, _)) in enumerate(zip(self._rows, row_plans)):
            # One divider per row, over that row's own cells. Colorbars sit in
            # their own divider cell, so the panels and their bars are placed by
            # the same absolute sizing.
            horizontal: list = [Size.Fixed(left)]
            slots: list[tuple[int, int | None]] = []
            for position, (cell, width) in enumerate(zip(row, widths)):
                horizontal.append(Size.Fixed(width))
                content = len(horizontal) - 1
                colorbar = None
                if cell.colorbar:
                    horizontal.append(Size.Fixed(self.colorbar_pad))
                    horizontal.append(Size.Fixed(self.colorbar_width))
                    colorbar = len(horizontal) - 1
                if position < len(row) - 1:
                    # The bar's tick labels overhang into this gap, so widen it when
                    # there is a bar to its left.
                    horizontal.append(
                        Size.Fixed(
                            self.col_gap
                            + (self.colorbar_label_width if cell.colorbar else 0.0)
                        )
                    )
                slots.append((content, colorbar))
            horizontal.append(Size.Fixed(right))

            divider = Divider(figure, (0, 0, 1, 1), horizontal, vertical, aspect=False)
            ny = 1 + 2 * (number_of_rows - 1 - row_index)
            for cell, (content, colorbar) in zip(row, slots):
                self.axes[cell.name] = figure.add_axes(
                    divider.get_position(),
                    axes_locator=divider.new_locator(nx=content, ny=ny),
                )
                if colorbar is not None:
                    self.colorbar_axes[cell.name] = figure.add_axes(
                        divider.get_position(),
                        axes_locator=divider.new_locator(nx=colorbar, ny=ny),
                    )

        # Share x-axes once every axes exists.
        for row in self._rows:
            for cell in row:
                if cell.sharex is not None:
                    self.axes[cell.name].sharex(self.axes[cell.sharex])

        if suptitle is not None:
            figure.suptitle(suptitle)

        self.figure = figure
        return figure


class BaseVisualizer:
    """Content drawing helpers plus static figure composition.

    Most visualizers are static: a subclass supplies :meth:`default_layout` and
    :meth:`panels` (a mapping of cell name to a ``panel(axs) -> mappable | None``
    callable) and renders with :meth:`render`. The drawing helpers are static so they
    can also be used standalone (e.g. to compose an ad-hoc figure with :meth:`compose`).
    For animation, subclass :class:`AnimatedVisualizer`.
    """

    @staticmethod
    def draw_image(
        axs: Axes,
        data: ArrayLike,
        *,
        cmap: str = "viridis",
        vmin: float | None = None,
        vmax: float | None = None,
        title: str | None = None,
        interpolation: str | None = None,
    ) -> ScalarMappable:
        """Show ``data`` as an image and return the mappable (for a colorbar)."""
        axs.set_xticks([])
        axs.set_yticks([])
        mappable = axs.imshow(
            np.asarray(data),
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            interpolation=interpolation,
        )
        if title is not None:
            axs.set_title(title)
        return mappable

    @staticmethod
    def draw_points(
        axs: Axes,
        x: ArrayLike,
        y: ArrayLike,
        *,
        marker: str = "o",
        color: str = "red",
        size: float = 8.0,
        edgecolor: str = "white",
        legend: bool = False,
        **kwargs: Any,
    ) -> None:
        """Overlay scatter points (markers only) on top of an image/axes.

        Pass ``label=...`` (forwarded to matplotlib) together with
        ``legend=True`` on the last points layer of a cell to draw a legend.
        """
        axs.plot(
            x,
            y,
            marker=marker,
            color=color,
            markersize=size,
            markeredgecolor=edgecolor,
            linestyle="none",
            **kwargs,
        )
        if legend:
            axs.legend()

    @staticmethod
    def draw_line(
        axs: Axes,
        curves: list[dict],
        *,
        hlines: tuple[float, ...] = (),
        xlabel: str | None = None,
        ylabel: str | None = None,
        title: str | None = None,
        yscale: str = "linear",
        legend: bool = False,
    ) -> None:
        """Draw line / errorbar curves.

        Each curve is a dict with keys ``x``, ``y`` and optional ``yerr``, ``style``,
        ``label``, ``color``.
        """
        for curve in curves:
            if curve.get("yerr") is not None:
                axs.errorbar(
                    curve["x"],
                    curve["y"],
                    yerr=curve["yerr"],
                    fmt=curve.get("style", "-"),
                    color=curve.get("color"),
                    label=curve.get("label"),
                    capsize=2,
                )
            else:
                axs.plot(
                    curve["x"],
                    curve["y"],
                    curve.get("style", "-"),
                    color=curve.get("color"),
                    label=curve.get("label"),
                )
        for value in hlines:
            axs.axhline(value, color="gray", linewidth=0.8)
        if xlabel is not None:
            axs.set_xlabel(xlabel)
        if ylabel is not None:
            axs.set_ylabel(ylabel)
        if title is not None:
            axs.set_title(title)
        axs.set_yscale(yscale)
        if legend:
            axs.legend()

    @staticmethod
    def draw_quiver(
        axs: Axes,
        x: ArrayLike,
        y: ArrayLike,
        u: ArrayLike,
        v: ArrayLike,
        *,
        scale: float | None = None,
        color: str = "C0",
        xlabel: str | None = None,
        ylabel: str | None = None,
        title: str | None = None,
        invert_y: bool = False,
    ) -> None:
        """Draw a vector field ``(u, v)`` anchored at ``(x, y)`` in data units.

        ``scale`` follows matplotlib's ``scale_units="xy"`` convention (drawn
        length = vector length / scale, so ``scale < 1`` magnifies). ``None``
        lets matplotlib autoscale. ``invert_y`` matches image coordinates
        (y grows downward).
        """
        axs.quiver(
            x, y, u, v,
            angles="xy", scale_units="xy", scale=scale, color=color, width=0.004,
        )
        axs.set_aspect("equal", adjustable="datalim")
        if invert_y:
            axs.invert_yaxis()
        if xlabel is not None:
            axs.set_xlabel(xlabel)
        if ylabel is not None:
            axs.set_ylabel(ylabel)
        if title is not None:
            axs.set_title(title)

    def default_layout(self) -> PlotLayout:
        raise NotImplementedError

    def panels(self) -> dict[str, Panel]:
        raise NotImplementedError

    @classmethod
    def compose(
        cls,
        layout: PlotLayout,
        panels: dict[str, Panel],
        *,
        suptitle: str | None = None,
    ) -> Figure:
        """Render a static figure: build ``layout`` and draw each panel once.

        Colorbars are added for cells that declared one and whose panel returns a
        mappable. Works without a subclass, so it is the entry point for mixing panels
        from different visualizers into one figure.
        """
        figure = layout.build(suptitle=suptitle)
        for name, panel in panels.items():
            mappable = panel(layout.axes[name])
            colorbar_axs = layout.colorbar_axes.get(name)
            if mappable is not None and colorbar_axs is not None:
                _warn_on_aspect_mismatch(layout, name, mappable)
                figure.colorbar(mappable, cax=colorbar_axs)
        return figure

    def render(
        self,
        *,
        layout: PlotLayout | None = None,
        panels: dict[str, Panel] | None = None,
        suptitle: str | None = None,
    ) -> Figure:
        """Render this visualizer's default (or a given) static figure."""
        layout = layout if layout is not None else self.default_layout()
        panels = panels if panels is not None else self.panels()
        return self.compose(layout, panels, suptitle=suptitle)


class AnimatedVisualizer(BaseVisualizer):
    """A :class:`BaseVisualizer` whose figure is animated, one static frame per
    index.

    A subclass supplies :meth:`frame_count` and :meth:`panels_for_frame` (the static
    :data:`Panel` mapping for a given frame, with the frame bound by closure) and
    optionally overrides :meth:`frame_suptitle`. An animation is just composing the
    static panels once per frame, so the same draw helpers and layout serve both stills
    and motion. :meth:`panels` defaults to frame 0, so an animated visualizer still
    renders as a static figure.
    """

    def frame_count(self) -> int:
        raise NotImplementedError

    def panels_for_frame(self, frame: int) -> dict[str, Panel]:
        raise NotImplementedError

    def frame_suptitle(self, frame: int) -> str | None:
        return None

    def panels(self) -> dict[str, Panel]:
        # Static still: the first frame.
        return self.panels_for_frame(0)


    def _frame_indices(self, max_frames: int | None) -> np.ndarray:
        total = self.frame_count()
        if max_frames is None or max_frames >= total:
            return np.arange(total)
        return np.linspace(0, total - 1, max_frames).astype(int)

    def _draw_frame(self, layout: PlotLayout, frame: int) -> None:
        for axs in layout.axes.values():
            axs.clear()
        for name, panel in self.panels_for_frame(frame).items():
            panel(layout.axes[name])
        suptitle = self.frame_suptitle(frame)
        if suptitle is not None and layout.figure is not None:
            layout.figure.suptitle(suptitle)

    def animate(
        self,
        *,
        layout: PlotLayout | None = None,
        fps: int = 2,
        max_frames: int | None = None,
    ) -> FuncAnimation:
        """Build a :class:`~matplotlib.animation.FuncAnimation` of the frames."""
        from matplotlib.animation import FuncAnimation

        layout = layout if layout is not None else self.default_layout()
        layout.build()
        frame_indices = self._frame_indices(max_frames)

        def update(frame: int) -> None:
            self._draw_frame(layout, int(frame))

        return FuncAnimation(
            layout.figure, update, frames=frame_indices, interval=1000 / fps
        )

    def save_gif(
        self,
        path: str,
        *,
        layout: PlotLayout | None = None,
        fps: int = 2,
        dpi: int = 100,
        max_frames: int | None = None,
    ) -> str:
        """Render the frames to an animated GIF at ``path``.

        Each frame is quantized with a 256-color adaptive palette and Floyd-Steinberg
        dithering. matplotlib's default GIF writer maps every frame onto the 216-color
        web-safe palette, which badly posterizes smooth gradients, and the adaptive,
        dithered palette keeps them smooth. The dithering happens in a second pass,
        because only the ``palette=`` form of ``quantize`` dithers.
        """
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from PIL import Image

        layout = layout if layout is not None else self.default_layout()
        figure = layout.build()
        figure.set_dpi(dpi)
        canvas = FigureCanvasAgg(figure)
        frame_indices = self._frame_indices(max_frames)

        frames = []
        for frame in frame_indices:
            self._draw_frame(layout, int(frame))
            canvas.draw()
            rgb = Image.fromarray(
                np.asarray(canvas.buffer_rgba()), "RGBA"
            ).convert("RGB")
            palette_image = rgb.quantize(
                colors=256, method=Image.Quantize.MEDIANCUT
            )
            frames.append(
                rgb.quantize(
                    palette=palette_image, dither=Image.Dither.FLOYDSTEINBERG
                )
            )
        plt.close(figure)

        frames[0].save(
            path,
            save_all=True,
            append_images=frames[1:],
            duration=int(round(1000 / fps)),
            loop=0,
            disposal=2,
        )
        return path


class PlotBuilder:
    """Builder that fills a :class:`PlotLayout`'s named cells with content.

    The layout defines the grid of named :class:`GridCell` cells (geometry only), and
    the builder draws into them *by name*. Every ``draw_*`` method takes the target cell
    name first and returns ``self`` so calls chain, and repeated calls on the same
    cell layer (e.g. an image then markers). :meth:`build` folds each cell's ops
    into one panel and composes the figure. It is pure sugar over
    :meth:`BaseVisualizer.compose`, so no dict or ``lambda axs:`` is needed.

    >>> layout = PlotLayout(column_width=4.0)
    >>> layout.add_row([
    ...     GridCell("truth", aspect=ratio, colorbar=True),
    ...     GridCell("detected", aspect=ratio, colorbar=True),
    ... ])
    >>> fig = (PlotBuilder(layout)
    ...     .draw_image("truth", ground_truth, cmap="magma", title="Ground truth")
    ...     .draw_image("detected", detected, cmap="magma", title="Detected")
    ...     .build())
    """

    def __init__(self, layout: PlotLayout) -> None:
        self.layout = layout
        # cell name -> list of draw ops, run in order (image first, overlays after)
        self._ops: dict[str, list[Panel]] = {}

    def _add(self, cell: str, op: Panel) -> PlotBuilder:
        self._ops.setdefault(cell, []).append(op)
        return self

    def draw_image(self, cell: str, data: ArrayLike, **kwargs: Any) -> PlotBuilder:
        """Draw an image into the named cell. Keyword arguments are forwarded to
        :meth:`BaseVisualizer.draw_image`.
        """
        return self._add(
            cell, lambda axs: BaseVisualizer.draw_image(axs, data, **kwargs)
        )

    def draw_line(
        self,
        cell: str,
        curves: list[dict],
        *,
        hlines: tuple[float, ...] = (),
        xlabel: str | None = None,
        ylabel: str | None = None,
        title: str | None = None,
        yscale: str = "linear",
        legend: bool = False,
    ) -> PlotBuilder:
        """Draw line / errorbar curves into the named cell."""
        return self._add(
            cell,
            lambda axs: BaseVisualizer.draw_line(
                axs, curves, hlines=hlines, xlabel=xlabel, ylabel=ylabel,
                title=title, yscale=yscale, legend=legend,
            ),
        )

    def draw_points(
        self, cell: str, x: ArrayLike, y: ArrayLike, **style: Any
    ) -> PlotBuilder:
        """Layer scatter markers onto the named cell (after its image)."""
        return self._add(
            cell, lambda axs: BaseVisualizer.draw_points(axs, x, y, **style)
        )

    def draw_quiver(
        self,
        cell: str,
        x: ArrayLike,
        y: ArrayLike,
        u: ArrayLike,
        v: ArrayLike,
        *,
        scale: float | None = None,
        color: str = "C0",
        xlabel: str | None = None,
        ylabel: str | None = None,
        title: str | None = None,
        invert_y: bool = False,
    ) -> PlotBuilder:
        """Draw a vector field into the named cell."""
        return self._add(
            cell,
            lambda axs: BaseVisualizer.draw_quiver(
                axs, x, y, u, v, scale=scale, color=color, xlabel=xlabel,
                ylabel=ylabel, title=title, invert_y=invert_y,
            ),
        )

    def build(self, *, suptitle: str | None = None) -> Figure:
        """Compose the figure from the cells' accumulated draw ops."""
        panels = {cell: self._fold(ops) for cell, ops in self._ops.items()}
        return BaseVisualizer.compose(self.layout, panels, suptitle=suptitle)

    @staticmethod
    def _fold(ops: list[Panel]) -> Panel:
        # One panel that runs every op for a cell, returning the last mappable (the
        # image's) so a declared colorbar still gets one.
        def panel(axs: Axes) -> ScalarMappable | None:
            mappable = None
            for op in ops:
                result = op(axs)
                if result is not None:
                    mappable = result
            return mappable

        return panel


def _is_image(value: Any) -> bool:
    """Whether ``value`` is a single 2D image rather than a collection of them."""
    return getattr(value, "ndim", None) == 2


def _as_rows(images: Any) -> list[list[Any]]:
    """Normalise the many shapes ``image_grid`` accepts into rows of images."""
    if _is_image(images):
        return [[images]]
    rows = list(images)
    if all(_is_image(item) for item in rows):
        return [rows]
    return [list(row) for row in rows]


def _per_panel(value: Any, count: int, label: str) -> list[Any]:
    """``value`` as one entry per panel, broadcasting a single value to all of them."""
    if value is None or isinstance(value, str) or np.isscalar(value):
        return [value] * count
    values = list(value)
    if len(values) != count:
        raise ValueError(
            f"Got {len(values)} values for {label} but {count} images. Pass one per "
            "image, or a single value for all of them."
        )
    return values


def image_grid(
    images: Any,
    titles: Any = None,
    cmap: Any = INTENSITY_CMAP,
    vmin: Any = None,
    vmax: Any = None,
    shared_scale: bool = False,
    symmetric: bool = False,
    colorbar: bool = True,
    column_width: float = 3.6,
    names: Sequence[str] | None = None,
    **layout_kwargs: Any,
) -> PlotBuilder:
    """Lay out images with their colorbars, and hand back the builder.

    Nothing is drawn until :meth:`PlotBuilder.build` is called, so markers and
    curves can still be layered on by name::

        image_grid([intensity, phase]).draw_points("0", x, y, color="red").build()

    Args:
        images: One 2D array, a sequence of them for a single row, or a sequence of such
            sequences for a row each.
        titles: One title, or one per image. None leaves them untitled.
        cmap: One colormap, or one per image.
        vmin: Lower limit of the colour scale, or one per image.
        vmax: Upper limit, or one per image.
        shared_scale: Put every panel on one scale, spanning all of them.
        symmetric: Centre the scale on zero, using the largest magnitude present.
        colorbar: Whether to give each panel a colorbar.
        column_width: Width of one grid column in inches.
        names: Cell names to address panels by. Defaults to ``"0"``, ``"1"``, and so on.
        **layout_kwargs: Passed to :class:`PlotLayout`, for example ``margins``.

    Returns:
        PlotBuilder: The builder, with every image queued and nothing drawn yet.

    Raises:
        ValueError: A per-image argument has the wrong length, or no images were given.
    """
    rows = _as_rows(images)
    flat = [image for row in rows for image in row]
    if not flat:
        raise ValueError("image_grid needs at least one image to draw.")

    if symmetric or shared_scale:
        finite = [np.asarray(image, dtype=float) for image in flat]
        if symmetric and vmin is None and vmax is None:
            limit = max(float(np.nanmax(np.abs(image))) for image in finite)
            vmin, vmax = -limit, limit
        elif shared_scale:
            if vmin is None:
                vmin = min(float(np.nanmin(image)) for image in finite)
            if vmax is None:
                vmax = max(float(np.nanmax(image)) for image in finite)

    count = len(flat)
    cell_names = list(names) if names is not None else [str(i) for i in range(count)]
    if len(cell_names) != count:
        raise ValueError(
            f"Got {len(cell_names)} names for {count} images."
        )
    per_title = _per_panel(titles, count, "titles")
    per_cmap = _per_panel(cmap, count, "cmap")
    per_vmin = _per_panel(vmin, count, "vmin")
    per_vmax = _per_panel(vmax, count, "vmax")
    per_colorbar = _per_panel(colorbar, count, "colorbar")

    layout = PlotLayout(column_width=column_width, **layout_kwargs)
    index = 0
    for row in rows:
        layout.add_row(
            [
                GridCell(
                    cell_names[index + offset],
                    # The whole point: the cell is shaped like the array going into it.
                    aspect=image.shape[0] / image.shape[1],
                    colorbar=per_colorbar[index + offset],
                )
                for offset, image in enumerate(row)
            ]
        )
        index += len(row)

    builder = PlotBuilder(layout)
    for position, image in enumerate(flat):
        builder.draw_image(
            cell_names[position],
            image,
            cmap=per_cmap[position],
            vmin=per_vmin[position],
            vmax=per_vmax[position],
            title=per_title[position],
        )
    return builder


def _warn_on_aspect_mismatch(
    layout: PlotLayout, name: str, mappable: Any, tolerance: float = 0.02
) -> None:
    """Warn when a cell is shaped unlike the image drawn into it."""
    cell = layout.cell(name)
    if cell is None or not isinstance(cell.aspect, (int, float)):
        return
    data = getattr(mappable, "get_array", lambda: None)()
    if data is None or getattr(data, "ndim", 0) != 2 or data.shape[1] == 0:
        return

    drawn = data.shape[0] / data.shape[1]
    if abs(drawn - float(cell.aspect)) > tolerance * max(drawn, float(cell.aspect)):
        warnings.warn(
            f"Cell {name!r} has aspect {float(cell.aspect):.3f} but the image drawn "
            f"into it is {data.shape[0]}x{data.shape[1]}, an aspect of {drawn:.3f}. ",
            stacklevel=3,
        )
