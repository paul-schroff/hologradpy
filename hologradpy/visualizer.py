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
  cell name to a ``panel(axs)`` callable -- no frame, since most plots are static.
  :class:`AnimatedVisualizer` adds the animation / GIF machinery for the few plots that
  move: an animation is just composing the static panels once per frame.

Because a panel only paints into the axes it is handed, panels from different
visualizers can be mixed and matched into one :class:`PlotLayout` and rendered together.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Callable

import numpy as np

if TYPE_CHECKING:
    from matplotlib.animation import FuncAnimation
    from matplotlib.axes import Axes
    from matplotlib.cm import ScalarMappable
    from matplotlib.figure import Figure


# A panel paints into the axes it is handed and returns a mappable if it wants a
# colorbar. Defined precisely for type checkers; at runtime it is just Callable so the
# module needs no eager matplotlib import (Axes / ScalarMappable are
# TYPE_CHECKING-only).
if TYPE_CHECKING:
    Panel = Callable[[Axes], ScalarMappable | None]
else:
    Panel = Callable


class VisualizationData:
    """Base for a result's visualization payload.

    A concrete subclass (e.g. ``RasterVisualizationData``) carries the arrays, fits and
    coordinates a visualizer needs to render a result. Kept as a plain, domain-agnostic
    marker base so result dataclasses across the codebase can expose a
    ``visualization_data: VisualizationData | None`` field -- and save / load it --
    without depending on any one producer.
    """


@dataclass
class GridCell:
    """Geometry of one named axes cell in a :class:`PlotLayout`.

    Parameters
    ----------
    name : str
        Key used to look the axes up in ``layout.axes[name]``.
    colspan : int
        Number of grid columns the cell spans.
    aspect : float | str
        Cell height-to-width ratio. A float is the ratio ``height / width``;
        ``"equal"`` is a square cell (ratio 1.0); ``"auto"`` leaves the height to
        ``height`` (or the row default) -- use it for line plots.
    colorbar : bool
        If True, a matched-height colorbar axes is appended on the right and
        exposed as ``layout.colorbar_axes[name]``.
    height : float | None
        Explicit cell height [inches], overriding ``aspect`` for the row height.
        Mainly for ``aspect="auto"`` line rows.
    sharex : str | None
        Name of another cell whose x-axis this cell should share.
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
    equal width; each row's height follows from its cells' aspect ratios so images stay
    undistorted and rows line up. Built on ``mpl_toolkits.axes_grid1.Divider`` (absolute
    inch sizing) with ``make_axes_locatable`` for matched colorbars.
    """

    def __init__(
        self,
        column_width: float = 3.2,
        col_gap: float = 0.3,
        row_gap: float = 0.5,
        margins: tuple[float, float, float, float] = (0.12, 0.12, 0.6, 0.18),
        colorbar_width: float = 0.15,
        colorbar_pad: float = 0.08,
    ) -> None:
        # margins: (left, right, top, bottom) in inches.
        self.column_width = column_width
        self.col_gap = col_gap
        self.row_gap = row_gap
        self.margins = margins
        self.colorbar_width = colorbar_width
        self.colorbar_pad = colorbar_pad

        self._rows: list[list[GridCell]] = []
        self.figure: Figure | None = None
        self.axes: dict[str, Axes] = {}
        self.colorbar_axes: dict[str, Axes] = {}

    def add_row(self, cells: list[GridCell]) -> PlotLayout:
        """Append a row of cells (left to right). Returns self for chaining."""
        self._rows.append(list(cells))
        return self

    def copy(self) -> PlotLayout:
        """Return an independent clone (same style + cells) for reuse.

        A configured layout can seed many figures; ``copy()`` lets you tweak one (e.g.
        add a row) without disturbing the original.
        """
        clone = PlotLayout(
            column_width=self.column_width,
            col_gap=self.col_gap,
            row_gap=self.row_gap,
            margins=self.margins,
            colorbar_width=self.colorbar_width,
            colorbar_pad=self.colorbar_pad,
        )
        clone._rows = [[replace(cell) for cell in row] for row in self._rows]
        return clone

    def _cell_height(self, cell: GridCell) -> float | None:
        """Natural cell height [inches], or None when sizing defers to the row."""
        if cell.height is not None:
            return cell.height
        if cell.aspect == "auto":
            return None
        width = (
            cell.colspan * self.column_width + (cell.colspan - 1) * self.col_gap
        )
        ratio = 1.0 if cell.aspect == "equal" else float(cell.aspect)
        return width * ratio

    def build(self, suptitle: str | None = None) -> Figure:
        """Create the figure and all cell axes; return the figure."""
        import matplotlib.pyplot as plt
        from mpl_toolkits.axes_grid1 import Divider, Size

        number_of_columns = max(
            sum(cell.colspan for cell in row) for row in self._rows
        )
        row_heights = []
        for row in self._rows:
            heights = [
                height
                for height in (self._cell_height(cell) for cell in row)
                if height is not None
            ]
            row_heights.append(max(heights) if heights else self.column_width)

        # A column needs a colorbar gutter if a (colspan-1) cell ending in it asks for
        # one. Colorbars are placed in their own Divider cell -- not via
        # make_axes_locatable -- so the single Divider positions everything and the
        # panels stay aligned.
        column_has_colorbar = [False] * number_of_columns
        for row in self._rows:
            column = 0
            for cell in row:
                if cell.colorbar:
                    column_has_colorbar[column + cell.colspan - 1] = True
                column += cell.colspan

        left, right, top, bottom = self.margins
        # The last column's colorbar ticks/labels sit in the right margin; widen it so
        # they are not clipped.
        if column_has_colorbar[-1]:
            right += 0.45
        gutter = self.colorbar_pad + self.colorbar_width
        figure_width = (
            left
            + right
            + number_of_columns * self.column_width
            + (number_of_columns - 1) * self.col_gap
            + sum(gutter for has in column_has_colorbar if has)
        )
        figure_height = (
            top
            + bottom
            + sum(row_heights)
            + (len(self._rows) - 1) * self.row_gap
        )

        figure = plt.figure(figsize=(figure_width, figure_height))

        # Horizontal sizes: left margin, then per column a content cell, an optional
        # colorbar gutter (pad + bar), and a gap. Record the Divider index of each
        # column's content cell and colorbar cell.
        horizontal: list = [Size.Fixed(left)]
        content_index: list[int] = []
        colorbar_index: list[int | None] = []
        for column in range(number_of_columns):
            horizontal.append(Size.Fixed(self.column_width))
            content_index.append(len(horizontal) - 1)
            if column_has_colorbar[column]:
                horizontal.append(Size.Fixed(self.colorbar_pad))
                horizontal.append(Size.Fixed(self.colorbar_width))
                colorbar_index.append(len(horizontal) - 1)
            else:
                colorbar_index.append(None)
            if column < number_of_columns - 1:
                horizontal.append(Size.Fixed(self.col_gap))
        horizontal.append(Size.Fixed(right))

        # Vertical sizes are bottom-to-top: bottom margin, rows in reverse order
        # interleaved with gaps, then top margin.
        vertical: list = [Size.Fixed(bottom)]
        for index, height in enumerate(reversed(row_heights)):
            vertical.append(Size.Fixed(height))
            if index < len(row_heights) - 1:
                vertical.append(Size.Fixed(self.row_gap))
        vertical.append(Size.Fixed(top))

        divider = Divider(figure, (0, 0, 1, 1), horizontal, vertical, aspect=False)

        number_of_rows = len(self._rows)
        for row_index, row in enumerate(self._rows):
            ny = 1 + 2 * (number_of_rows - 1 - row_index)
            column = 0
            for cell in row:
                last_column = column + cell.colspan - 1
                nx = content_index[column]
                nx1 = content_index[last_column] + 1
                axs = figure.add_axes(
                    divider.get_position(),
                    axes_locator=divider.new_locator(nx=nx, nx1=nx1, ny=ny),
                )
                self.axes[cell.name] = axs
                if cell.colorbar:
                    cbar_nx = colorbar_index[last_column]
                    self.colorbar_axes[cell.name] = figure.add_axes(
                        divider.get_position(),
                        axes_locator=divider.new_locator(nx=cbar_nx, ny=ny),
                    )
                column += cell.colspan

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
        data,
        *,
        cmap: str = "viridis",
        vmin: float | None = None,
        vmax: float | None = None,
        title: str | None = None,
        interpolation: str | None = None,
    ) -> ScalarMappable:
        """Show ``data`` as an image; return the mappable (for a colorbar)."""
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
        x,
        y,
        *,
        marker: str = "o",
        color: str = "red",
        size: float = 8.0,
        edgecolor: str = "white",
        **kwargs,
    ) -> None:
        """Overlay scatter points (markers only) on top of an image/axes."""
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
    """A :class:`BaseVisualizer` whose figure is animated -- one static frame per
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

        Each frame is quantized with a 256-colour adaptive palette and Floyd-Steinberg
        dithering. matplotlib's default GIF writer maps every frame onto the 216-colour
        web-safe palette, which badly posterizes smooth gradients; the adaptive,
        dithered palette keeps them smooth. (Note ``quantize(colors=...)`` median-cut
        does not dither -- only the ``palette=`` path does -- so the dithering happens
        in a second pass.)
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

    The layout defines the grid of named :class:`GridCell`s (geometry only); the builder
    draws into them *by name*. Every ``draw_*`` method takes the target cell name first
    and returns ``self`` so calls chain, and repeated calls on the same cell layer (e.g.
    an image then markers). :meth:`build` folds each cell's ops into one panel and
    composes the figure -- it is pure sugar over :meth:`BaseVisualizer.compose`, so no
    dict or ``lambda axs:`` is needed.

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

    def draw_image(
        self,
        cell: str,
        data,
        *,
        cmap: str = "viridis",
        vmin: float | None = None,
        vmax: float | None = None,
        title: str | None = None,
        interpolation: str | None = None,
    ) -> PlotBuilder:
        """Draw an image into the named cell."""
        return self._add(
            cell,
            lambda axs: BaseVisualizer.draw_image(
                axs, data, cmap=cmap, vmin=vmin, vmax=vmax, title=title,
                interpolation=interpolation,
            ),
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

    def draw_points(self, cell: str, x, y, **style) -> PlotBuilder:
        """Layer scatter markers onto the named cell (after its image)."""
        return self._add(
            cell, lambda axs: BaseVisualizer.draw_points(axs, x, y, **style)
        )

    def build(self, *, suptitle: str | None = None) -> Figure:
        """Compose the figure from the cells' accumulated draw ops."""
        panels = {cell: self._fold(ops) for cell, ops in self._ops.items()}
        return BaseVisualizer.compose(self.layout, panels, suptitle=suptitle)

    @staticmethod
    def _fold(ops: list[Panel]) -> Panel:
        # One panel that runs every op for a cell, returning the last mappable (the
        # image's) so a declared colorbar still gets one.
        def panel(axs):
            mappable = None
            for op in ops:
                result = op(axs)
                if result is not None:
                    mappable = result
            return mappable

        return panel
