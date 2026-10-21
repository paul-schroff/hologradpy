"""The pixel frame and the physical frame of a plane of 4 by 6 pixels.

The positions come from :func:`hologradpy.grids.get_spatial_grid` and
:func:`hologradpy.grids.plane_center`, so the figure follows the package.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.patches import FancyArrowPatch, Rectangle

from hologradpy.grids import get_spatial_grid, plane_center

PLANE_RESOLUTION = (4, 6)  # (height, width)
HIGHLIGHTED_PIXEL = (1, 3)  # (column, row)
# The physical frame is drawn this many pitches to the right of the pixel frame.
PHYSICAL_PANEL_OFFSET = 8.8
# How far an axis arrow reaches past the edge of the plane, in pitches.
AXIS_OVERHANG = 0.7
PIXEL_FRAME_COLOUR = "tab:blue"
PHYSICAL_FRAME_COLOUR = "tab:red"
HIGHLIGHT_COLOUR = "#fde3b5"
ORIGIN_COLOUR = "0.85"

plt.rcParams["savefig.bbox"] = "tight"
plt.rcParams["savefig.pad_inches"] = 0.02

height, width = PLANE_RESOLUTION
# Positions in pitches. Adding zero makes a negative zero print as 0 in the labels.
x, y = (axis.numpy() + 0.0 for axis in get_spatial_grid(PLANE_RESOLUTION, (1.0, 1.0)))
origin = plane_center(PLANE_RESOLUTION)
left, right = x.min() - 0.5, x.max() + 0.5
bottom, top = y.min() - 0.5, y.max() + 0.5
# The titles sit above the tip of the y axis.
title_height = top + AXIS_OVERHANG + 0.3


def draw_pixels(axes: Axes, offset: float, labels: dict[tuple[int, int], str]) -> None:
    """Draw every pixel at its physical position, in pitches, and label some.

    Args:
        axes: The axes to draw on.
        offset: The shift of the plane along the horizontal axis, in pitches.
        labels: The text of a pixel, keyed by its ``(column, row)``.
    """
    for row in range(height):
        for column in range(width):
            face = {HIGHLIGHTED_PIXEL: HIGHLIGHT_COLOUR, origin: ORIGIN_COLOUR}.get(
                (column, row), "white"
            )
            axes.add_patch(
                Rectangle(
                    (offset + x[row, column] - 0.5, y[row, column] - 0.5),
                    1.0,
                    1.0,
                    facecolor=face,
                    edgecolor="0.45",
                )
            )
    for (column, row), text in labels.items():
        axes.text(
            offset + x[row, column],
            y[row, column],
            text,
            ha="center",
            va="center",
            fontsize=9,
        )


def arrow(
    axes: Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    colour: str,
    curvature: float = 0.0,
) -> None:
    """Draw an arrow from exactly ``start`` to exactly ``end``.

    Args:
        axes: The axes to draw on.
        start: Where the arrow starts.
        end: Where its head points.
        colour: The colour of the arrow.
        curvature: The bend of the arrow, as the ``rad`` of an ``arc3`` connection.
            Positive values bend it to the right of its direction.
    """
    axes.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=13,
            color=colour,
            linewidth=1.5,
            shrinkA=0.0,
            shrinkB=0.0,
            connectionstyle=f"arc3,rad={curvature}",
        )
    )


figure, axes = plt.subplots(figsize=(9.2, 3.6))
axes.set_aspect("equal")
axes.axis("off")

# The pixel frame counts columns to the right and rows down from the pixel (0, 0).
draw_pixels(
    axes,
    0.0,
    {
        HIGHLIGHTED_PIXEL: "{}, {}".format(*HIGHLIGHTED_PIXEL),
        origin: "{}, {}".format(*origin),
    },
)
first_x, first_y = x[0, 0], y[0, 0]
axes.text(first_x - 0.25, first_y + 0.25, "0, 0", ha="center", va="center", fontsize=9)
for column in range(width):
    axes.text(
        x[0, column],
        top + 0.12,
        str(column),
        ha="center",
        va="bottom",
        color=PIXEL_FRAME_COLOUR,
    )
for row in range(height):
    axes.text(
        left - 0.12,
        y[row, 0],
        str(row),
        ha="right",
        va="center",
        color=PIXEL_FRAME_COLOUR,
    )
arrow(axes, (first_x, first_y), (right + AXIS_OVERHANG, first_y), PIXEL_FRAME_COLOUR)
arrow(axes, (first_x, first_y), (first_x, bottom - AXIS_OVERHANG), PIXEL_FRAME_COLOUR)
axes.text(
    right + AXIS_OVERHANG + 0.1,
    first_y,
    "column",
    va="center",
    color=PIXEL_FRAME_COLOUR,
)
axes.text(
    first_x,
    bottom - AXIS_OVERHANG - 0.1,
    "row",
    ha="center",
    va="top",
    color=PIXEL_FRAME_COLOUR,
)
# A positive angle turns +column towards +row, clockwise on the screen.
arrow(
    axes,
    (first_x + 1.25, first_y - 0.3),
    (first_x + 0.3, first_y - 1.25),
    PIXEL_FRAME_COLOUR,
    curvature=-0.4,
)
axes.text(
    first_x + 1.2,
    first_y - 1.05,
    "positive angle",
    va="top",
    fontsize=9,
    color=PIXEL_FRAME_COLOUR,
)
axes.text(
    right,
    bottom - 0.12,
    "image[row, column]",
    ha="right",
    va="top",
    family="monospace",
    fontsize=9,
)
axes.text(
    (left + right) / 2,
    title_height,
    "Pixel frame, (column, row)",
    ha="center",
    va="bottom",
    fontsize=12,
    color=PIXEL_FRAME_COLOUR,
)

# The physical frame has its origin at the pixel (width // 2, height // 2), with x to
# the right and y up.
offset = PHYSICAL_PANEL_OFFSET
highlighted_column, highlighted_row = HIGHLIGHTED_PIXEL
draw_pixels(
    axes,
    offset,
    {
        HIGHLIGHTED_PIXEL: f"{x[highlighted_row, highlighted_column]:g}, "
        f"{y[highlighted_row, highlighted_column]:g}"
    },
)
axes.text(offset - 0.25, -0.25, "0, 0", ha="center", va="center", fontsize=9)
for column in range(width):
    axes.text(
        offset + x[0, column],
        bottom - 0.12,
        f"{x[0, column]:g}",
        ha="center",
        va="top",
        color=PHYSICAL_FRAME_COLOUR,
    )
for row in range(height):
    axes.text(
        offset + left - 0.12,
        y[row, 0],
        f"{y[row, 0]:g}",
        ha="right",
        va="center",
        color=PHYSICAL_FRAME_COLOUR,
    )
arrow(axes, (offset, 0.0), (offset + right + AXIS_OVERHANG, 0.0), PHYSICAL_FRAME_COLOUR)
arrow(axes, (offset, 0.0), (offset, top + AXIS_OVERHANG), PHYSICAL_FRAME_COLOUR)
axes.text(
    offset + right + AXIS_OVERHANG + 0.1,
    0.0,
    "x",
    va="center",
    fontsize=12,
    color=PHYSICAL_FRAME_COLOUR,
)
axes.text(
    offset + 0.12,
    top + AXIS_OVERHANG,
    "y",
    va="center",
    fontsize=12,
    color=PHYSICAL_FRAME_COLOUR,
)
# A positive angle turns +x towards +y, counterclockwise on the screen.
arrow(
    axes,
    (offset + 1.25, 0.3),
    (offset + 0.3, 1.25),
    PHYSICAL_FRAME_COLOUR,
    curvature=0.4,
)
axes.text(
    offset + 1.2,
    1.05,
    "positive angle",
    va="bottom",
    fontsize=9,
    color=PHYSICAL_FRAME_COLOUR,
)
axes.text(
    offset + (left + right) / 2,
    title_height,
    "Physical frame, (x, y)",
    ha="center",
    va="bottom",
    fontsize=12,
    color=PHYSICAL_FRAME_COLOUR,
)

# The limits hold the arrows and a margin above the titles, and the tight bounding box
# crops the rest.
axes.set_xlim(left - 0.5, offset + right + AXIS_OVERHANG + 0.05)
axes.set_ylim(bottom - AXIS_OVERHANG - 0.05, title_height + 0.65)
figure.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0)
