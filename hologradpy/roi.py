"""The native region-of-interest value object.

``ROI`` is the ``(row, col)`` rectangular region of interest and the single abstraction
for ROI handling with named constructors :meth:`ROI.centered`, :meth:`ROI.from_bounds`
and :meth:`ROI.detect`, and methods :meth:`crop`, :meth:`pad`. It works on numpy and
torch arrays.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeVar

import torch
from numpy.typing import NDArray

from array_api_compat import array_namespace, device as array_device

from .serialization import record_type

ArrayLike = TypeVar("ArrayLike", torch.Tensor, NDArray)


@record_type("roi")
@dataclass(frozen=True)
class ROI:
    """A rectangular region of interest in native ``(row, col)`` pixel coordinates.

    ``top_row`` / ``left_column`` are the top-left corner and ``height`` / ``width``
    the extent, so ``image[roi.rows, roi.columns]`` (or :meth:`crop`) selects it.
    """

    top_row: int
    left_column: int
    height: int
    width: int

    @classmethod
    def centered(cls, center: tuple[float, float], size: tuple[int, int]) -> ROI:
        """An ROI of ``(height, width)`` ``size`` centred on ``(row, col)`` ``center``.

        The corner is floored, matching the pixel-grid convention used across the
        calibrators. ``size`` is coerced to int first, so a ``size`` carrying a float
        (for example a fractional magnification) still yields an integer-pixel ROI.
        """
        center_row, center_col = center
        height, width = int(size[0]), int(size[1])
        return cls(
            int(center_row) - height // 2,
            int(center_col) - width // 2,
            height,
            width,
        )

    def moved_inside(self, bounds: tuple[int, int]) -> ROI:
        """Moves the ROI so it sits inside ``bounds``, keeping its size.

        Args:
            bounds: The ``(height, width)`` to stay within, usually a sensor.

        Returns:
            The region, moved if it had to be.

        Raises:
            ValueError: The region is larger than ``bounds``, so moving cannot place it
                there. Kept at its size it would report one shape and crop to another.
        """
        height_bound, width_bound = bounds
        if self.height > height_bound or self.width > width_bound:
            raise ValueError(
                f"A {self.height} x {self.width} region does not fit inside "
                f"{height_bound} x {width_bound}."
            )
        return ROI(
            max(0, min(self.top_row, height_bound - self.height)),
            max(0, min(self.left_column, width_bound - self.width)),
            self.height,
            self.width,
        )

    @classmethod
    def from_bounds(cls, top: int, bottom: int, left: int, right: int) -> ROI:
        """From ``(top, bottom, left, right)`` pixel indices, the convention returned by
        :meth:`to_bounds` and used by array slicing."""
        return cls(int(top), int(left), int(bottom) - int(top), int(right) - int(left))

    @classmethod
    def detect(
        cls, image: ArrayLike, threshold: float = 0.5, pad: int = 10
    ) -> ROI:
        """The ROI bounding pixels above ``threshold * max(image)``, padded by ``pad``
        pixels per side and clipped to the image extent."""
        xp = array_namespace(image)
        rows, cols = xp.nonzero(image > threshold * xp.max(image))
        top = int(xp.clip(xp.min(rows) - pad, 0, image.shape[0]))
        bottom = int(xp.clip(xp.max(rows) + pad + 1, 0, image.shape[0]))
        left = int(xp.clip(xp.min(cols) - pad, 0, image.shape[1]))
        right = int(xp.clip(xp.max(cols) + pad + 1, 0, image.shape[1]))
        return cls(top, left, bottom - top, right - left)

    def crop(self, image: ArrayLike) -> ArrayLike:
        """Crop ``image`` (any array with two trailing spatial axes) to this ROI."""
        return image[..., self.rows, self.columns]

    def pad(
        self, image: ArrayLike, original_shape: tuple[int, int]
    ) -> ArrayLike:
        """Inverse of :meth:`crop`: place ``image`` back into a zero array of
        ``original_shape`` ``(height, width)`` at this ROI."""
        xp = array_namespace(image)
        output = xp.zeros(
            (*image.shape[:-2], *original_shape),
            dtype=image.dtype,
            device=array_device(image),
        )
        output[..., self.rows, self.columns] = image
        return output

    @property
    def rows(self) -> slice:
        """The row (axis-0) slice this ROI covers."""
        return slice(self.top_row, self.top_row + self.height)

    @property
    def columns(self) -> slice:
        """The column (axis-1) slice this ROI covers."""
        return slice(self.left_column, self.left_column + self.width)

    def to_bounds(self) -> tuple[int, int, int, int]:
        """To ``(top, bottom, left, right)`` pixel indices."""
        return (
            self.top_row,
            self.top_row + self.height,
            self.left_column,
            self.left_column + self.width,
        )
