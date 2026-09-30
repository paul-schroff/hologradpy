"""The native region-of-interest value object.

``ROI`` is the ``(row, col)`` rectangular region of interest and the single abstraction
for ROI handling. Its named constructors are :meth:`ROI.centered`,
:meth:`ROI.from_bounds` and :meth:`ROI.detect`, and its methods include :meth:`crop` and
:meth:`pad`. :meth:`lies_inside`, :meth:`moved_inside` and :meth:`trimmed_to` relate a
region to the bounds of a frame. It works on numpy and torch arrays.
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
    the extent, so :meth:`crop` selects ``image[..., roi.rows, roi.columns]`` from an
    image the region lies inside.
    """

    top_row: int
    left_column: int
    height: int
    width: int

    @classmethod
    def centered(cls, center: tuple[float, float], size: tuple[int, int]) -> ROI:
        """An ROI of ``(height, width)`` ``size`` centered on ``(row, col)`` ``center``.

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

    def trimmed_to(self, bounds: tuple[int, int]) -> ROI:
        """Trims the ROI to the part of it inside ``bounds``, keeping its position.

        Args:
            bounds: The ``(height, width)`` to stay within, usually a sensor.

        Returns:
            The part of the region inside ``bounds``, which is the region itself when
            it lies inside already.

        Raises:
            ValueError: No part of the region lies inside ``bounds``.
        """
        height_bound, width_bound = int(bounds[0]), int(bounds[1])
        top, bottom, left, right = self.to_bounds()
        trimmed = ROI.from_bounds(
            max(top, 0),
            min(bottom, height_bound),
            max(left, 0),
            min(right, width_bound),
        )
        if trimmed.height <= 0 or trimmed.width <= 0:
            raise ValueError(
                f"No part of {self} lies inside {height_bound} x {width_bound}."
            )
        return trimmed

    def lies_inside(self, bounds: tuple[int, int]) -> bool:
        """Whether the ROI holds at least one pixel, and every pixel of it lies inside
        ``bounds``.

        Args:
            bounds: The ``(height, width)`` to lie within, usually a frame.

        Returns:
            True for a nonempty region inside ``bounds``.
        """
        height_bound, width_bound = int(bounds[0]), int(bounds[1])
        top, bottom, left, right = self.to_bounds()
        return (
            self.height >= 1
            and self.width >= 1
            and top >= 0
            and left >= 0
            and bottom <= height_bound
            and right <= width_bound
        )

    @classmethod
    def from_bounds(cls, top: int, bottom: int, left: int, right: int) -> ROI:
        """From ``(top, bottom, left, right)`` pixel indices, the convention returned by
        :meth:`to_bounds` and used by array slicing.
        """
        return cls(int(top), int(left), int(bottom) - int(top), int(right) - int(left))

    @classmethod
    def detect(
        cls,
        image: ArrayLike,
        threshold: float = 0.5,
        pad: int = 10,
        mask: ArrayLike | None = None,
    ) -> ROI:
        """The ROI bounding pixels above ``threshold * max(image)``, padded by ``pad``
        pixels per side and trimmed to the image extent.

        Args:
            image: The image to search, with two spatial axes.
            threshold: The fraction of the maximum a pixel has to exceed.
            pad: The pixels added on each side of the bounding box.
            mask: True at the pixels to consider, in the shape of ``image``. The
                maximum is taken over these pixels, and only these pixels can exceed
                the threshold. Every pixel is considered when None.

        Returns:
            The padded bounding box of the pixels above the threshold.
        """
        xp = array_namespace(image)
        if mask is None:
            above = image > threshold * xp.max(image)
        else:
            above = mask & (image > threshold * xp.max(image[mask]))
        rows, cols = xp.nonzero(above)
        top = int(xp.clip(xp.min(rows) - pad, 0, image.shape[0]))
        bottom = int(xp.clip(xp.max(rows) + pad + 1, 0, image.shape[0]))
        left = int(xp.clip(xp.min(cols) - pad, 0, image.shape[1]))
        right = int(xp.clip(xp.max(cols) + pad + 1, 0, image.shape[1]))
        return cls(top, left, bottom - top, right - left)

    def crop(self, image: ArrayLike) -> ArrayLike:
        """Crop ``image`` (any array with two trailing spatial axes) to this ROI.

        Raises:
            ValueError: The region is empty or reaches outside the two trailing axes of
                ``image``. :meth:`trimmed_to` gives the part of a region inside them.
        """
        height, width = (int(size) for size in image.shape[-2:])
        if not self.lies_inside((height, width)):
            raise ValueError(
                f"{self} does not lie inside the {height} x {width} image. Trim it to "
                "the image first with trimmed_to."
            )
        return image[..., self.rows, self.columns]

    def pad(
        self, image: ArrayLike, original_shape: tuple[int, int]
    ) -> ArrayLike:
        """Inverse of :meth:`crop`: place ``image`` back into a zero array of
        ``original_shape`` ``(height, width)`` at this ROI.

        Raises:
            ValueError: The region is empty or reaches outside ``original_shape``.
        """
        if not self.lies_inside(original_shape):
            raise ValueError(
                f"{self} does not lie inside the {int(original_shape[0])} x "
                f"{int(original_shape[1])} frame."
            )
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
