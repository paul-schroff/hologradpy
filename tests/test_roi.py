"""Tests for relating a region of interest to the bounds of a frame, and for detecting
one in the pixels kept by a mask.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from hologradpy.roi import ROI

BOUNDS = (40, 60)


def test_a_region_inside_the_bounds_is_left_as_it_is() -> None:
    region = ROI(5, 7, 10, 12)
    assert region.trimmed_to(BOUNDS) == region


@pytest.mark.parametrize(
    ("region", "trimmed"),
    [
        (ROI(-5, -8, 20, 20), ROI(0, 0, 15, 12)),  # over the top-left corner
        (ROI(30, 50, 20, 20), ROI(30, 50, 10, 10)),  # over the bottom-right corner
        (ROI(-10, -10, 80, 90), ROI(0, 0, *BOUNDS)),  # larger than the bounds
    ],
)
def test_a_region_over_an_edge_keeps_the_part_inside(
    region: ROI, trimmed: ROI
) -> None:
    assert region.trimmed_to(BOUNDS) == trimmed


def test_a_trimmed_region_crops_what_lies_inside() -> None:
    """The crop covers the part of the region inside the image. ``crop`` refuses a
    region reaching off the image, so the region is trimmed first.
    """
    image = np.arange(BOUNDS[0] * BOUNDS[1]).reshape(BOUNDS)
    region = ROI(-5, 50, 20, 20)
    np.testing.assert_array_equal(
        region.trimmed_to(BOUNDS).crop(image), image[0:15, 50:60]
    )
    with pytest.raises(ValueError, match="does not lie inside"):
        region.crop(image)


@pytest.mark.parametrize(
    "to_array", [np.asarray, torch.as_tensor], ids=["numpy", "torch"]
)
@pytest.mark.parametrize(
    "region",
    [
        ROI(31, 50, 10, 10),  # over the bottom edge
        ROI(-1, 7, 10, 12),  # over the top edge
        ROI(-12, 7, 10, 12),  # wholly above, which slicing wraps from the bottom
        ROI(5, 7, 0, 12),  # no pixels
    ],
)
def test_crop_refuses_a_region_reaching_off_the_image(region: ROI, to_array) -> None:
    image = to_array(np.zeros(BOUNDS))
    with pytest.raises(ValueError, match="does not lie inside the 40 x 60 image"):
        region.crop(image)


def test_crop_checks_the_trailing_axes_of_a_stack() -> None:
    """A stack of frames is checked against the size of one frame."""
    stack = np.zeros((3, *BOUNDS))
    assert ROI(30, 50, 10, 10).crop(stack).shape == (3, 10, 10)
    with pytest.raises(ValueError, match="does not lie inside"):
        ROI(31, 50, 10, 10).crop(stack)


def test_pad_refuses_a_region_reaching_off_the_frame() -> None:
    with pytest.raises(ValueError, match="does not lie inside the 40 x 60 frame"):
        ROI(-12, 7, 10, 12).pad(np.ones((10, 12)), BOUNDS)


@pytest.mark.parametrize(
    ("region", "inside"),
    [
        (ROI(5, 7, 10, 12), True),
        (ROI(0, 0, *BOUNDS), True),  # the whole frame
        (ROI(30, 50, 10, 10), True),  # against the bottom-right corner
        (ROI(31, 50, 10, 10), False),  # one row over the bottom edge
        (ROI(-1, 7, 10, 12), False),  # one row over the top edge
        (ROI(5, 7, 0, 12), False),  # no pixels
        (ROI(-10, -10, 80, 90), False),  # larger than the frame
    ],
)
def test_lies_inside_needs_every_pixel_inside_a_nonempty_region(
    region: ROI, inside: bool
) -> None:
    assert region.lies_inside(BOUNDS) is inside


def test_detect_bounds_only_the_pixels_the_mask_keeps() -> None:
    """A brighter blob outside the mask neither sets the threshold nor enters the
    region.
    """
    image = np.zeros(BOUNDS)
    image[5:8, 5:8] = 10.0  # the brighter blob, masked out
    image[30:33, 40:44] = 4.0
    kept = np.ones(BOUNDS, dtype=bool)
    kept[:20, :20] = False

    assert ROI.detect(image, threshold=0.5, pad=0) == ROI(5, 5, 3, 3)
    assert ROI.detect(image, threshold=0.5, pad=0, mask=kept) == ROI(30, 40, 3, 4)


@pytest.mark.parametrize(
    "region", [ROI(40, 0, 5, 5), ROI(0, -9, 5, 9), ROI(-20, -20, 10, 10)]
)
def test_a_region_outside_the_bounds_is_refused(region: ROI) -> None:
    with pytest.raises(ValueError, match="No part of"):
        region.trimmed_to(BOUNDS)
