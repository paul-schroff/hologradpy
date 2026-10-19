"""The holograms and frames a fit reads: built from a capture file, cropped to the
region of interest, background subtracted, scaled to mean 1 over the region, and
batched by a torch data loader.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import DataLoader

from hologradpy.datasets import CaptureStore, HologramFrames
from hologradpy.serialization import SaveableRecord, record_type

FRAME_SHAPE = (12, 16)
LEVEL_SHAPE = (4, 5)


@record_type("test_hologram_frames_capture")
@dataclass
class _Capture(SaveableRecord):
    """The smallest record that can describe a capture."""

    note: str = "a capture"


def _model() -> SimpleNamespace:
    """What the set reads from a model: the device and the real dtype."""
    return SimpleNamespace(
        device=torch.device("cpu"), init_field=SimpleNamespace(dtype_r=torch.float64)
    )


def _write_capture(path, count: int = 3) -> tuple[list[np.ndarray], list[np.ndarray]]:
    rng = np.random.default_rng(0)
    frames = [100.0 + 4000.0 * rng.random(FRAME_SHAPE) for _ in range(count)]
    levels = [
        rng.integers(0, 256, LEVEL_SHAPE, dtype=np.uint8) for _ in range(count)
    ]
    CaptureStore.write(
        path, _Capture(), camera_images=frames, slm_levels=levels, phase_bitdepth=8
    )
    return frames, levels


def _region_mask() -> np.ndarray:
    """A disc of the sensor, so the region is not its bounding box."""
    rows, columns = np.indices(FRAME_SHAPE)
    return (rows - 6.0) ** 2 + (columns - 9.0) ** 2 <= 16.0


def test_a_capture_is_cropped_and_scaled_to_mean_one(tmp_path) -> None:
    """Every frame is cropped to the bounding box of the mask and scaled to mean 1 over
    the mask, the levels and the bit depth are those stored, and the file is closed.
    """
    path = tmp_path / "capture.asdf"
    frames, levels = _write_capture(path)
    mask = _region_mask()

    hologram_frames = HologramFrames.from_capture(path, _model(), mask)

    rows, columns = np.nonzero(mask)
    box = (slice(rows.min(), rows.max() + 1), slice(columns.min(), columns.max() + 1))
    region = mask[box]
    assert len(hologram_frames) == 3
    assert hologram_frames.bitdepth == 8
    assert hologram_frames.band_mask is None
    assert hologram_frames.background_level is None
    np.testing.assert_array_equal(hologram_frames.region.numpy(), region)
    for index in range(3):
        expected = frames[index][box] / frames[index][box][region].mean()
        np.testing.assert_allclose(
            hologram_frames.frames[index].numpy(), expected, rtol=1e-6
        )
        np.testing.assert_array_equal(
            hologram_frames.levels[index].numpy(), levels[index]
        )
    # The file is free to be written over.
    _write_capture(path)


def test_a_background_is_subtracted_before_the_frames_are_scaled(tmp_path) -> None:
    """A level and a whole-sensor frame are both taken off every frame, and the
    background level is their mean over the region.
    """
    path = tmp_path / "capture.asdf"
    frames, _ = _write_capture(path)
    mask = _region_mask()
    rows, columns = np.nonzero(mask)
    box = (slice(rows.min(), rows.max() + 1), slice(columns.min(), columns.max() + 1))
    region = mask[box]
    background_frame = np.linspace(0.0, 50.0, mask.size).reshape(FRAME_SHAPE)

    for background, level in (
        (7.0, 7.0),
        (background_frame, float(background_frame[box][region].mean())),
    ):
        hologram_frames = HologramFrames.from_capture(
            path, _model(), mask, background=background
        )
        lifted = frames[1] - background
        expected = lifted[box] / lifted[box][region].mean()
        assert hologram_frames.background_level == level
        np.testing.assert_allclose(
            hologram_frames.frames[1].numpy(), expected, rtol=1e-6
        )


def test_appended_holograms_are_batched_by_a_data_loader() -> None:
    """Appended frames are scaled to mean 1 over the region, and a data loader stacks
    the levels, the frames and the positions of a batch.
    """
    region = torch.zeros(FRAME_SHAPE, dtype=torch.bool)
    region[2:9, 3:12] = True
    hologram_frames = HologramFrames(region, bitdepth=8, dtype=torch.float64)
    rng = np.random.default_rng(1)
    for _ in range(2):
        hologram_frames.append(
            rng.integers(0, 256, LEVEL_SHAPE, dtype=np.uint8),
            torch.as_tensor(5.0 * rng.random(FRAME_SHAPE)),
        )

    (batch,) = DataLoader(hologram_frames, batch_size=2)

    assert len(hologram_frames) == 2
    assert batch["levels"].shape == (2, *LEVEL_SHAPE)
    assert batch["frame"].shape == (2, *FRAME_SHAPE)
    assert batch["index"].tolist() == [0, 1]
    means = (hologram_frames.frames * region).sum(dim=(-2, -1)) / region.sum()
    torch.testing.assert_close(means, torch.ones(2, dtype=torch.float64))


def test_a_subset_keeps_the_grid_and_the_holograms_it_names(tmp_path) -> None:
    path = tmp_path / "capture.asdf"
    _write_capture(path)
    hologram_frames = HologramFrames.from_capture(path, _model(), _region_mask())

    subset = hologram_frames.subset([2, 0])

    assert len(subset) == 2
    assert subset.roi == hologram_frames.roi
    assert subset.bitdepth == hologram_frames.bitdepth
    torch.testing.assert_close(subset.frames[0], hologram_frames.frames[2])
    torch.testing.assert_close(subset.levels[1], hologram_frames.levels[0])
