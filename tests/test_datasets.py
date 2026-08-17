"""One set, one file: what the two stores write, read and survive.

A capture is camera frames and the levels that produced them, from a bench. A
retrieval's steps are the optimiser's own parameter, from a search. They are separate
classes because they are separate things, and the combinations that would mean nothing
are not writable rather than rejected at runtime.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest
import torch

from hologradpy.datasets import (
    SAMPLE_DTYPE,
    CaptureStore,
    RetrievalStepStore,
    SampleDataset,
)
from hologradpy.serialization import SaveableRecord, record_type


@record_type("test_capture")
@dataclass
class _Capture(SaveableRecord):
    """The smallest record that can describe a capture."""

    note: str = "a capture"
    seed: int = 0


def _frames(count: int, shape=(6, 8)) -> list[np.ndarray]:
    rng = np.random.default_rng(0)
    return [rng.random(shape) * 4095 for _ in range(count)]


def _levels(count: int, shape=(4, 5), bitdepth: int = 8) -> list[np.ndarray]:
    rng = np.random.default_rng(1)
    dtype = np.uint8 if bitdepth <= 8 else np.uint16
    return [rng.integers(0, 2**bitdepth, shape, dtype=dtype) for _ in range(count)]


def _capture(path, frames, levels=None, bitdepth: int = 8) -> None:
    with CaptureStore.capture(
        path,
        _Capture(),
        frame_shape=frames[0].shape,
        slm_levels=_levels(len(frames)) if levels is None else levels,
        phase_bitdepth=bitdepth,
    ) as store:
        for frame in frames:
            store.append(frame)


def _steps(path, count: int, shape=(4, 4)) -> list[np.ndarray]:
    fractions = [np.full(shape, index / 8, dtype=float) for index in range(count)]
    with RetrievalStepStore.capture(path, frame_shape=shape) as store:
        for fraction in fractions:
            store.append(fraction)
    return fractions


# --- A capture ---------------------------------------------------------------------


def test_a_capture_reads_back_frame_for_frame(tmp_path) -> None:
    path = tmp_path / "capture.asdf"
    frames = _frames(4)
    _capture(path, frames)

    with CaptureStore.open(path) as store:
        assert len(store) == 4
        for index, frame in enumerate(frames):
            assert np.allclose(store.read(index)["camera_image"], frame, atol=1e-3)


def test_a_capture_holds_the_levels_that_produced_each_frame(tmp_path) -> None:
    path = tmp_path / "capture.asdf"
    frames, levels = _frames(3), _levels(3)
    _capture(path, frames, levels)

    with CaptureStore.open(path) as store:
        sample = store.read(1)
        assert sorted(sample) == ["camera_image", "slm_levels"]
        assert np.array_equal(sample["slm_levels"], levels[1])
        assert np.allclose(sample["camera_image"], frames[1], atol=1e-3)


def test_the_record_travels_in_the_same_file(tmp_path) -> None:
    """One path is the whole dataset: the frames and what describes them."""
    path = tmp_path / "capture.asdf"
    _capture(path, _frames(2))

    with CaptureStore.open(path) as store:
        record = store.record()

    assert isinstance(record, _Capture)
    assert record.note == "a capture"


def test_length_comes_from_the_file(tmp_path) -> None:
    """Not from a stored count, so a file cannot disagree with itself."""
    path = tmp_path / "capture.asdf"
    _capture(path, _frames(5))

    with CaptureStore.open(path) as store:
        assert len(store) == 5


def test_frames_are_stored_narrow(tmp_path) -> None:
    """A camera hands back float64, which is twice the width the data justifies."""
    path = tmp_path / "capture.asdf"
    _capture(path, [np.asarray(frame, dtype=np.float64) for frame in _frames(2)])

    with CaptureStore.open(path) as store:
        assert store.read(0)["camera_image"].dtype == SAMPLE_DTYPE


@pytest.mark.parametrize("bitdepth,dtype", [(8, np.uint8), (12, np.uint16)])
def test_levels_keep_their_integer_width(tmp_path, bitdepth: int, dtype) -> None:
    """What the panel held, at a quarter of float32's width, read back unchanged."""
    path = tmp_path / "capture.asdf"
    levels = _levels(2, bitdepth=bitdepth)
    _capture(path, _frames(2), levels, bitdepth=bitdepth)

    with CaptureStore.open(path) as store:
        assert store.phase_bitdepth == bitdepth
        assert store.read(1)["slm_levels"].dtype == dtype
        assert np.array_equal(store.read(1)["slm_levels"], levels[1])


def test_write_takes_a_finished_capture(tmp_path) -> None:
    """For a set that is only complete at the end, like a feedback run."""
    path = tmp_path / "run.asdf"
    frames, levels = _frames(3), _levels(3)
    CaptureStore.write(
        path,
        _Capture(note="a run"),
        camera_images=frames,
        slm_levels=levels,
        phase_bitdepth=8,
    )

    with CaptureStore.open(path) as store:
        assert len(store) == 3
        assert store.record().note == "a run"
        assert np.allclose(store.read(2)["camera_image"], frames[2], atol=1e-3)
        assert np.array_equal(store.read(2)["slm_levels"], levels[2])


# --- A retrieval's steps -----------------------------------------------------------


def test_the_steps_read_back_as_the_search_parameter(tmp_path) -> None:
    path = tmp_path / "retrieval_steps.asdf"
    fractions = _steps(path, 3)

    with RetrievalStepStore.open(path) as store:
        assert len(store) == 3
        sample = store.read(1)
        assert set(sample) == {"slm_fraction"}
        assert np.allclose(sample["slm_fraction"], fractions[1], atol=1e-6)


def test_the_steps_are_continuous(tmp_path) -> None:
    """The optimiser's own parameter, not a quantised level: rounding it would lose
    where the search had got to."""
    path = tmp_path / "retrieval_steps.asdf"
    with RetrievalStepStore.capture(path, frame_shape=(4, 4)) as store:
        store.append(np.full((4, 4), 0.3125))

    with RetrievalStepStore.open(path) as store:
        assert store.read(0)["slm_fraction"].dtype == SAMPLE_DTYPE
        assert np.allclose(store.read(0)["slm_fraction"], 0.3125)


def test_the_steps_carry_no_record(tmp_path) -> None:
    """What describes a retrieval is its result, which is saved separately."""
    path = tmp_path / "retrieval_steps.asdf"
    _steps(path, 2)

    with RetrievalStepStore.open(path) as store:
        assert not hasattr(store, "record")


# --- Survival ----------------------------------------------------------------------


def test_a_capture_cut_short_keeps_its_frames(tmp_path) -> None:
    """A bench capture does get interrupted, and the frames taken so far are the point
    of streaming rather than buffering."""
    path = tmp_path / "capture.asdf"
    frames = _frames(6)
    _capture(path, frames, _levels(6))

    # Cut mid-frame, which is the unkind case: the partial frame is discarded and the
    # whole ones remain.
    size = path.stat().st_size
    frame_bytes = frames[0].size * np.dtype(SAMPLE_DTYPE).itemsize
    with open(path, "r+b") as handle:
        handle.truncate(size - frame_bytes - frame_bytes // 3)

    with CaptureStore.open(path) as store:
        assert len(store) == 4
        assert np.allclose(store.read(3)["camera_image"], frames[3], atol=1e-3)


def test_a_moved_file_still_reads(tmp_path) -> None:
    """Nothing in the file points at where it lives."""
    original = tmp_path / "here" / "capture.asdf"
    _capture(original, _frames(2))

    moved = tmp_path / "elsewhere.asdf"
    original.rename(moved)

    with CaptureStore.open(moved) as store:
        assert len(store) == 2
        assert store.record().note == "a capture"


def test_the_header_says_what_is_inside(tmp_path) -> None:
    path = tmp_path / "capture.asdf"
    _capture(path, _frames(2))
    header = path.read_bytes()[:3000].decode("latin-1")

    assert header.startswith("#ASDF")
    assert "camera_images" in header
    assert "slm_levels" in header


def test_reading_one_sample_does_not_read_the_rest(tmp_path) -> None:
    """Lazily loaded and memory-mapped, which is what a training loop over a set larger
    than memory needs.

    Asserted through the store's own reader rather than a hand-rolled asdf.open, which
    would pin the file format and let the reader quietly load the lot.
    """
    path = tmp_path / "capture.asdf"
    _capture(path, _frames(8, shape=(64, 64)), _levels(8))

    with CaptureStore.open(path) as store:
        images = store._series("camera_image")
        assert type(images).__name__ == "NDArrayType"
        assert images.shape == (8, 64, 64)


def test_a_sample_outlives_the_store(tmp_path) -> None:
    """The arrays are read on demand, so what read() hands back has to be a copy rather
    than a view into a file the caller is about to close."""
    path = tmp_path / "capture.asdf"
    _capture(path, _frames(3))

    with CaptureStore.open(path) as store:
        sample = store.read(1)
        record = store.record()

    assert np.isfinite(sample["camera_image"]).all()
    assert isinstance(record, _Capture)


def test_length_while_capturing_counts_what_was_appended(tmp_path) -> None:
    """Asking how far a capture has got is the obvious thing to do during one, and it
    used to raise because the count was only ever read back off the file."""
    store = CaptureStore.capture(
        tmp_path / "capture.asdf",
        _Capture(),
        frame_shape=(6, 8),
        slm_levels=_levels(2),
        phase_bitdepth=8,
    )
    try:
        assert len(store) == 0
        store.append(np.zeros((6, 8)))
        store.append(np.ones((6, 8)))
        assert len(store) == 2
    finally:
        store.close()


def test_a_store_open_for_writing_cannot_be_read(tmp_path) -> None:
    store = CaptureStore.capture(
        tmp_path / "capture.asdf",
        _Capture(),
        frame_shape=(4, 4),
        slm_levels=_levels(1),
        phase_bitdepth=8,
    )
    try:
        with pytest.raises(RuntimeError, match="open for writing"):
            store.read(0)
    finally:
        store.close()


def test_appending_to_a_reader_raises(tmp_path) -> None:
    path = tmp_path / "capture.asdf"
    _capture(path, _frames(2))

    with CaptureStore.open(path) as store:
        with pytest.raises(RuntimeError, match="not open for appending"):
            store.append(np.zeros((6, 8)))


# --- The torch dataset -------------------------------------------------------------


def test_dataset_length_and_indexing(tmp_path) -> None:
    path = tmp_path / "capture.asdf"
    frames = _frames(3)
    _capture(path, frames)

    with CaptureStore.open(path) as store:
        dataset = SampleDataset(store)
        assert len(dataset) == 3
        assert np.allclose(dataset[2]["camera_image"], frames[2], atol=1e-3)
        # A DataLoader hands the index over as a tensor.
        assert np.allclose(
            dataset[torch.tensor(1)]["camera_image"], frames[1], atol=1e-3
        )


def test_dataset_reads_a_retrieval_too(tmp_path) -> None:
    """It only asks a store for its length and its samples, so either kind will do."""
    path = tmp_path / "retrieval_steps.asdf"
    fractions = _steps(path, 2)

    with RetrievalStepStore.open(path) as store:
        dataset = SampleDataset(store)
        assert len(dataset) == 2
        assert np.allclose(dataset[1]["slm_fraction"], fractions[1], atol=1e-6)


def test_dataset_applies_the_transform(tmp_path) -> None:
    path = tmp_path / "capture.asdf"
    _capture(path, _frames(2))

    with CaptureStore.open(path) as store:
        dataset = SampleDataset(
            store,
            transform=lambda sample: {
                **sample, "camera_image": sample["camera_image"] * 0
            },
        )
        assert not dataset[0]["camera_image"].any()


def test_dataset_caches_when_asked(tmp_path) -> None:
    """Rereading every epoch is what the cache exists to avoid, so it has to be the same
    object rather than an equal one."""
    path = tmp_path / "capture.asdf"
    _capture(path, _frames(1))

    with CaptureStore.open(path) as store:
        cached = SampleDataset(store, cache=True)
        assert cached[0] is cached[0]

        uncached = SampleDataset(store, cache=False)
        assert uncached[0] is not uncached[0]
