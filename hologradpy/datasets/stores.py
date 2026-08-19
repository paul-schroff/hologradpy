"""The file a set is written to holds one ASDF tree and one streamed block. ASDF
allows a single streamed block and it must be last, so everything else, the record
included, is known before the first frame.
"""

from __future__ import annotations

import gc
import os
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any, BinaryIO, Sequence, TypedDict, TypeVar

import asdf
import numpy as np
from asdf.tags.core import NDArrayType
from numpy.typing import DTypeLike, NDArray

from ..phase_levels import level_dtype
from ..serialization import (
    SaveableRecord,
    attach_source_path,
    check_record_recognized,
)

SAMPLE_STORE_SUFFIX = ".asdf"

# What a sample can hold, and how each series is named in the tree.
_BLOCK_NAMES = {
    "camera_image": "camera_images",
    "slm_levels": "slm_levels",
    "slm_fraction": "slm_fractions",
}

SERIES = tuple(_BLOCK_NAMES)

SAMPLE_DTYPE = np.float32

# The depth the stored levels were quantized at.
PHASE_BITDEPTH_KEY = "phase_bitdepth"


class CapturedSample(TypedDict):
    """Sample captured from a camera and the SLM that produced it."""
    camera_image: NDArray[np.float32]
    slm_levels: NDArray


class RetrievalSample(TypedDict):
    """One pattern from partway through a phase retrieval."""
    slm_fraction: NDArray[np.float32]


class _SampleStore:
    """The file mechanics shared by every kind of set."""

    def __init__(
        self,
        path: str | os.PathLike,
        *,
        file: asdf.AsdfFile | None = None,
        handle: BinaryIO | None = None,
        frame_dtype: DTypeLike = SAMPLE_DTYPE,
    ) -> None:
        self.path = Path(path)
        self._file = file
        self._handle = handle
        self._frame_dtype = frame_dtype
        self._appended = 0

    @classmethod
    def _start(
        cls: type[StoreType],
        path: str | os.PathLike,
        tree: dict,
        *,
        streaming: str,
        frame_shape: tuple[int, int],
        frame_dtype: DTypeLike,
        **state: Any,
    ) -> StoreType:
        """Write everything but the streamed series, and hold the file open for it."""
        tree[_BLOCK_NAMES[streaming]] = asdf.Stream(list(frame_shape), frame_dtype)

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        handle = open(path, "wb")
        asdf.AsdfFile(tree).write_to(handle)
        return cls(path, handle=handle, frame_dtype=frame_dtype, **state)

    def append(self, frame: NDArray) -> int:
        """Append one frame of the streamed series, and return its index."""
        if self._handle is None:
            raise RuntimeError(
                f"{self.path} is not open for appending. Build one with capture() to "
                "stream into."
            )
        frame = np.ascontiguousarray(frame, dtype=self._frame_dtype)
        self._handle.write(frame.tobytes())
        # Flushed per frame, so a failed capture leaves the frames it took.
        self._handle.flush()
        self._appended += 1
        return self._appended - 1

    @classmethod
    def open(cls: type[StoreType], path: str | os.PathLike) -> StoreType:
        """Open a store for reading.

        Memory-mapped and lazily loaded, so opening a large set costs no more than
        reading its header and one sample costs one frame.
        """
        return cls(path, file=asdf.open(str(Path(path)), lazy_load=True, memmap=True))

    def __len__(self) -> int:
        """How many samples the file holds."""
        if self._handle is not None:
            return self._appended
        for name in SERIES:
            array = self._series(name)
            if array is not None:
                return int(array.shape[0])
        return 0

    def _read_series(self, names: Sequence[str]) -> dict:
        """The named series in full, still belonging to the open file."""
        return {name: self._series(name) for name in names}

    def _series(self, name: str) -> NDArrayType | None:
        return self._require_file().tree.get(_BLOCK_NAMES[name])

    def _require_file(self) -> asdf.AsdfFile:
        if self._file is None:
            raise RuntimeError(
                f"{self.path} is open for writing, not reading. Close it and reopen "
                "with open()."
            )
        return self._file

    def close(self) -> None:
        """Close the file, which a memory-mapped read holds open."""
        if self._handle is not None:
            self._handle.close()
            self._handle = None
        if self._file is not None:
            self._file.close()
            self._file = None
            # Closing is not enough as the mapping survives in a reference cycle.
            # Until the collector breaks it, Windows refuses to move or delete the file.
            gc.collect()

    def __enter__(self: StoreType) -> StoreType:
        return self

    def __exit__(self, *exception: object) -> None:
        self.close()


StoreType = TypeVar("StoreType", bound=_SampleStore)


class CaptureStore(_SampleStore):
    """Camera frames and the levels that produced them.

    The record describing the capture travels in the same file, so one path is the whole
    dataset.
    """

    def __init__(
        self, *args: Any, phase_bitdepth: int | None = None, **kwargs: Any
    ) -> None:
        super().__init__(*args, **kwargs)
        self._phase_bitdepth = phase_bitdepth

    @classmethod
    def capture(
        cls,
        path: str | os.PathLike,
        record: SaveableRecord,
        *,
        frame_shape: tuple[int, int],
        slm_levels: Sequence[NDArray],
        phase_bitdepth: int,
        frame_dtype: DTypeLike = SAMPLE_DTYPE,
    ) -> CaptureStore:
        """Open a store and write everything but the camera frames.

        Args:
            path: The file to write.
            record: What describes the capture. Written now, so it has to be complete
                before the first frame: anything the capture would learn late, an
                exposure or a region, is measured ahead of this call.
            frame_shape: ``(height, width)`` of one camera frame.
            slm_levels: The patterns to display, known up front because the tree is
                written before the first frame streams.
            phase_bitdepth: The depth those levels were quantized at.
            frame_dtype: What one frame is stored as. The default is wide enough for any
                camera this drives.

        Returns:
            CaptureStore: Open for appending. Close it, or use it as a context manager.
        """
        tree = _tree(record, phase_bitdepth)
        tree[_BLOCK_NAMES["slm_levels"]] = _stack(
            slm_levels, "slm_levels", phase_bitdepth
        )
        return cls._start(
            path,
            tree,
            streaming="camera_image",
            frame_shape=frame_shape,
            frame_dtype=frame_dtype,
            phase_bitdepth=phase_bitdepth,
        )

    @classmethod
    def write(
        cls,
        path: str | os.PathLike,
        record: SaveableRecord,
        *,
        camera_images: Sequence[NDArray],
        slm_levels: Sequence[NDArray],
        phase_bitdepth: int,
        frame_dtype: DTypeLike = SAMPLE_DTYPE,
    ) -> Path:
        """Write a complete capture in one call, for one that fits in memory."""
        tree = _tree(record, phase_bitdepth)
        tree[_BLOCK_NAMES["camera_image"]] = _stack(
            camera_images, "camera_image", phase_bitdepth, frame_dtype
        )
        tree[_BLOCK_NAMES["slm_levels"]] = _stack(
            slm_levels, "slm_levels", phase_bitdepth
        )

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        asdf.AsdfFile(tree).write_to(str(path))
        return path

    def read(self, index: int) -> CapturedSample:
        """One frame and the levels that produced it."""
        series = self._read_series(("camera_image", "slm_levels"))
        return {
            name: np.array(array[index])
            for name, array in series.items()
            if array is not None
        }

    @property
    def phase_bitdepth(self) -> int | None:
        """The depth the stored levels were quantized at, None when none was given."""
        if self._file is None:
            return self._phase_bitdepth
        return self._file.tree.get(PHASE_BITDEPTH_KEY)

    def record(self) -> SaveableRecord:
        """Description of this capture.

        Raises:
            TypeError: when the file holds a record no class is registered for.
        """
        record = self._require_file().tree.get("record")
        check_record_recognized(record, self.path)
        record = _realised(record)
        attach_source_path(record, self.path)
        return record


class RetrievalStepStore(_SampleStore):
    """The steps a phase retrieval took, every nth iteration."""

    @classmethod
    def capture(
        cls,
        path: str | os.PathLike,
        *,
        frame_shape: tuple[int, int],
        frame_dtype: DTypeLike = SAMPLE_DTYPE,
    ) -> RetrievalStepStore:
        """Open a store to stream the search's parameter into."""
        return cls._start(
            path,
            {},
            streaming="slm_fraction",
            frame_shape=frame_shape,
            frame_dtype=frame_dtype,
        )

    def read(self, index: int) -> RetrievalSample:
        return {"slm_fraction": np.array(self._series("slm_fraction")[index])}


def _realised(value: Any) -> Any:
    """Read a lazily loaded value in, so it outlives the file it came from. Only ASDF's
    own lazy array type is converted. Tensors and complex amplitudes are already read in
    by their converters, and everything else is left as it is.
    """
    if isinstance(value, NDArrayType):
        return np.array(value)
    if is_dataclass(value) and not isinstance(value, type):
        for entry in fields(value):
            # Set through object, since most records are frozen.
            object.__setattr__(value, entry.name, _realised(getattr(value, entry.name)))
        return value
    if isinstance(value, list):
        return [_realised(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_realised(item) for item in value)
    if isinstance(value, dict):
        return {key: _realised(item) for key, item in value.items()}
    return value


def _dtype_for(
    series: str, phase_bitdepth: int | None, frame_dtype: DTypeLike = SAMPLE_DTYPE
) -> DTypeLike:
    if series == "slm_levels":
        return level_dtype(phase_bitdepth or 8)
    return frame_dtype


def _tree(record: SaveableRecord | None, phase_bitdepth: int | None) -> dict:
    """The tree's non-sample entries, indicating what the samples are."""
    tree: dict = {} if record is None else {"record": record}
    if phase_bitdepth is not None:
        tree[PHASE_BITDEPTH_KEY] = int(phase_bitdepth)
    return tree


def _stack(
    arrays: Sequence[NDArray],
    series: str,
    phase_bitdepth: int | None,
    frame_dtype: DTypeLike = SAMPLE_DTYPE,
) -> NDArray:
    """The series as one array, in the width the samples justify."""
    return np.asarray(
        np.stack([np.asarray(a) for a in arrays]),
        dtype=_dtype_for(series, phase_bitdepth, frame_dtype),
    )
