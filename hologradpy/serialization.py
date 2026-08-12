"""Saving and loading of measurement records.

Three kinds of thing get written to disk in this library, and they are kept deliberately
separate:

* **Records** are immutable measurement results and metadata: camera mappings, device
  snapshots, wavefront calibrations, dataset manifests. They are small, they carry no
  learnable state, and they are handled by :class:`SaveableRecord` below.
* **Checkpoints** are anything carrying learnable state, saved through torch by
  :meth:`hologradpy.optics.systems.OpticalSystem.save` and
  :meth:`hologradpy.optics.modules.OpticsModule.save`.
* **Datasets** are bulk per-sample arrays, written as ``.npy`` files into a directory
  next to a record that acts as the manifest. The directory is supplied by the caller
  when loading and is deliberately *not* stored in the manifest, so a dataset stays
  readable after it is moved or copied.

A record is written inside a small envelope rather than on its own, so the file states
which class and which format version it holds. That turns the two ways this can go
wrong, a file from a newer format and a file holding the wrong type, into clear errors
instead of confusing attribute failures much later.
"""

from __future__ import annotations

import os
import pickle
from pathlib import Path
from typing import TypeVar

FORMAT_VERSION = 1


class SaveableRecord:
    """Mixin giving a result dataclass a versioned, type-checked ``save`` / ``load`` 
    pair.

    Mix it into a dataclass and the pair comes for free::

        @dataclass(frozen=True)
        class CameraMapping(SaveableRecord):
            ...

        mapping.save("mapping.pkl")
        mapping = CameraMapping.load("mapping.pkl")

    The underlying format is pickle, which means a saved record holds references to the
    classes it was built from and will not survive those classes being moved or renamed.
    That is an accepted trade for a research library: a record is a snapshot of one
    measurement, cheap to regenerate, and pickle keeps nested types (a
    ``ComplexAmplitude``, a ``CameraMapping``) intact without a hand-written schema for
    every field.
    """

    def save(self, filename: str | os.PathLike) -> None:
        """Write this record to ``filename``."""
        envelope = {
            "format_version": FORMAT_VERSION,
            "class_name": type(self).__name__,
            "record": self,
        }
        with open(Path(filename), "wb") as file:
            pickle.dump(envelope, file)

    @classmethod
    def load(cls: type[RecordType], filename: str | os.PathLike) -> RecordType:
        """Read a record of this class back from ``filename``.

        Raises:
            ValueError: the file was written by a different format version.
            TypeError: the file holds a record of an unrelated class.
        """
        path = Path(filename)
        with open(path, "rb") as file:
            envelope = pickle.load(file)

        version = envelope.get("format_version")
        if version != FORMAT_VERSION:
            raise ValueError(
                f"{path} was written with record format version {version}, but "
                f"this version of hologradpy reads version {FORMAT_VERSION}."
            )

        record = envelope["record"]
        if not isinstance(record, cls):
            raise TypeError(
                f"{path} holds a {type(record).__name__}, not a {cls.__name__}."
            )
        return record

RecordType = TypeVar("RecordType", bound=SaveableRecord)
