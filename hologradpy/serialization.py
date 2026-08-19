"""Saving and loading of measurement records.

Three kinds of thing get written to disk in this library, and they are kept deliberately
separate:

* **Records** are measurement results and metadata: camera mappings, device snapshots,
  wavefront calibrations, retrieval results. They carry no learnable state and they are
  handled by :class:`SaveableRecord` below. A record may hold arrays, and some hold one
  per iteration of a run, but a full sensor frame per sample belongs in a dataset.
* **Checkpoints** are anything carrying learnable state, saved through torch by
  :meth:`hologradpy.optics.systems.OpticalSystem.save` and
  :meth:`hologradpy.optics.modules.OpticsModule.save`.
* **Datasets** are bulk per-sample arrays, written by
  :class:`hologradpy.datasets.CaptureStore` and
  :class:`hologradpy.datasets.RetrievalStepStore` into one self-describing file.

Only the record format is safe to open from someone else. A checkpoint is a pickle, read
back with ``weights_only=False`` because it stores the constructor arguments needed to
rebuild the object, so opening one runs whatever it contains.

A record is written as ASDF: a YAML tree naming its contents, followed by the arrays as
binary blocks. What identifies a type on disk is a versioned tag URI,
``asdf://hologradpy.org/tags/camera_mapping-1.0.0``.

A class joins the format by being a dataclass with decorator :func:`record_type`. Types
that are not dataclasses need a converter as implemented for torch tensors
and :class:`~hologradpy.optics.complex_amplitude.ComplexAmplitude`.
"""

from __future__ import annotations

import contextlib
import os
import typing
from collections.abc import Iterator
from dataclasses import MISSING, fields, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, TypeVar

import asdf
import numpy as np
import torch
from asdf.extension import Converter, Extension, SerializationContext

from .optics.complex_amplitude import ComplexAmplitude
from .phase_levels import LinearResponse, LookupResponse

NAMESPACE = "asdf://hologradpy.org"

# Every type that can be written, by the stable name it is written under.
RECORD_TYPES: dict[str, type] = {}

T = TypeVar("T")


def record_type(name: str, version: int = 1) -> Callable[[type[T]], type[T]]:
    """Register a class under a name that does not change when the class moves.

    Args:
        name: What the class is called on disk. Snake case by convention, and never
            changed once files carrying it exist.
        version: Bumped when the fields change in a way older files need help with, at
            which point the class also gains a :meth:`SaveableRecord._migrate`.

    Returns:
        Callable[[type[T]], type[T]]: A decorator that registers the class and hands
            it back.

    Raises:
        TypeError: when the class is not a dataclass. The converter writes a record
            by walking :func:`dataclasses.fields`, so a plain class registers without
            complaint and then fails at the first save.
    """

    def decorate(cls: type[T]) -> type[T]:
        if not is_dataclass(cls):
            raise TypeError(
                f"{cls.__name__} is not a dataclass, so it cannot be written as the "
                f"{name!r} record. Make it one, or give it a Converter as the torch "
                "tensor and ComplexAmplitude types have."
            )
        cls.RECORD_TYPE = name
        cls.RECORD_VERSION = version
        RECORD_TYPES[name] = cls
        install_extension()
        return cls

    return decorate


def _tag_for(name: str, version: int) -> str:
    return f"{NAMESPACE}/tags/{name}-{version}.0.0"


def _name_and_version(tag: str) -> tuple[str, int]:
    stem = tag.rsplit("/", 1)[-1]
    name, _, semver = stem.rpartition("-")
    return name, int(semver.split(".")[0])


class RecordConverter(Converter):
    """One converter for every registered dataclass.

    ASDF picks a converter by the object's type and writes the tag returned by
    :meth:`select_tag`, so a single converter serves all of them.
    """

    def __init__(self, record_types: dict[str, type]) -> None:
        self._record_types = record_types

    @property
    def tags(self) -> list[str]:
        # Every version of each name, not only the current one: a converter is consulted
        # only for the tags it claims, so claiming the current version alone would leave
        # an older file unrecognized and its migration unreachable.
        return [
            _tag_for(name, version)
            for name, cls in self._record_types.items()
            for version in range(1, cls.RECORD_VERSION + 1)
        ]

    @property
    def types(self) -> list[type]:
        return list(self._record_types.values())

    def select_tag(self, obj: Any, tags: list[str], ctx: SerializationContext) -> str:
        return _tag_for(type(obj).RECORD_TYPE, type(obj).RECORD_VERSION)

    def to_yaml_tree(self, obj: Any, tag: str, ctx: SerializationContext) -> dict:
        # Values pass straight through: ASDF walks them and tags anything it knows,
        # including nested records and numpy arrays.
        return {
            field.name: _to_yaml(getattr(obj, field.name)) for field in fields(obj)
        }

    def from_yaml_tree(self, node: dict, tag: str, ctx: SerializationContext) -> Any:
        name, version = _name_and_version(tag)
        if name not in RECORD_TYPES:
            raise TypeError(
                f"This file holds a {name!r} record, which no class is registered for. "
                "It was written by a newer version, or the registration was removed."
            )
        cls = RECORD_TYPES[name]
        stored = dict(node)

        if version < cls.RECORD_VERSION:
            stored = cls._migrate(version, stored)

        declared = {field.name: field for field in fields(cls)}
        required = {
            field.name
            for field in declared.values()
            if field.default is MISSING and field.default_factory is MISSING
        }
        missing = required - stored.keys()
        if missing:
            raise TypeError(
                f"A stored {name!r} is missing {sorted(missing)}, which "
                f"{cls.__name__} requires and has no default for. Add a _migrate for "
                f"version {version}."
            )

        # Unknown keys are dropped, so a field the class no longer declares is not an
        # error. Declared tuple fields are coerced back, since YAML has no tuple.
        return cls(
            **{
                key: _coerce(stored[key], declared[key].type)
                for key in declared.keys() & stored.keys()
            }
        )


class TensorConverter(Converter):
    """torch tensors, stored as the numpy array ASDF already handles."""

    tags: list[str] = [f"{NAMESPACE}/tags/torch_tensor-1.0.0"]
    types: list[type] = [torch.Tensor, torch.nn.Parameter]

    def to_yaml_tree(
        self, obj: torch.Tensor, tag: str, ctx: SerializationContext
    ) -> dict:
        return {"data": obj.detach().cpu().numpy()}

    def from_yaml_tree(
        self, node: dict, tag: str, ctx: SerializationContext
    ) -> torch.Tensor:
        return torch.from_numpy(np.asarray(node["data"]))


class ComplexAmplitudeConverter(Converter):
    tags: list[str] = [f"{NAMESPACE}/tags/complex_amplitude-1.0.0"]
    types: list[type] = [ComplexAmplitude]

    def to_yaml_tree(
        self, obj: ComplexAmplitude, tag: str, ctx: SerializationContext
    ) -> dict:
        # Power is not stored: it is the integral of the intensity, so the data carries
        # it already.
        return {
            "data": obj.as_tensor().detach().cpu().numpy(),
            "wavelength": obj.wavelength.detach().cpu().numpy(),
            "pixel_size": obj.pixel_size.detach().cpu().numpy(),
        }

    def from_yaml_tree(
        self, node: dict, tag: str, ctx: SerializationContext
    ) -> ComplexAmplitude:
        return ComplexAmplitude(
            torch.from_numpy(np.asarray(node["data"])),
            wavelength=torch.from_numpy(np.asarray(node["wavelength"])),
            pixel_size=torch.from_numpy(np.asarray(node["pixel_size"])),
        )


def _to_yaml(value: Any) -> Any:
    """Prepare a value for the YAML tree.

    Only two things need help: YAML has no tuple, so one would come back as a list and
    fail somewhere far away, and a datetime is not a scalar ASDF round-trips through a
    tag of its own.

    Args:
        value: The field value to write, walked into if it is a tuple, list or dict.

    Returns:
        Any: The value, with every tuple and datetime inside it wrapped in a dict
            carrying a marker key.
    """
    if isinstance(value, tuple):
        return {"__tuple__": [_to_yaml(item) for item in value]}
    if isinstance(value, datetime):
        return {"__datetime__": value.isoformat()}
    if isinstance(value, list):
        return [_to_yaml(item) for item in value]
    if isinstance(value, dict):
        return {key: _to_yaml(item) for key, item in value.items()}
    return value


def _from_yaml(value: Any) -> Any:
    if isinstance(value, dict):
        if set(value) == {"__tuple__"}:
            return tuple(_from_yaml(item) for item in value["__tuple__"])
        if set(value) == {"__datetime__"}:
            return datetime.fromisoformat(value["__datetime__"])
        return {key: _from_yaml(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_from_yaml(item) for item in value]
    return value


def _coerce(value: Any, declared_type: type | str) -> Any:
    """Restore a declared tuple field, which YAML flattened to a list.

    Args:
        value: The stored value, still in its YAML tree form.
        declared_type: The annotation the dataclass field carries, either the object
            itself or the string a postponed annotation leaves behind.

    Returns:
        Any: The value, as a tuple where the field declares one.
    """
    value = _from_yaml(value)
    if typing.get_origin(declared_type) is tuple and isinstance(value, list):
        return tuple(value)
    if isinstance(declared_type, str) and declared_type.startswith("tuple"):
        if isinstance(value, list):
            return tuple(value)
    return value


class RecordExtension(Extension):
    extension_uri = f"{NAMESPACE}/extensions/records-1.0.0"

    def __init__(self, record_types: dict[str, type]) -> None:
        self._record_converter = RecordConverter(record_types)

    @property
    def converters(self) -> list[Converter]:
        return [
            self._record_converter,
            TensorConverter(),
            ComplexAmplitudeConverter(),
        ]

    @property
    def tags(self) -> list[str]:
        return [tag for converter in self.converters for tag in converter.tags]


def install_extension() -> None:
    """Register every currently known record type with ASDF.

    Called again whenever the registry changes.
    """
    config = asdf.get_config()
    for existing in list(config.extensions):
        if getattr(existing, "extension_uri", None) == RecordExtension.extension_uri:
            config.remove_extension(existing.extension_uri)
    config.add_extension(RecordExtension(RECORD_TYPES))


@contextlib.contextmanager
def registered_as(name: str, cls: type) -> Iterator[None]:
    """Point a stable name at a different class for the duration of the block.

    For reading a file whose class has been superseded.

    Args:
        name: The name on disk to point elsewhere.
        cls: The class that name is read as inside the block.
    """
    previous = RECORD_TYPES.get(name)
    RECORD_TYPES[name] = cls
    install_extension()
    try:
        yield
    finally:
        if previous is None:
            del RECORD_TYPES[name]
        else:
            RECORD_TYPES[name] = previous
        install_extension()


def check_record_recognized(record: Any, source: str | os.PathLike) -> None:
    """Raise if ASDF handed back a raw tagged tree in place of a record.

    Args:
        record: Whatever came back from the tree.
        source: The file it came from, named in the error.

    Raises:
        TypeError: when no class is registered for the tag on disk.
    """
    tag = getattr(record, "_tag", None)
    if tag is not None:
        raise TypeError(
            f"{source} holds a {tag!r} record, which no class is registered for. "
            "It was written by a newer version of hologradpy, or by a class whose "
            "registration has since been removed."
        )


def attach_source_path(value: Any, path: str | os.PathLike) -> None:
    """Tell every record in ``value`` which file it came from.

    Args:
        value: A record, or a dataclass, list, tuple or dict holding records.
        path: The file the records were read from.
    """
    if isinstance(value, SaveableRecord):
        object.__setattr__(value, "_source_path", Path(path))
    if is_dataclass(value) and not isinstance(value, type):
        for entry in fields(value):
            attach_source_path(getattr(value, entry.name), path)
    elif isinstance(value, (list, tuple)):
        for item in value:
            attach_source_path(item, path)
    elif isinstance(value, dict):
        for item in value.values():
            attach_source_path(item, path)


class SaveableRecord:
    """Mixin giving a result dataclass a versioned, type-checked ``save``/``load``
    pair.

    Mix it into a dataclass, wear :func:`record_type`, and the pair comes for free::

        @record_type("camera_mapping")
        @dataclass(frozen=True)
        class CameraMapping(SaveableRecord):
            ...

        mapping.save("mapping.asdf")
        mapping = CameraMapping.load("mapping.asdf")
    """

    RECORD_TYPE: str
    RECORD_VERSION: int = 1

    _source_path: Path | None = None

    @property
    def source_directory(self) -> Path | None:
        """The directory this record was read from, or None if it was never read."""
        return None if self._source_path is None else Path(self._source_path).parent

    @classmethod
    def _migrate(cls, version: int, stored: dict) -> dict:
        """Bring an older record's fields up to this class's current shape.

        Overridden alongside a bump to ``version`` in :func:`record_type`. The default
        passes the fields through, which is right while only additions with defaults and
        removals have happened.

        Args:
            version: The version the stored fields were written at.
            stored: The fields as they were read from the file.

        Returns:
            dict: The fields in the shape this class's constructor takes.
        """
        return stored

    def save(self, filename: str | os.PathLike) -> None:
        """Write this record to ``filename``.

        Args:
            filename: The file to write.
        """
        asdf.AsdfFile({"record": self}).write_to(str(Path(filename)))

    @classmethod
    def load(cls: type[RecordType], filename: str | os.PathLike) -> RecordType:
        """Read a record of this class back from ``filename``.

        Args:
            filename: The file to read.

        Returns:
            RecordType: The record the file holds.

        Raises:
            TypeError: when the file holds a record of an unrelated class, or one
                that no class is registered for.
        """
        path = Path(filename)
        # memmap=False so the arrays outlive the closed file, which a small record can
        # afford and a caller expects.
        with asdf.open(str(path), lazy_load=False, memmap=False) as file:
            record = file["record"]

        check_record_recognized(record, path)

        if not isinstance(record, cls):
            raise TypeError(
                f"{path} holds a {type(record).__name__}, not a {cls.__name__}."
            )
        attach_source_path(record, path)
        return record


RecordType = TypeVar("RecordType", bound=SaveableRecord)


def register_dataclass(cls: type[T], name: str, version: int = 1) -> type[T]:
    """Register a nested dataclass that is not itself a record.

    ``ROI``, the visualization data and anything else that only ever travels inside a
    record. Written for classes that cannot wear the decorator because they are declared
    elsewhere. :func:`record_type` does the checking.

    Args:
        cls: The dataclass to register.
        name: What the class is called on disk.
        version: Bumped when the fields change in a way older files need help with.

    Returns:
        type[T]: The same class, now carrying its name and version.
    """
    return record_type(name, version)(cls)


register_dataclass(LinearResponse, "linear_phase_response")
register_dataclass(LookupResponse, "lookup_phase_response")
