"""Saving and loading of measurement records.

Three kinds of thing get written to disk in this library, and they are kept deliberately
separate:

* **Records** are immutable measurement results and metadata: camera mappings, device
  snapshots, wavefront calibrations, retrieval results. They are small, they carry no
  learnable state, and they are handled by :class:`SaveableRecord` below.
* **Checkpoints** are anything carrying learnable state, saved through torch by
  :meth:`hologradpy.optics.systems.OpticalSystem.save` and
  :meth:`hologradpy.optics.modules.OpticsModule.save`.
* **Datasets** are bulk per-sample arrays, written by
  :class:`hologradpy.datasets.CaptureStore` and
  :class:`hologradpy.datasets.RetrievalStepStore` into one self-describing file.

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
from dataclasses import MISSING, fields, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import TypeVar

import asdf
import numpy as np
import torch
from asdf.extension import Converter, Extension

from .optics.complex_amplitude import ComplexAmplitude
from .phase_levels import LinearResponse, LookupResponse

NAMESPACE = "asdf://hologradpy.org"

# Every type that can be written, by the stable name it is written under.
RECORD_TYPES: dict[str, type] = {}


def record_type(name: str, version: int = 1):
    """Register a class under a name that does not change when the class moves.

    Args:
        name: What the class is called on disk. Snake case by convention, and never
            changed once files carrying it exist.
        version: Bumped when the fields change in a way older files need help with, at
            which point the class also gains a :meth:`SaveableRecord._migrate`.
    """

    def decorate(cls):
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
        # an older file unrecognised and its migration unreachable.
        return [
            _tag_for(name, version)
            for name, cls in self._record_types.items()
            for version in range(1, cls.RECORD_VERSION + 1)
        ]

    @property
    def types(self) -> list[type]:
        return list(self._record_types.values())

    def select_tag(self, obj, tags, ctx) -> str:
        return _tag_for(type(obj).RECORD_TYPE, type(obj).RECORD_VERSION)

    def to_yaml_tree(self, obj, tag, ctx) -> dict:
        # Values pass straight through: ASDF walks them and tags anything it knows,
        # including nested records and numpy arrays.
        return {
            field.name: _to_yaml(getattr(obj, field.name)) for field in fields(obj)
        }

    def from_yaml_tree(self, node, tag, ctx):
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

    tags = [f"{NAMESPACE}/tags/torch_tensor-1.0.0"]
    types = [torch.Tensor, torch.nn.Parameter]

    def to_yaml_tree(self, obj, tag, ctx) -> dict:
        return {"data": obj.detach().cpu().numpy()}

    def from_yaml_tree(self, node, tag, ctx):
        return torch.from_numpy(np.asarray(node["data"]))


class ComplexAmplitudeConverter(Converter):
    tags = [f"{NAMESPACE}/tags/complex_amplitude-1.0.0"]
    types = [ComplexAmplitude]

    def to_yaml_tree(self, obj, tag, ctx) -> dict:
        # Power is not stored: it is the integral of the intensity, so the data carries
        # it already.
        return {
            "data": obj.as_tensor().detach().cpu().numpy(),
            "wavelength": obj.wavelength.detach().cpu().numpy(),
            "pixel_size": obj.pixel_size.detach().cpu().numpy(),
        }

    def from_yaml_tree(self, node, tag, ctx):
        return ComplexAmplitude(
            torch.from_numpy(np.asarray(node["data"])),
            wavelength=torch.from_numpy(np.asarray(node["wavelength"])),
            pixel_size=torch.from_numpy(np.asarray(node["pixel_size"])),
        )


def _to_yaml(value):
    """Prepare a value for the YAML tree.

    Only two things need help: YAML has no tuple, so one would come back as a list and
    fail somewhere far away, and a datetime is not a scalar ASDF round-trips through a
    tag of its own.
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


def _from_yaml(value):
    if isinstance(value, dict):
        if set(value) == {"__tuple__"}:
            return tuple(_from_yaml(item) for item in value["__tuple__"])
        if set(value) == {"__datetime__"}:
            return datetime.fromisoformat(value["__datetime__"])
        return {key: _from_yaml(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_from_yaml(item) for item in value]
    return value


def _coerce(value, declared_type):
    """Restore a declared tuple field, which YAML flattened to a list."""
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
    def converters(self):
        return [
            self._record_converter,
            TensorConverter(),
            ComplexAmplitudeConverter(),
        ]

    @property
    def tags(self):
        return [
            *self._record_converter.tags,
            *TensorConverter.tags,
            *ComplexAmplitudeConverter.tags,
        ]


def install_extension() -> None:
    """Register every currently known record type with ASDF. Called again whenever the 
    registry changes.
    """
    config = asdf.get_config()
    for existing in list(config.extensions):
        if getattr(existing, "extension_uri", None) == RecordExtension.extension_uri:
            config.remove_extension(existing.extension_uri)
    config.add_extension(RecordExtension(RECORD_TYPES))


@contextlib.contextmanager
def registered_as(name: str, cls):
    """Point a stable name at a different class for the duration of the block. For 
    reading a file whose class has been superseded.
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


class SaveableRecord:
    """Mixin giving a result dataclass a versioned, type-checked ``save`` / ``load``
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

    @classmethod
    def _migrate(cls, version: int, stored: dict) -> dict:
        """Bring an older record's fields up to this class's current shape.

        Overridden alongside a bump to ``version`` in :func:`record_type`. The default
        passes the fields through, which is right while only additions with defaults and
        removals have happened.
        """
        return stored

    def save(self, filename: str | os.PathLike) -> None:
        """Write this record to ``filename``."""
        asdf.AsdfFile({"record": self}).write_to(str(Path(filename)))

    @classmethod
    def load(cls: type[RecordType], filename: str | os.PathLike) -> RecordType:
        """Read a record of this class back from ``filename``.

        Raises:
            TypeError: the file holds a record of an unrelated class.
        """
        path = Path(filename)
        # memmap=False so the arrays outlive the closed file, which a small record can
        # afford and a caller expects.
        with asdf.open(str(path), lazy_load=False, memmap=False) as file:
            record = file["record"]

        # An unrecognised tag is not an error to ASDF, which hands back the raw tree, so
        # the record that no class claims is caught here rather than surfacing as an
        # attribute failure much later.
        tag = getattr(record, "_tag", None)
        if tag is not None:
            raise TypeError(
                f"{path} holds a {tag!r} record, which no class is registered for. It "
                "was written by a newer version of hologradpy, or by a class whose "
                "registration has since been removed."
            )

        if not isinstance(record, cls):
            raise TypeError(
                f"{path} holds a {type(record).__name__}, not a {cls.__name__}."
            )
        return record


RecordType = TypeVar("RecordType", bound=SaveableRecord)


def register_dataclass(cls, name: str, version: int = 1):
    """Register a nested dataclass that is not itself a record.

    ``ROI``, the visualization data and anything else that only ever travels inside a
    record. Written for classes that cannot wear the decorator because they are declared
    elsewhere.
    """
    if not is_dataclass(cls):
        raise TypeError(f"{cls.__name__} is not a dataclass, so it needs a Converter.")
    return record_type(name, version)(cls)


register_dataclass(LinearResponse, "linear_phase_response")
register_dataclass(LookupResponse, "lookup_phase_response")
