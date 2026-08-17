"""The record format: what survives being written, moved and reopened.

Records are the archival half of what this library writes. A saved calibration outlives
the code that wrote it, so the format identifies a type by a versioned tag URI rather
than by an import path, and the tests that matter here are the ones about drift: a class
that moves, a field that is added, removed or renamed, a file from an older version.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pytest
import torch

from hologradpy.calibration.camera_mapping import (
    CameraMapping,
    CameraMappingVisualizationData,
    FocalSpotFit,
)
from hologradpy.hardware.slm import SLMData
from hologradpy.optics.complex_amplitude import ComplexAmplitude
from hologradpy.phase_levels import LinearResponse
from hologradpy.roi import ROI
from hologradpy.serialization import (
    RECORD_TYPES,
    SaveableRecord,
    record_type,
    registered_as,
)


@record_type("test_shapes")
@dataclass
class Shapes(SaveableRecord):
    """The shapes that make a record awkward, rather than the easy ones."""

    when: datetime
    size: tuple[int, int]
    mask: np.ndarray
    field_: ComplexAmplitude
    tensor: torch.Tensor
    roi: ROI
    nested: list[dict]
    notes: str = "unset"


def _field() -> ComplexAmplitude:
    return ComplexAmplitude(
        torch.linspace(0, 1, 16).reshape(4, 4) + 0j,
        wavelength=torch.tensor(1.039e-6),
        pixel_size=torch.tensor([12.5e-6, 12.5e-6]),
        power=1e-3,
    )


def _shapes() -> Shapes:
    return Shapes(
        when=datetime(2026, 8, 16, 9, 30, 15, 123456),
        size=(900, 1440),
        mask=np.eye(6, dtype=bool),
        field_=_field(),
        tensor=torch.arange(12.0).reshape(3, 4),
        roi=ROI(10, 20, 30, 40),
        nested=[{"camera_image": "a.npy"}, {"phase_pattern": "b.npy"}],
    )


def _mapping() -> CameraMapping:
    return CameraMapping(
        timestamp=datetime.now(),
        name="test",
        transform=np.eye(3),
        detected_points=[(1.0, 2.0), (3.0, 4.0)],
        calculated_points=[(1.0, 2.0), (3.0, 4.0)],
        zeroth_order_position=(32.0, 32.0),
        spot_fit=FocalSpotFit(waist=2.5),
    )


# --- The round trip ----------------------------------------------------------------


def test_the_awkward_shapes_survive(tmp_path) -> None:
    path = tmp_path / "shapes.asdf"
    original = _shapes()
    original.save(path)
    reloaded = Shapes.load(path)

    assert reloaded.when == original.when
    # A tuple, not the list YAML would otherwise hand back.
    assert reloaded.size == (900, 1440)
    assert isinstance(reloaded.size, tuple)
    assert reloaded.mask.dtype == np.bool_
    assert np.array_equal(reloaded.mask, original.mask)
    assert torch.equal(reloaded.tensor, original.tensor)
    assert reloaded.roi == original.roi
    assert reloaded.nested == original.nested
    assert reloaded.notes == "unset"


def test_a_field_survives(tmp_path) -> None:
    """ComplexAmplitude cannot be rebuilt from its state, so it has a converter."""
    path = tmp_path / "field.asdf"
    original = _shapes()
    original.save(path)
    reloaded = Shapes.load(path)

    assert isinstance(reloaded.field_, ComplexAmplitude)
    assert torch.equal(reloaded.field_.as_tensor(), original.field_.as_tensor())
    assert torch.allclose(reloaded.field_.wavelength, original.field_.wavelength)
    assert torch.allclose(reloaded.field_.pixel_size, original.field_.pixel_size)
    # Power is the integral of the intensity, so the data carries it.
    assert torch.allclose(reloaded.field_.power(), original.field_.power())


def test_a_real_record_survives(tmp_path) -> None:
    """A nested one: the mapping holds a CameraData, which holds an ROI."""
    path = tmp_path / "mapping.asdf"
    mapping = _mapping()
    mapping.save(path)
    reloaded = CameraMapping.load(path)

    assert np.array_equal(reloaded.transform, mapping.transform)
    assert reloaded.zeroth_order_position == (32.0, 32.0)
    assert reloaded.detected_points == mapping.detected_points
    assert reloaded.rotation_degrees == pytest.approx(mapping.rotation_degrees)


def test_the_yaml_half_is_readable(tmp_path) -> None:
    """The point of the format: a header a human can read without this library."""
    path = tmp_path / "shapes.asdf"
    _shapes().save(path)
    header = path.read_bytes()[:2000].decode("latin-1")

    assert header.startswith("#ASDF")
    assert "asdf://hologradpy.org/tags/test_shapes-1.0.0" in header
    assert "asdf://hologradpy.org/tags/roi-1.0.0" in header


def test_loading_the_wrong_class_raises(tmp_path) -> None:
    path = tmp_path / "shapes.asdf"
    _shapes().save(path)
    with pytest.raises(TypeError, match="not a CameraMapping"):
        CameraMapping.load(path)


def test_an_unregistered_dataclass_is_refused(tmp_path) -> None:
    """Caught when writing, so a missing registration cannot produce a file that will
    not read back."""

    @dataclass
    class Unregistered:
        x: int

    @record_type("test_holder")
    @dataclass
    class Holder(SaveableRecord):
        payload: object

    with pytest.raises(Exception):
        Holder(Unregistered(1)).save(tmp_path / "bad.asdf")


def test_registering_a_plain_class_is_refused() -> None:
    """A record is written by walking its fields, so a class that has none registers
    happily and then fails at the first save, a long way from the registration."""
    with pytest.raises(TypeError, match="not a dataclass"):

        @record_type("test_plain")
        class Plain(SaveableRecord):
            pass


def test_a_snapshot_carrying_arrays_compares_and_hashes() -> None:
    """A device snapshot ends up in a dict or a set, and an array field answers neither
    question: == returns an array and hash() refuses. The array fields are therefore
    excluded from both, so two snapshots of the same device are equal."""
    response = LinearResponse(bitdepth=8, phase_scaling=1.0)
    correction = np.linspace(0, 1, 16).reshape(4, 4)

    def snapshot() -> SLMData:
        return SLMData(
            name="panel",
            resolution=(4, 4),
            pixel_size=(1e-5, 1e-5),
            wavelength=1.039e-6,
            settle_time_s=0.0,
            phase_response=response,
            phase_correction=correction.copy(),
        )

    assert snapshot() == snapshot()
    assert len({snapshot(), snapshot()}) == 1


# --- Drift -------------------------------------------------------------------------


def test_a_class_that_moved_still_loads(tmp_path) -> None:
    """The whole point. The tag on disk names the record, not the import path, so the
    class can move to another module between writing and reading."""
    path = tmp_path / "shapes.asdf"
    _shapes().save(path)

    @dataclass
    class MovedShapes(SaveableRecord):
        when: datetime
        size: tuple[int, int]
        mask: np.ndarray
        field_: ComplexAmplitude
        tensor: torch.Tensor
        roi: ROI
        nested: list[dict]
        notes: str = "unset"

    MovedShapes.RECORD_TYPE = "test_shapes"
    MovedShapes.RECORD_VERSION = 1

    with registered_as("test_shapes", MovedShapes):
        reloaded = MovedShapes.load(path)

    assert isinstance(reloaded, MovedShapes)
    assert reloaded.size == (900, 1440)


def test_pickle_would_not_survive_that_move(tmp_path) -> None:
    """The negative control, in a separate interpreter so the class really is absent.

    This is what the format replaced: pickle stores the import path, so moving a class
    orphans every file that embedded it.
    """
    home = tmp_path / "old_home.py"
    home.write_text(
        textwrap.dedent(
            """
            from dataclasses import dataclass

            @dataclass
            class Payload:
                value: int
            """
        ),
        encoding="utf-8",
    )
    pickled = tmp_path / "payload.pkl"

    write = textwrap.dedent(
        f"""
        import pickle, sys
        sys.path.insert(0, {str(tmp_path)!r})
        from old_home import Payload
        with open({str(pickled)!r}, "wb") as file:
            pickle.dump(Payload(7), file)
        """
    )
    assert subprocess.run([sys.executable, "-c", write]).returncode == 0

    home.unlink()  # the class moves away
    read = textwrap.dedent(
        f"""
        import pickle, sys
        sys.path.insert(0, {str(tmp_path)!r})
        with open({str(pickled)!r}, "rb") as file:
            pickle.load(file)
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", read], capture_output=True, text=True
    )
    assert result.returncode != 0
    assert "No module named 'old_home'" in result.stderr


def test_an_added_field_defaults_and_a_removed_one_is_ignored(tmp_path) -> None:
    """Fields drift without a version bump as long as the change is additive with a
    default, or a removal."""
    path = tmp_path / "drift.asdf"

    @record_type("test_drift")
    @dataclass
    class Before(SaveableRecord):
        kept: int
        removed: str

    Before(kept=7, removed="gone").save(path)

    @dataclass
    class After(SaveableRecord):
        kept: int
        added: float = 1.5

    After.RECORD_TYPE = "test_drift"
    After.RECORD_VERSION = 1

    with registered_as("test_drift", After):
        reloaded = After.load(path)

    assert reloaded.kept == 7
    assert reloaded.added == 1.5
    assert not hasattr(reloaded, "removed")


def test_a_renamed_field_needs_a_migration(tmp_path) -> None:
    """A rename is not additive, so it takes a version bump and a _migrate, and the
    error names the fix when there is not one."""
    path = tmp_path / "renamed.asdf"

    @record_type("test_rename")
    @dataclass
    class Version1(SaveableRecord):
        old_name: str

    Version1(old_name="value").save(path)

    @dataclass
    class Unmigrated(SaveableRecord):
        new_name: str

    Unmigrated.RECORD_TYPE = "test_rename"
    Unmigrated.RECORD_VERSION = 2

    with registered_as("test_rename", Unmigrated):
        with pytest.raises(TypeError, match="new_name.*requires|requires"):
            Unmigrated.load(path)

    @dataclass
    class Migrated(SaveableRecord):
        new_name: str

        @classmethod
        def _migrate(cls, version: int, stored: dict) -> dict:
            if version < 2:
                stored["new_name"] = stored.pop("old_name")
            return stored

    Migrated.RECORD_TYPE = "test_rename"
    Migrated.RECORD_VERSION = 2

    with registered_as("test_rename", Migrated):
        assert Migrated.load(path).new_name == "value"


def test_a_record_from_a_newer_version_says_so(tmp_path) -> None:
    path = tmp_path / "unknown.asdf"

    @record_type("test_transient")
    @dataclass
    class Transient(SaveableRecord):
        value: int

    Transient(1).save(path)
    del RECORD_TYPES["test_transient"]

    from hologradpy.serialization import install_extension

    install_extension()
    try:
        with pytest.raises(Exception, match="no class is registered|test_transient"):
            Transient.load(path)
    finally:
        RECORD_TYPES["test_transient"] = Transient
        install_extension()


# --- Size --------------------------------------------------------------------------


def test_a_mapping_leaves_its_frames_behind(tmp_path) -> None:
    """A mapping travels inside every dataset and every feedback run, and the frames it
    was fit from are about 10 MB on a megapixel camera.

    Every mapper puts them in visualization_data, so this holds whichever one produced
    the mapping rather than only for the coarse one.
    """
    mapping = _mapping()
    mapping.visualization_data = CameraMappingVisualizationData(
        camera_image=np.zeros((256, 256)),
        simulated_image=np.zeros((256, 256)),
    )

    lean = tmp_path / "lean.asdf"
    full = tmp_path / "full.asdf"
    mapping.lean().save(lean)
    mapping.save(full)

    assert CameraMapping.load(lean).visualization_data is None
    assert CameraMapping.load(full).visualization_data is not None
    # Two 256 x 256 float frames, so the difference is most of the file.
    assert lean.stat().st_size < 0.1 * full.stat().st_size
    # lean() must not consume the mapping it was called on.
    assert mapping.visualization_data is not None
