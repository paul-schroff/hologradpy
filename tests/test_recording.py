"""Tests for the generalized RecordingMixin on OpticsModule / OpticalSystem.

A module opts in by overriding ``recordables()``; the base provides ``record`` /
``record_samples`` / ``history``. An OpticalSystem fans the toggle out to its layers and
aggregates their histories.
"""

import pytest
import torch

from hologradpy.optics.complex_amplitude import ComplexAmplitude, FieldGeometry
from hologradpy.optics.modules.abstract import OpticsModule
from hologradpy.optics.systems.abstract import OpticalSystem

pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")

GEOMETRY = FieldGeometry(
    resolution=(4, 4),
    pixel_size=torch.tensor([1e-5, 1e-5]),
    wavelength=torch.tensor(0.5e-6),
)


def _field() -> ComplexAmplitude:
    data = torch.ones(4, 4, dtype=torch.complex64)
    return ComplexAmplitude(data, GEOMETRY.wavelength, GEOMETRY.pixel_size)


class _Counter(OpticsModule):
    """Sampling-preserving module that records an incrementing counter."""

    def __init__(self) -> None:
        super().__init__()
        self._value = torch.tensor(0.0)

    def forward(self, complex_amplitude: ComplexAmplitude) -> ComplexAmplitude:
        self._value = self._value + 1
        return complex_amplitude

    def recordables(self) -> dict[str, torch.Tensor]:
        return {"n": self._value}


class _Identity(OpticsModule):
    """Sampling-preserving module that declares nothing to record."""

    def forward(self, complex_amplitude: ComplexAmplitude) -> ComplexAmplitude:
        return complex_amplitude


def test_records_declared_values_only_while_on():
    module = _Counter()
    # Not recording by default.
    module(_field())
    assert module.history == {}
    # record_samples records each forward's recordables.
    with module.record_samples():
        module(_field())
        module(_field())
    assert set(module.history) == {"n"}
    torch.testing.assert_close(module.history["n"], torch.tensor([2.0, 3.0]))
    # Recording is off again after the block.
    module(_field())
    assert module.history["n"].shape == (2,)


def test_record_toggle_clears_history_on_enable():
    module = _Counter()
    module.record()
    module(_field())
    module.record(False)
    assert module.history["n"].shape == (1,)
    # Re-enabling starts fresh.
    module.record()
    assert module.history == {}


def test_module_without_recordables_records_nothing():
    module = _Identity()
    with module.record_samples():
        module(_field())
    assert module.history == {}


def test_optical_system_fans_out_and_aggregates_by_layer():
    system = OpticalSystem(GEOMETRY, counter=_Counter(), identity=_Identity())
    with system.record_samples():
        system()
        system()
    history = system.history
    # Only the layer that declared recordables shows up, keyed by layer name.
    assert set(history) == {"counter"}
    torch.testing.assert_close(history["counter"]["n"], torch.tensor([1.0, 2.0]))
    # Turning recording off (via the context exit) stops accumulation.
    system()
    assert system.history["counter"]["n"].shape == (2,)
