"""The lazy-init subclassing contract for OpticsModule.

A subclass just implements forward/adjoint (and optionally lazy_init to build
state); initialization is automatic and symmetric -- forward inits from its input,
adjoint requires a prior forward / initialize_from_geometry. Output geometry
defaults to the input; set_output_geometry() changes it. No subclass writes an
init/guard by hand.
"""

import pytest
import torch

from hologradpy.propagation.complex_amplitude import ComplexAmplitude, FieldGeometry
from hologradpy.propagation.optics_module import OpticsModule

GEOMETRY = FieldGeometry(
    resolution=(4, 4),
    pixel_size=torch.tensor([1e-5, 1e-5]),
    wavelength=torch.tensor(0.5e-6),
)


def _field() -> ComplexAmplitude:
    return ComplexAmplitude(
        torch.ones(4, 4, dtype=torch.complex64),
        GEOMETRY.wavelength,
        GEOMETRY.pixel_size,
    )


class _Preserving(OpticsModule):
    """Sampling-preserving: overrides only lazy_init to build state -- no super()
    call, no geometry code, no init/guard in forward/adjoint."""

    def lazy_init(self, complex_amplitude: ComplexAmplitude) -> None:
        self.register_buffer("gain", torch.ones(()))

    def forward(self, complex_amplitude: ComplexAmplitude) -> ComplexAmplitude:
        return complex_amplitude

    def adjoint(self, complex_amplitude: ComplexAmplitude) -> ComplexAmplitude:
        return complex_amplitude


class _Downsample(OpticsModule):
    """Geometry-changing: declares output sampling via set_output_geometry()."""

    def lazy_init(self, complex_amplitude: ComplexAmplitude) -> None:
        self.set_output_geometry(resolution=(2, 2), pixel_size=(2e-5, 2e-5))

    def forward(self, complex_amplitude: ComplexAmplitude) -> ComplexAmplitude:
        data = complex_amplitude.as_tensor()[..., ::2, ::2]
        return ComplexAmplitude(data, complex_amplitude.wavelength, self.pixel_size_out)


def test_forward_auto_inits_without_manual_guard():
    module = _Preserving()
    assert not module.initialized
    out = module(_field())  # first forward triggers lazy_init automatically
    assert module.initialized
    assert hasattr(module, "gain")  # lazy_init built its state
    assert isinstance(out, ComplexAmplitude)


def test_sampling_preserving_output_equals_input():
    module = _Preserving()
    module(_field())
    assert module.resolution_out == (4, 4)
    torch.testing.assert_close(
        module.pixel_size_out.reshape(-1)[:2],
        GEOMETRY.pixel_size.reshape(-1)[:2].to(module.pixel_size_out.dtype),
    )


def test_adjoint_auto_inits_after_forward():
    module = _Preserving()
    module(_field())  # initialise via forward
    out = module.adjoint(_field())  # no manual _ensure_initialized needed
    assert isinstance(out, ComplexAmplitude)


def test_adjoint_before_any_forward_raises_clearly():
    module = _Preserving()
    with pytest.raises(RuntimeError, match="initialised"):
        module.adjoint(_field())


def test_set_output_geometry_changes_output_sampling():
    module = _Downsample()
    out = module(_field())
    assert module.resolution_out == (2, 2)
    assert out.resolution == (2, 2)
    torch.testing.assert_close(
        module.pixel_size_out.reshape(-1)[:2],
        torch.tensor([2e-5, 2e-5], dtype=module.pixel_size_out.dtype),
    )
