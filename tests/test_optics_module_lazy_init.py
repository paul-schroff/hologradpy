"""The lazy-init subclassing contract for OpticsModule.

A subclass just implements forward/adjoint (and optionally lazy_init to build
state); initialization is automatic and symmetric -- forward inits from its input,
adjoint requires a prior forward / initialize_from_geometry. Output geometry
defaults to the input; set_output_geometry() changes it. No subclass writes an
init/guard by hand.
"""

import pytest
import torch

from hologradpy.optics.complex_amplitude import ComplexAmplitude, FieldGeometry
from hologradpy.optics.modules.abstract import OpticsModule

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
    call, no geometry code, no init/guard in forward/adjoint.
    """

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
    module(_field())  # initialize via forward
    out = module.adjoint(_field())  # no manual _ensure_initialized needed
    assert isinstance(out, ComplexAmplitude)


def test_adjoint_before_any_forward_raises_clearly():
    module = _Preserving()
    with pytest.raises(RuntimeError, match="initialized"):
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


# --- Checkpoints -------------------------------------------------------------------


def _lens():
    from hologradpy.optics.modules.propagators.fourier_lens_fft import FourierLensFFT

    return FourierLensFFT(focal_length=0.1, padded_resolution=(8, 8))


def _warp():
    from hologradpy.optics.modules.geometric_transforms import GeometricWarp

    return GeometricWarp(
        resolution_out=(4, 4),
        pixel_size_out=(1e-5, 1e-5),
        scale_factor=(1.1, 0.9),
        shift=(0.3, -0.2),
        angle=2.0,
    )


def _pixelwise():
    from hologradpy.optics.modules.slm_fields.pixelwise import PixelwiseSLMField

    return PixelwiseSLMField(
        init_field=ComplexAmplitude(
            torch.linspace(0.2, 1.0, 16).reshape(4, 4).to(torch.complex64),
            GEOMETRY.wavelength,
            GEOMETRY.pixel_size,
        )
    )


@pytest.mark.parametrize("build", [_lens, _warp, _pixelwise], ids=lambda f: f.__name__)
def test_a_module_reopens_as_what_it_was(build, tmp_path):
    """save() and from_file() are a pair, so a saved module comes back producing the
    same field. The base used to write a file its own from_file could not read.
    """
    module = build()
    before = module(_field())
    path = str(tmp_path / "module.pt")
    module.save(path)

    reopened = type(module).from_file(path)
    torch.testing.assert_close(
        reopened(_field()).as_tensor(), before.as_tensor(), rtol=1e-5, atol=1e-6
    )


def test_a_checkpoint_refuses_the_wrong_class(tmp_path):
    """One module's weights landing in another's parameters either fails on a shape a
    long way from here, or does not fail at all.
    """
    path = str(tmp_path / "lens.pt")
    lens = _lens()
    lens(_field())
    lens.save(path)

    with pytest.raises(TypeError, match="saved from a FourierLensFFT"):
        type(_warp()).from_file(path)

    with pytest.raises(TypeError, match="saved from a FourierLensFFT"):
        _warp().load_weights(path)
