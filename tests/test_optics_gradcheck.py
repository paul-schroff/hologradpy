"""Gradient correctness for every OpticsModule, via ``torch.autograd.gradcheck``.

``gradcheck`` compares the analytic gradient against central finite differences.
For complex tensors it checks the conjugate Wirtinger derivative, which is the
convention ``.grad`` uses, so a dropped conjugation, a flipped sign, a severed
graph (gradient identically zero) and a NaN gradient are all first-class
failures. That makes it a far stronger check than the ``torch.isfinite(grad)``
assertions used elsewhere in this suite, which pass happily on a wildly wrong
gradient.

Two details of the setup are load-bearing, and getting either wrong silently
hides real bugs:

1. **The probe field is built by dispatch**, as
   ``ComplexAmplitude(constant) * exp(1j * phase)``, so the autograd graph lands
   on the *wrapper* and ``_data`` is detached. That is what happens in a real
   forward pass. Building the field directly from the leaf instead
   (``ComplexAmplitude(exp(1j * phase))``) puts the graph on ``_data``, takes the
   ``as_tensor`` fast path, and hides the whole severed-graph bug class.
2. **Ordered pairs of modules are checked, not just single modules.** Every
   module passes on its own. The defects live at the seam between two modules,
   because the two families move the autograd graph differently.

Everything runs in float64: ``gradcheck`` warns and fails on complex64 even when
the gradient is correct.
"""

from __future__ import annotations

import itertools

import pytest
import torch

from hologradpy.optics.complex_amplitude import ComplexAmplitude
from hologradpy.optics.modules.diagonal_elements import (
    SimpleLens,
    ZernikePhase,
)
from hologradpy.optics.modules.slm_fields import PixelwiseSLMField
from hologradpy.optics.modules.geometric_transforms import GeometricWarp
from hologradpy.optics.modules.propagators import (
    FourierLensCZT,
    FourierLensFFT,
    FourierLensNUFFT,
)
from hologradpy.optics.modules.propagators.angular_spectrum_method import (
    AngularSpectrumMethod,
)
from hologradpy.optics.modules.virtual_slms.abstract import VirtualSLM
from hologradpy.optics.modules.virtual_slms.zernike_slm import ZernikeSLM

pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")

RESOLUTION: tuple[int, int] = (8, 8)
PIXEL_SIZE: tuple[float, float] = (10e-6, 10e-6)
WAVELENGTH = torch.tensor(800e-9, dtype=torch.float64)

# gradcheck cost scales with the number of leaf elements, so the grid is kept
# small. fast_mode uses a random projection instead of the full Jacobian: it was
# verified to give an identical pass/fail verdict here at roughly 6x the speed.
GRADCHECK_KWARGS = dict(
    eps=1e-6,
    atol=1e-7,
    rtol=1e-5,
    check_grad_dtypes=True,
    check_undefined_grad=True,
    check_batched_grad=False,  # vmap is not implemented through this subclass
    fast_mode=True,
    nondet_tol=0.0,
)


@pytest.fixture(autouse=True)
def double_precision():
    """Run every test in this module in float64.

    The modules build their internal tensors at the default dtype, and
    ``gradcheck`` fails on complex64 inputs even when the gradient is exactly
    right (the finite-difference step is swamped by rounding).
    """
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        yield
    finally:
        torch.set_default_dtype(previous)


MODULE_FACTORIES: dict[str, callable] = {
    "AngularSpectrum": lambda: AngularSpectrumMethod(propagation_distance=1e-3),
    "FourierLensCZT": lambda: FourierLensCZT(
        focal_length=0.1,
        resolution_out=RESOLUTION,
        pixel_size_out=(5e-6, 5e-6),
    ),
    "FourierLensFFT": lambda: FourierLensFFT(focal_length=0.1),
    "FourierLensNUFFT": lambda: FourierLensNUFFT(
        focal_length=0.1,
        resolution_out=RESOLUTION,
        pixel_size_out=(5e-6, 5e-6),
    ),
    "GeometricWarp": lambda: GeometricWarp(
        resolution_out=RESOLUTION,
        pixel_size_out=PIXEL_SIZE,
        angle=5.0,
        shift=(1.0, 1.0),
    ),
    "SimpleLens": lambda: SimpleLens(focal_length=0.1, aperture_radius=1e-3),
    "PixelwiseSLMField": lambda: PixelwiseSLMField(),
    "VirtualSLM": lambda: VirtualSLM(phase_scaling=1.0),
    "ZernikePhase": lambda: ZernikePhase(
        number_of_radial_orders=3,
        initial_coefficients=torch.linspace(0.1, 1.0, 6),
    ),
    "ZernikeSLM": lambda: ZernikeSLM(
        phase_scaling=1.0,
        number_of_radial_orders=3,
        initial_coefficients=torch.linspace(0.1, 1.0, 6),
    ),
}

# The modules fall into two families that carry the autograd graph differently,
# and the ordered-pair cases below exist to cover the seam between them:
#
# * resampling modules (the two Fourier lenses that zoom, and the warp) take the
#   field out to a plain tensor through ``flatten_batch``, then rebuild a field
#   with ``unflatten_batch``;
# * every other module stays inside ``__torch_dispatch__``.
#
# A resampling module followed by a dispatch-based one used to lose the gradient
# entirely, because ``unflatten_batch`` built the new field straight from the
# constructor and ``_make_wrapper_subclass`` produces an autograd leaf.
# ``ComplexAmplitude.from_tensor`` now bridges that crossing.
RESAMPLING_MODULES = frozenset(
    {"FourierLensCZT", "FourierLensNUFFT", "GeometricWarp"}
)


def _constant_field_data() -> torch.Tensor:
    """A fixed, non-degenerate complex field to modulate the leaf phase onto."""
    generator = torch.Generator().manual_seed(0)
    amplitude = 0.6 + 0.4 * torch.rand(RESOLUTION, generator=generator)
    phase = 0.3 * torch.randn(RESOLUTION, generator=generator)
    return amplitude * torch.exp(1j * phase)


def _make_function(names: list[str]):
    """A gradcheck-able ``phase -> plain tensor`` closure over a module chain.

    ``gradcheck`` cannot take a ``ComplexAmplitude`` as an input (it flattens
    inputs to 1D, which the constructor rejects), so the leaf is a plain real
    phase, the field is built inside, and a plain tensor comes back out.
    """
    modules = [MODULE_FACTORIES[name]() for name in names]
    constant = _constant_field_data()

    def function(phase: torch.Tensor) -> torch.Tensor:
        field = ComplexAmplitude(constant, WAVELENGTH, PIXEL_SIZE)
        field = field * torch.exp(1j * phase)
        for module in modules:
            field = module(field)
        return field.as_tensor()

    return function


def _check(names: list[str]) -> None:
    function = _make_function(names)
    generator = torch.Generator().manual_seed(1)
    phase = (0.2 * torch.randn(RESOLUTION, generator=generator)).requires_grad_(True)

    # Run once so every lazily initialized module builds its parameters before
    # the finite differences start.
    function(phase)

    assert torch.autograd.gradcheck(function, (phase,), **GRADCHECK_KWARGS)


@pytest.mark.parametrize("name", sorted(MODULE_FACTORIES))
def test_single_module_gradient_is_correct(name: str) -> None:
    """Each module on its own differentiates correctly w.r.t. the input field."""
    _check([name])


def test_nufft_geometry_parameters_are_not_learnable() -> None:
    """The NUFFT lens cannot learn its own geometry, and that is upstream.

    torchkbnufft does not propagate a gradient to the k-space trajectory, so
    ``scale_factor`` / ``shift`` / ``angle`` stay at ``None`` even when the
    transform is rebuilt from the live parameters on every call. Pinned here
    because they are ordinary parameters: flipping ``requires_grad`` raises no
    error, it just silently yields nothing. The chirp-z lens is the contrast
    case and does receive gradients.
    """
    field = ComplexAmplitude(_constant_field_data(), WAVELENGTH, PIXEL_SIZE)

    lens = MODULE_FACTORIES["FourierLensNUFFT"]()
    lens(field)
    names = ("scale_factor", "shift", "angle")
    parameters = [getattr(lens, name) for name in names]
    for parameter in parameters:
        parameter.requires_grad_(True)
    # Rebuild from the live parameters, the fix one would reach for first.
    lens._transform = lens._build_transform(field)

    output = lens(field).as_tensor()
    if output.requires_grad:
        gradients = torch.autograd.grad(
            output.abs().pow(2).sum(), parameters, allow_unused=True
        )
        assert all(gradient is None for gradient in gradients)

    # The chirp-z lens, by contrast, does learn its geometry.
    chirp_z = MODULE_FACTORIES["FourierLensCZT"]()
    chirp_z(field)
    chirp_z_parameters = [getattr(chirp_z, name) for name in names]
    for parameter in chirp_z_parameters:
        parameter.requires_grad_(True)

    gradients = torch.autograd.grad(
        chirp_z(field).as_tensor().abs().pow(2).sum(),
        chirp_z_parameters,
        allow_unused=True,
    )
    assert all(gradient is not None for gradient in gradients)
    assert all(float(gradient.abs().sum()) > 0.0 for gradient in gradients)


def test_from_tensor_keeps_the_graph_across_a_dispatch_op() -> None:
    """``from_tensor`` connects a graph-carrying tensor to the field wrapper.

    The plain constructor cannot: ``_make_wrapper_subclass`` yields an autograd
    leaf, so a gradient flowing back through a ``__torch_dispatch__`` operation
    stops at the wrapper. This pins the difference so the bridge cannot quietly
    disappear.
    """
    generator = torch.Generator().manual_seed(2)
    phase_source = 0.3 * torch.randn(RESOLUTION, generator=generator)
    constant = _constant_field_data()

    def gradient(build) -> torch.Tensor | None:
        phase = phase_source.clone().requires_grad_(True)
        field = build(constant * torch.exp(1j * phase))
        # A dispatch-based module is the case the leaf defect breaks.
        out = MODULE_FACTORIES["SimpleLens"]()(field)
        value = out.as_tensor().abs().pow(2).sum()
        (result,) = torch.autograd.grad(value, [phase], allow_unused=True)
        return result

    bridged = gradient(
        lambda data: ComplexAmplitude.from_tensor(data, WAVELENGTH, PIXEL_SIZE)
    )
    assert bridged is not None
    assert torch.isfinite(bridged).all()
    assert bridged.abs().sum() > 0

    # The constructor is documented as leaving the field disconnected.
    plain = gradient(lambda data: ComplexAmplitude(data, WAVELENGTH, PIXEL_SIZE))
    assert plain is None


@pytest.mark.parametrize(
    "first, second", list(itertools.permutations(sorted(MODULE_FACTORIES), 2))
)
def test_module_pair_gradient_is_correct(first: str, second: str) -> None:
    """Gradients survive the seam between two chained modules.

    Single-module checks all pass, so this is the test that actually pins the
    graph-severing defects: they only appear once the field crosses between the
    two module families described at :data:`RESAMPLING_MODULES`.
    """
    _check([first, second])
