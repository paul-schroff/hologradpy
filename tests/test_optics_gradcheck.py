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

# FINUFFT spreads and interpolates across threads, so the order it sums in varies
# from run to run and the backward pass is not bit-reproducible. Measured over
# eight repeats in float64 the spread is 1e-17 absolute against a gradient of
# 2.6e-2 -- four ulps, i.e. summation order and nothing else. The tolerance below
# is far above that and far below any error worth catching, and it applies only
# to the chains that contain the NUFFT so every other module still has to be
# exactly reproducible.
NONDETERMINISTIC_MODULES = frozenset({"FourierLensNUFFT"})
NONDETERMINISM_TOLERANCE = 1e-12


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

    kwargs = dict(GRADCHECK_KWARGS)
    if NONDETERMINISTIC_MODULES.intersection(names):
        kwargs["nondet_tol"] = NONDETERMINISM_TOLERANCE

    assert torch.autograd.gradcheck(function, (phase,), **kwargs)


@pytest.mark.parametrize("name", sorted(MODULE_FACTORIES))
def test_single_module_gradient_is_correct(name: str) -> None:
    """Each module on its own differentiates correctly w.r.t. the input field."""
    _check([name])


def test_nufft_geometry_parameters_are_learnable() -> None:
    """The NUFFT lens learns its own focal-plane affine.

    FINUFFT differentiates with respect to its sample points, so a gradient runs
    all the way back through the trajectory to ``scale_factor`` / ``shift`` /
    ``angle``. Two things have to hold and both are pinned here: the gradient
    must exist at all (these are ordinary parameters, so a severed graph raises
    nothing -- it silently yields ``None``), and it must be the right value,
    which central differences supply.

    The finite differences need the transform at ``eps=1e-14``. At the ``1e-6``
    FINUFFT defaults to, the interpolation error is the same size as the change a
    small step produces and the difference measures noise -- which is a statement
    about the reference, not about the gradient, and the analytic values here are
    identical either way.

    The Kaiser-Bessel backend this replaced could not do any of it: its kernel
    interpolation was not differentiable in the trajectory, so all three stayed at
    ``None`` and the focal-plane affine could only be calibrated on the CZT lens.
    """
    field = ComplexAmplitude(_constant_field_data(), WAVELENGTH, PIXEL_SIZE)
    names = ("scale_factor", "shift", "angle")

    def build():
        lens = FourierLensNUFFT(
            focal_length=0.1,
            resolution_out=RESOLUTION,
            pixel_size_out=(5e-6, 5e-6),
            nufft_kwargs=dict(eps=1e-14),
        )
        lens(field)  # lazily initialize, so the parameters exist
        return lens

    def cost(lens):
        return lens(field).as_tensor().abs().pow(2).sum()

    lens = build()
    parameters = [getattr(lens, name) for name in names]
    for parameter in parameters:
        parameter.requires_grad_(True)
    analytic = torch.autograd.grad(cost(lens), parameters, allow_unused=True)

    assert all(gradient is not None for gradient in analytic)
    assert all(float(gradient.abs().sum()) > 0.0 for gradient in analytic)

    step = 1e-6
    for name, gradient in zip(names, analytic):
        for index in range(gradient.numel()):
            shifted = []
            for sign in (+1, -1):
                probe = build()
                with torch.no_grad():
                    getattr(probe, name).reshape(-1)[index] += sign * step
                shifted.append(float(cost(probe)))
            numeric = (shifted[0] - shifted[1]) / (2 * step)
            torch.testing.assert_close(
                float(gradient.reshape(-1)[index]),
                numeric,
                rtol=1e-4,
                atol=1e-12,
                msg=lambda message, parameter=name: (
                    f"the {parameter} gradient disagrees with central "
                    f"differences: {message}"
                ),
            )


def test_nufft_and_czt_agree_on_the_scale_and_shift_gradients() -> None:
    """Two independent evaluations of the same derivative, to seven digits.

    ``scale_factor`` and ``shift`` move the focal-plane sample points in exactly
    the same way for both lenses, so the exact chirp-z lens is a reference for the
    interpolating one that shares none of its machinery. ``angle`` is left out:
    the chirp-z lens rotates by shearing the padded field and the NUFFT rotates
    its trajectory, which are the same map only up to the shear's own resampling,
    so their derivatives in the angle are not the same number and never were.
    """
    field = ComplexAmplitude(_constant_field_data(), WAVELENGTH, PIXEL_SIZE)
    names = ("scale_factor", "shift")

    def geometry_gradient(lens):
        lens(field)  # lazily initialize, so the parameters exist
        parameters = [getattr(lens, name) for name in names]
        for parameter in parameters:
            parameter.requires_grad_(True)
        return torch.autograd.grad(
            lens(field).as_tensor().abs().pow(2).sum(),
            parameters,
            allow_unused=True,
        )

    from_nufft = geometry_gradient(MODULE_FACTORIES["FourierLensNUFFT"]())
    from_chirp_z = geometry_gradient(MODULE_FACTORIES["FourierLensCZT"]())

    for name, nufft, chirp_z in zip(names, from_nufft, from_chirp_z):
        torch.testing.assert_close(
            nufft,
            chirp_z,
            rtol=1e-5,
            atol=1e-12,
            msg=lambda message, parameter=name: (
                f"the {parameter} gradient disagrees with the exact chirp-z "
                f"lens: {message}"
            ),
        )


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
