"""Every hand-written ``adjoint()`` really is the conjugate transpose.

``adjoint`` is used by algorithms that do not go through autograd, so no
gradient test can reach it: ``gradcheck`` exercises the *backward* that autograd
derives from ``forward``, never the method written by hand. The check that does
reach it is the defining identity of the adjoint,

    <A x, y> == <x, A* y>

with the complex inner product ``<a, b> = sum(conj(a) * b)``. ``x`` lives on the
module's input grid and ``y`` on its output grid, which for a resampling
propagator are different.

This is a value test, not a gradient test: it deliberately says nothing about
the conjugation handling inside ``__torch_dispatch__``, because the modules that
call ``.conj()`` do so on plain tensors, where the dispatcher materialises the
conjugate bit for them.
"""

from __future__ import annotations

import pytest
import torch

from hologradpy.optics.complex_amplitude import ComplexAmplitude
from hologradpy.optics.modules.diagonal_elements import (
    DoubletLens,
    SimpleLens,
    ZernikePhase,
)
from hologradpy.optics.modules.slm_fields import PixelwiseSLMField
from hologradpy.optics.modules.geometric_transforms import GeometricWarp
from hologradpy.optics.modules.hardware_models.pointing_instability import (
    PointingInstability,
)
from hologradpy.optics.modules.hardware_models.power_instability import (
    PowerInstability,
)
from hologradpy.optics.modules.propagators import (
    FourierLensCZT,
    FourierLensFFT,
    FourierLensNUFFT,
)
from hologradpy.optics.modules.propagators.angular_spectrum_method import (
    AngularSpectrumMethod,
)
from hologradpy.optics.modules.virtual_slms.abstract import VirtualSLM

pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")

RESOLUTION: tuple[int, int] = (8, 8)
PIXEL_SIZE: tuple[float, float] = (10e-6, 10e-6)
WAVELENGTH = torch.tensor(800e-9, dtype=torch.float64)

EXACT_TOLERANCE = 1e-9

ADJOINT_FACTORIES: dict[str, callable] = {
    "AngularSpectrum": lambda: AngularSpectrumMethod(propagation_distance=1e-3),
    "DoubletLens": lambda: DoubletLens(
        refractive_index_flint=1.6,
        refractive_index_crown=1.5,
        radius_crown=0.1,
        radius_crown_flint=-0.1,
        radius_flint=-0.5,
    ),
    "FourierLensCZT": lambda: FourierLensCZT(
        focal_length=0.1,
        resolution_out=RESOLUTION,
        pixel_size_out=(5e-6, 5e-6),
    ),
    # Padding puts a zero pad in the forward and its transpose, a crop, in the adjoint.
    # Get either the offset or the pairing wrong and the two stop being transposes.
    "FourierLensCZTPadded": lambda: FourierLensCZT(
        focal_length=0.1,
        resolution_out=RESOLUTION,
        pixel_size_out=(5e-6, 5e-6),
        angle=7.0,
        padded_resolution=(12, 12),
    ),
    "FourierLensFFT": lambda: FourierLensFFT(focal_length=0.1),
    "FourierLensNUFFT": lambda: FourierLensNUFFT(
        focal_length=0.1,
        resolution_out=RESOLUTION,
        pixel_size_out=(5e-6, 5e-6),
    ),
    "PointingInstability": lambda: PointingInstability(
        tilt_std=(1e-4, 1e-4), seed=0
    ),
    "PowerInstability": lambda: PowerInstability(power_std=0.0, seed=0),
    "SimpleLens": lambda: SimpleLens(focal_length=0.1, aperture_radius=1e-3),
    "PixelwiseSLMField": lambda: PixelwiseSLMField(),
    "ZernikePhase": lambda: ZernikePhase(
        number_of_radial_orders=3,
        initial_coefficients=torch.linspace(0.1, 1.0, 6),
    ),
}

TOLERANCES: dict[str, float] = {}

# Modules that deliberately provide no adjoint. Listed so that adding one is a
# conscious act and so the gap is visible: an adjoint-based (non-autograd)
# solver cannot include these.
WITHOUT_ADJOINT: dict[str, callable] = {
    "GeometricWarp": lambda: GeometricWarp(
        resolution_out=RESOLUTION, pixel_size_out=PIXEL_SIZE
    ),
    "VirtualSLM": lambda: VirtualSLM(phase_scaling=1.0),
}


@pytest.fixture(autouse=True)
def double_precision():
    """Float64 throughout, so the tolerances measure the module and not the
    accumulation of float32 rounding.
    """
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        yield
    finally:
        torch.set_default_dtype(previous)


def _random_field(
    resolution: tuple[int, int],
    pixel_size: tuple[float, float],
    seed: int,
) -> ComplexAmplitude:
    generator = torch.Generator().manual_seed(seed)
    real = torch.randn(resolution, generator=generator)
    imag = torch.randn(resolution, generator=generator)
    return ComplexAmplitude(real + 1j * imag, WAVELENGTH, pixel_size)


def _inner_product(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    return torch.sum(left.conj() * right)


@pytest.mark.parametrize("name", sorted(ADJOINT_FACTORIES))
def test_adjoint_is_the_conjugate_transpose(name: str) -> None:
    module = ADJOINT_FACTORIES[name]()

    field = _random_field(RESOLUTION, PIXEL_SIZE, seed=0)
    # The first call also initializes the module lazily and fixes its output
    # geometry. Modules that sample per frame (the instability models) reapply
    # whatever the most recent forward drew, so the adjoint must follow it.
    forward_field = module(field)
    output_resolution = tuple(forward_field.resolution)
    output_pixel_size = tuple(forward_field.pixel_size[0].tolist())

    probe = _random_field(output_resolution, output_pixel_size, seed=1)
    adjoint_probe = module.adjoint(probe)

    left = _inner_product(forward_field.as_tensor(), probe.as_tensor())
    right = _inner_product(field.as_tensor(), adjoint_probe.as_tensor())

    scale = max(abs(complex(left)), abs(complex(right)))
    assert scale > 0.0, "degenerate probe: both inner products vanished"
    relative_error = abs(complex(left) - complex(right)) / scale

    assert relative_error < TOLERANCES.get(name, EXACT_TOLERANCE), (
        f"{name}: <Ax, y> = {complex(left)} but <x, A*y> = {complex(right)} "
        f"(relative error {relative_error:.3e}, ratio {complex(left / right)})"
    )


@pytest.mark.parametrize("name", sorted(WITHOUT_ADJOINT))
def test_modules_without_an_adjoint_say_so(name: str) -> None:
    """These raise rather than silently returning something that is not an
    adjoint.
    """
    module = WITHOUT_ADJOINT[name]()
    field = _random_field(RESOLUTION, PIXEL_SIZE, seed=0)
    module(field)

    with pytest.raises(NotImplementedError):
        module.adjoint(field)
