"""Shared fixtures and registries for the OpticsModule test suite.

The canonical field layout is ``(*batch, component, wavelength, H, W)``. The component
axis is always at ``dim=-4`` and holds one value for a scalar field or three for a field
vector. The wavelength axis is always at ``dim=-3``, and everything before them is
batch. A plane ``(H, W)`` and a stack ``(n_wavelengths, H, W)`` are promoted at
construction, so every field has rank four or more.

The ND contract suite exercises each concrete :class:`OpticsModule` across a range of
ranks and component counts using the registries defined here.
"""

from __future__ import annotations

import torch

from hologradpy.optics.complex_amplitude import ComplexAmplitude
from hologradpy.optics.modules.propagators import FourierLensFFT
from hologradpy.optics.modules.propagators.fourier_lens_nufft import (
    FourierLensNUFFT,
)
from hologradpy.optics.modules.diagonal_elements import (
    SimpleLens,
    DoubletLens,
    ZernikePhase,
)
from hologradpy.optics.modules.slm_fields import PixelwiseSLMField
from hologradpy.optics.modules.geometric_transforms import (
    GeometricWarp,
)
from hologradpy.optics.modules.grid_adapter import GridAdapter
from hologradpy.optics.modules.pixel_crosstalk import (
    FreeKernelCrosstalk,
    NeighbourDifferenceCrosstalk,
    SuperGaussianCrosstalk,
)
from hologradpy.optics.modules.propagators.angular_spectrum_method import (
    AngularSpectrumMethod,
)
from hologradpy.optics.modules.virtual_slms.abstract import VirtualSLM
from hologradpy.optics.modules.virtual_slms.zernike_slm import ZernikeSLM


RESOLUTION: tuple[int, int] = (16, 16)
PIXEL_SIZE: tuple[float, float] = (10e-6, 10e-6)

# Zernike config used by the registered ZernikeSLM factory.
# ``number_of_radial_orders`` 4 gives 4*5/2 = 10 coefficients; a fixed 1D
# coefficient vector keeps fresh instances deterministic and broadcasts across
# any wavelength count.
ZERNIKE_RADIAL_ORDERS: int = 4
ZERNIKE_COEFFICIENTS = torch.linspace(0.1, 1.0, 10)


# Each factory returns a fresh, minimally-configured module so that lazy
# initialization state never leaks between test cases. All parameters are
# deterministic, so a single field processed alone must reproduce the
# corresponding slice of a batched forward.
MODULE_FACTORIES: dict[str, callable] = {
    "VirtualSLM": lambda: VirtualSLM(phase_scaling=1.0),
    "VirtualSLMFreeKernel": lambda: VirtualSLM(
        phase_scaling=1.0, pixel_crosstalk=FreeKernelCrosstalk(upscale_factor=4)
    ),
    "VirtualSLMSuperGaussian": lambda: VirtualSLM(
        phase_scaling=1.0, pixel_crosstalk=SuperGaussianCrosstalk(upscale_factor=2)
    ),
    "VirtualSLMNeighbour": lambda: VirtualSLM(
        phase_scaling=1.0,
        pixel_crosstalk=NeighbourDifferenceCrosstalk(upscale_factor=2),
        quantize=True,
    ),
    "GridAdapter": lambda: GridAdapter(factor=2),
    "GridAdapterIdentity": lambda: GridAdapter(factor=1),
    "FourierLensFFT": lambda: FourierLensFFT(focal_length=0.1),
    "FourierLensNUFFT": lambda: FourierLensNUFFT(
        focal_length=0.1,
        resolution_out=RESOLUTION,
        pixel_size_out=(5e-6, 5e-6),
    ),
    "PixelwiseSLMField": lambda: PixelwiseSLMField(),
    "GeometricWarp": lambda: GeometricWarp(
        resolution_out=RESOLUTION,
        pixel_size_out=PIXEL_SIZE,
        angle=5.0,
        shift=(2.0, 2.0),
    ),
    "ZernikeSLM": lambda: ZernikeSLM(
        phase_scaling=1.0,
        number_of_radial_orders=ZERNIKE_RADIAL_ORDERS,
        initial_coefficients=ZERNIKE_COEFFICIENTS,
    ),
    "SimpleLens": lambda: SimpleLens(focal_length=0.1, aperture_radius=1e-3),
    "DoubletLens": lambda: DoubletLens(
        refractive_index_flint=1.6,
        refractive_index_crown=1.5,
        radius_crown=0.1,
        radius_crown_flint=-0.1,
        radius_flint=-0.5,
    ),
    "AngularSpectrumMethod": lambda: AngularSpectrumMethod(
        propagation_distance=1e-3
    ),
    "ZernikePhase": lambda: ZernikePhase(
        number_of_radial_orders=ZERNIKE_RADIAL_ORDERS,
        initial_coefficients=ZERNIKE_COEFFICIENTS,
    ),
}


# label -> (data shape as constructed, number_of_wavelengths). Covers both promoted
# forms, a scalar and a vector field with no batch, a batched scalar, the
# singleton-wavelength batch that the old NUFFT squeeze() silently mangled, a batched
# vector, and a multi-axis batch.
RANK_CASES: dict[str, tuple[tuple[int, ...], int]] = {
    "2d": ((16, 16), 1),
    "3d": ((2, 16, 16), 2),
    "4d_scalar": ((1, 2, 16, 16), 2),
    "4d_vector": ((3, 2, 16, 16), 2),
    "5d": ((3, 1, 2, 16, 16), 2),
    "5d_single_wl": ((4, 1, 1, 16, 16), 1),
    "5d_vector": ((2, 3, 2, 16, 16), 2),
    "6d": ((2, 3, 1, 2, 16, 16), 2),
}

# The cases that carry a batch axis, so that a per-element comparison has something to
# iterate. The others describe a single field.
BATCHED_IDS: list[str] = ["5d", "5d_single_wl", "5d_vector", "6d"]

# The cases carrying a field vector, for the checks that components stay independent.
VECTOR_IDS: list[str] = ["4d_vector", "5d_vector"]


def make_wavelength(number_of_wavelengths: int) -> torch.Tensor:
    if number_of_wavelengths == 1:
        return torch.tensor(800e-9)
    return torch.linspace(800e-9, 900e-9, number_of_wavelengths)


def make_field(
    shape: tuple[int, ...],
    number_of_wavelengths: int,
    seed: int = 0,
) -> ComplexAmplitude:
    """Build a deterministic random ``ComplexAmplitude`` of the given shape."""
    generator = torch.Generator().manual_seed(seed)
    real = torch.rand(*shape, generator=generator)
    imag = torch.rand(*shape, generator=generator)
    data = (real + 1j * imag).to(torch.complex64)
    return ComplexAmplitude(data, make_wavelength(number_of_wavelengths), PIXEL_SIZE)
