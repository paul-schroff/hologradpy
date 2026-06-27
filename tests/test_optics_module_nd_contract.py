"""ND batch contract for concrete :class:`OpticsModule` implementations.

Every concrete ``OpticsModule`` must accept a field of arbitrary batch rank
``(*batch, wavelength, H, W)`` and:

* return a :class:`ComplexAmplitude`,
* preserve the input rank (no silent ``squeeze`` of singleton batch /
  wavelength axes),
* report output geometry consistent with ``resolution_out`` /
  ``pixel_size_out`` regardless of leading dimensions, and
* process each batch element independently.

These are parametrized across the module registry and a spread of ranks,
including the singleton-wavelength batch ``(B, 1, H, W)`` that previously broke
``FourierLensNUFFT``.
"""

from __future__ import annotations

import warnings

import pytest
import torch

from hologradpy.propagation.complex_amplitude import (
    ComplexAmplitude,
    FieldGeometry,
)

from .registry import MODULE_FACTORIES, RANK_CASES, make_field


# Lazy-init copy-constructs parameters from tensors, which torch warns about;
# it is orthogonal to what these tests assert.
pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")


MODULE_IDS = list(MODULE_FACTORIES)
RANK_IDS = list(RANK_CASES)


@pytest.mark.parametrize("module_name", MODULE_IDS)
@pytest.mark.parametrize("rank", RANK_IDS)
def test_returns_complex_amplitude(module_name: str, rank: str) -> None:
    shape, n_wavelengths = RANK_CASES[rank]
    field = make_field(shape, n_wavelengths)

    output = MODULE_FACTORIES[module_name]()(field)

    assert isinstance(output, ComplexAmplitude)


@pytest.mark.parametrize("module_name", MODULE_IDS)
@pytest.mark.parametrize("rank", RANK_IDS)
def test_rank_preserved(module_name: str, rank: str) -> None:
    """Output rank equals input rank: batch and wavelength axes survive even
    when singleton. This is the regression guard for the NUFFT squeeze bug."""
    shape, n_wavelengths = RANK_CASES[rank]
    field = make_field(shape, n_wavelengths)

    output = MODULE_FACTORIES[module_name]()(field)

    assert output.ndim == field.ndim
    # Leading batch dimensions are passed through unchanged.
    assert output.batch_shape == field.batch_shape


@pytest.mark.parametrize("module_name", MODULE_IDS)
@pytest.mark.parametrize("rank", RANK_IDS)
def test_output_geometry(module_name: str, rank: str) -> None:
    shape, n_wavelengths = RANK_CASES[rank]
    field = make_field(shape, n_wavelengths)

    module = MODULE_FACTORIES[module_name]()
    output = module(field)

    assert output.resolution == tuple(module.resolution_out)
    assert output.number_of_wavelengths == n_wavelengths
    torch.testing.assert_close(output.wavelength, field.wavelength)


@pytest.mark.parametrize("module_name", MODULE_IDS)
@pytest.mark.parametrize("rank", ["4d", "5d"])
def test_batch_independence(module_name: str, rank: str) -> None:
    """Each element of a batched forward equals that element processed alone.

    A fresh module instance is used for the single-element forward; because all
    registered modules are deterministic and lazy-init depends only on
    wavelength / resolution / pixel size (never on batch), the two paths must
    agree.
    """
    shape, n_wavelengths = RANK_CASES[rank]
    field = make_field(shape, n_wavelengths)

    batched_output = MODULE_FACTORIES[module_name]()(field)

    # Iterate the leading batch axis only; the remaining batch dims (if any)
    # stay attached to each element.
    for index in range(shape[0]):
        element = field[index]
        single_output = MODULE_FACTORIES[module_name]()(element)

        torch.testing.assert_close(
            batched_output._data[index],
            single_output._data,
            rtol=1e-4,
            atol=1e-4,
        )


@pytest.mark.parametrize("module_name", MODULE_IDS)
def test_lazy_init_lifecycle(module_name: str) -> None:
    module = MODULE_FACTORIES[module_name]()

    assert module.initialized is False
    with pytest.raises(ValueError):
        _ = module.input_geometry

    field = make_field(*RANK_CASES["3d"])
    module(field)

    assert module.initialized is True
    assert module.resolution_out is not None
    assert module.pixel_size_out is not None


@pytest.mark.parametrize("module_name", MODULE_IDS)
def test_forward_is_repeatable(module_name: str) -> None:
    """A second forward call (after the init hook is removed) still works and
    yields the same result as the first."""
    module = MODULE_FACTORIES[module_name]()
    field = make_field(*RANK_CASES["4d"])

    first = module(field)
    second = module(field)

    torch.testing.assert_close(first._data, second._data)


def test_nufft_singleton_wavelength_batch_not_squeezed() -> None:
    """Explicit regression test for the historical NUFFT ``squeeze`` bug:
    a ``(B, 1, H, W)`` batch must not collapse to ``(B, H, W)`` and then be
    misread as ``B`` wavelengths."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        field = make_field(*RANK_CASES["4d_single_wl"])
        output = MODULE_FACTORIES["FourierLensNUFFT"]()(field)

    assert output.ndim == 4
    assert output.shape[0] == field.shape[0]
    assert output.shape[1] == 1
    assert output.number_of_wavelengths == 1


# NUFFT output-plane pixel size; matches the registered FourierLensNUFFT
# factory in registry.py.
NUFFT_OUTPUT_PIXEL_SIZE = (5e-6, 5e-6)


def _output_plane_field(shape, n_wavelengths):
    """A complex field laid out on the NUFFT output grid."""
    wavelength = (
        torch.tensor(800e-9)
        if n_wavelengths == 1
        else torch.linspace(800e-9, 900e-9, n_wavelengths)
    )
    generator = torch.Generator().manual_seed(1)
    data = (
        torch.rand(*shape, generator=generator)
        + 1j * torch.rand(*shape, generator=generator)
    ).to(torch.complex64)
    return ComplexAmplitude(data, wavelength, NUFFT_OUTPUT_PIXEL_SIZE)


def test_nufft_adjoint_satisfies_inner_product_identity() -> None:
    """``adjoint`` is the true conjugate transpose of ``forward``:
    ``<A x, y> == <x, A* y>`` for input-plane x and output-plane y."""
    module = MODULE_FACTORIES["FourierLensNUFFT"]()
    x = make_field((2, 16, 16), 2)
    module(x)  # lazily initialise

    y = _output_plane_field((2, 16, 16), 2)

    forward_x = module(x)
    adjoint_y = module.adjoint(y)

    lhs = torch.sum(torch.conj(forward_x._data) * y._data)
    rhs = torch.sum(torch.conj(x._data) * adjoint_y._data)

    torch.testing.assert_close(lhs, rhs, rtol=1e-3, atol=1e-2)


@pytest.mark.parametrize("rank", RANK_IDS)
def test_nufft_adjoint_rank_and_geometry(rank: str) -> None:
    shape, n_wavelengths = RANK_CASES[rank]
    module = MODULE_FACTORIES["FourierLensNUFFT"]()
    module(make_field(shape, n_wavelengths))  # lazily initialise

    output_field = _output_plane_field(shape, n_wavelengths)
    restored = module.adjoint(output_field)

    # Adjoint lands in the input plane with the input rank preserved.
    assert restored.ndim == output_field.ndim
    assert restored.resolution == tuple(module.resolution_in)
    torch.testing.assert_close(restored.pixel_size, module.pixel_size_in)


def _input_plane_geometry(n_wavelengths):
    wavelength = (
        torch.tensor(800e-9)
        if n_wavelengths == 1
        else torch.linspace(800e-9, 900e-9, n_wavelengths)
    )
    return FieldGeometry(wavelength, torch.tensor([[10e-6, 10e-6]]), (16, 16))


def test_initialize_from_geometry_enables_adjoint_without_forward() -> None:
    """A module initialised from the input geometry can run ``adjoint``
    without any prior ``forward``, matching the forward-initialised path."""
    geometry = _input_plane_geometry(2)
    output_field = _output_plane_field((2, 16, 16), 2)

    forward_initialised = MODULE_FACTORIES["FourierLensNUFFT"]()
    forward_initialised(make_field((2, 16, 16), 2))
    expected = forward_initialised.adjoint(output_field)

    geometry_initialised = MODULE_FACTORIES["FourierLensNUFFT"]()
    assert geometry_initialised.initialized is False
    geometry_initialised.initialize_from_geometry(geometry)
    assert geometry_initialised.initialized is True

    restored = geometry_initialised.adjoint(output_field)
    torch.testing.assert_close(restored._data, expected._data)


def test_adjoint_before_initialization_raises() -> None:
    module = MODULE_FACTORIES["FourierLensNUFFT"]()
    output_field = _output_plane_field((2, 16, 16), 2)

    with pytest.raises(RuntimeError, match="must be initialised"):
        module.adjoint(output_field)


@pytest.mark.parametrize("module_name", MODULE_IDS)
def test_initialize_from_geometry_matches_forward_init(
    module_name: str,
) -> None:
    """initialize_from_geometry sets the same output geometry that a forward
    on the equivalent field would, for every module."""
    geometry = _input_plane_geometry(2)

    via_geometry = MODULE_FACTORIES[module_name]()
    via_geometry.initialize_from_geometry(geometry)

    via_forward = MODULE_FACTORIES[module_name]()
    via_forward(make_field((2, 16, 16), 2))

    assert via_geometry.resolution_out == via_forward.resolution_out
    torch.testing.assert_close(via_geometry.pixel_size_out, via_forward.pixel_size_out)


# Square propagators whose output plane matches the input plane (same
# resolution and pixel size), so an input-plane field doubles as a valid
# adjoint input.
SQUARE_PROPAGATORS = [
    "AngularSpectrumMethod",
    "SimpleLens",
    "DoubletLens",
    "ZernikePhase",
]


@pytest.mark.parametrize("module_name", SQUARE_PROPAGATORS)
def test_square_propagator_adjoint_identity(module_name: str) -> None:
    """``adjoint`` is the conjugate transpose of ``forward``:
    ``<A x, y> == <x, A* y>``."""
    module = MODULE_FACTORIES[module_name]()
    x = make_field((2, 16, 16), 2, seed=0)
    module(x)  # lazily initialise
    y = make_field((2, 16, 16), 2, seed=1)

    forward_x = module(x)
    adjoint_y = module.adjoint(y)

    lhs = torch.sum(torch.conj(forward_x._data) * y._data)
    rhs = torch.sum(torch.conj(x._data) * adjoint_y._data)

    torch.testing.assert_close(lhs, rhs, rtol=1e-3, atol=1e-2)


@pytest.mark.parametrize("module_name", SQUARE_PROPAGATORS)
def test_square_propagator_adjoint_before_init_raises(
    module_name: str,
) -> None:
    module = MODULE_FACTORIES[module_name]()
    with pytest.raises(RuntimeError, match="must be initialised"):
        module.adjoint(make_field((2, 16, 16), 2))
