"""The zeroth order a hologram leaves, as the model adds it ahead of the lens.

The module adds the light at the phase wraps, the beam on the wrapped edges times one
complex weight, and the field of an image-plane kernel. Both reach the focal plane
through the model's own lens, held between calls to ``update_wraps``.
"""

from __future__ import annotations

import math

import pytest
import torch

from hologradpy.grids import get_spatial_grid
from hologradpy.optics.complex_amplitude import ComplexAmplitude, FieldGeometry
from hologradpy.optics.modules import (
    PixelwiseSLMField,
    SuperGaussianCrosstalk,
    VirtualSLM,
    ZerothOrderField,
)
from hologradpy.optics.systems import SLMFFT

RESOLUTION = 32
PITCH = 12.5e-6
WAVELENGTH = 800e-9
FOCAL_LENGTH = 0.25
WRAP_WEIGHT = -0.3 + 0.05j


def _beam() -> torch.Tensor:
    x, y = get_spatial_grid((RESOLUTION, RESOLUTION), (PITCH, PITCH))
    return torch.exp(-(x**2 + y**2) / (0.15e-3) ** 2).to(torch.complex64)


def _model(crosstalk: bool = False) -> SLMFFT:
    """A Fourier model padded to twice the SLM, so a focal pixel is one kernel pitch."""
    geometry = FieldGeometry(
        resolution=(RESOLUTION, RESOLUTION),
        pixel_size=torch.tensor([PITCH, PITCH]),
        wavelength=torch.tensor(WAVELENGTH),
    )
    virtual_slm = VirtualSLM(
        full_scale_cycles=1.0,
        pixel_crosstalk=SuperGaussianCrosstalk(upscale_factor=2) if crosstalk else None,
    )
    beam = ComplexAmplitude(_beam(), torch.tensor(WAVELENGTH), (PITCH, PITCH))
    return SLMFFT(
        geometry,
        virtual_slm,
        PixelwiseSLMField(beam),
        FOCAL_LENGTH,
        padded_resolution=(2 * RESOLUTION, 2 * RESOLUTION),
    ).build()


def _grating(period: float = 9.3) -> torch.Tensor:
    ramp = 2 * math.pi * torch.arange(RESOLUTION) / period
    return ramp[None, :].expand(RESOLUTION, RESOLUTION).clone()


def _added_focal_field(
    module: ZerothOrderField, phase: torch.Tensor, crosstalk: bool = False
) -> torch.Tensor:
    """The focal field the module adds, the model with it less the model without."""
    plain = _model(crosstalk)
    plain.virtual_slm.set_phase(phase)
    with_field = _model(crosstalk)
    with_field.virtual_slm.set_phase(phase)
    with_field.insert_after("virtual_slm", "zeroth_order_field", module)
    module.update_wraps(with_field)
    with torch.no_grad():
        return (with_field().as_tensor() - plain().as_tensor()).squeeze()


def _wrapped_edges(fraction: torch.Tensor) -> torch.Tensor:
    """The edges at which ``fraction`` of full scale jumps by more than half of it,
    counted pixel by pixel on the pixel of lower column or row index.
    """
    edges = torch.zeros_like(fraction)
    rows, columns = fraction.shape
    for row in range(rows):
        for column in range(columns):
            here = float(fraction[row, column])
            for next_row, next_column in ((row, column + 1), (row + 1, column)):
                if next_row < rows and next_column < columns:
                    there = float(fraction[next_row, next_column])
                    edges[row, column] += abs(there - here) > 0.5
    return edges


def _edges_on_display(levels: torch.Tensor) -> torch.Tensor:
    """The edges the module finds for ``levels`` on the model, with a unit beam."""
    model = _model()
    model.virtual_slm.set_levels(levels, 8)
    module = ZerothOrderField(FOCAL_LENGTH, wrap_weight=1.0)
    module.update_wraps(model)
    return module.beam_at_wraps / _beam()


def _gaussian_kernel(size: int = 15, offset: float = 1.0) -> torch.Tensor:
    """A focal spot one pitch off centre. Its field fits the SLM's aperture, and the
    window holds it down to 0.002 of its peak, so neither edge cuts it.
    """
    rows, columns = torch.meshgrid(
        torch.arange(size) - size // 2, torch.arange(size) - size // 2, indexing="ij"
    )
    return torch.exp(-((columns - offset) ** 2 + rows**2) / 8.0).to(torch.complex64)


# --- The edges at which the displayed phase wraps -------------------------------


def test_a_blazed_grating_counts_each_wrap_on_the_pixel_before_it() -> None:
    levels = (torch.arange(RESOLUTION) % 8 * 32).expand(RESOLUTION, RESOLUTION)

    edges = _edges_on_display(levels)

    expected = torch.zeros(RESOLUTION, RESOLUTION)
    expected[:, [7, 15, 23]] = 1.0
    torch.testing.assert_close(edges.real, expected)


def test_a_step_of_half_full_scale_is_not_a_wrap() -> None:
    half = torch.zeros(RESOLUTION, RESOLUTION, dtype=torch.int64)
    half[:, 1::2] = 128
    more = half.clone()
    more[:, 1::2] = 129

    assert float(_edges_on_display(half).abs().sum()) == 0.0
    assert float(_edges_on_display(more).real[0].sum()) == RESOLUTION - 1


def test_the_edges_follow_both_axes_of_any_pattern() -> None:
    generator = torch.Generator().manual_seed(0)
    levels = torch.randint(0, 256, (RESOLUTION, RESOLUTION), generator=generator)

    edges = _edges_on_display(levels)

    torch.testing.assert_close(edges.real, _wrapped_edges(levels / 256.0))


# --- What the module adds -------------------------------------------------------


def test_a_field_with_nothing_to_add_leaves_the_model_unchanged() -> None:
    added = _added_focal_field(ZerothOrderField(FOCAL_LENGTH), _grating())

    assert float(added.abs().max()) == 0.0


def test_a_kernel_alone_lands_upright_at_the_zeroth_order() -> None:
    kernel = _gaussian_kernel()
    module = ZerothOrderField(FOCAL_LENGTH, kernel=kernel)

    added = _added_focal_field(module, _grating())

    centre, half = RESOLUTION, kernel.shape[-1] // 2
    window = added[centre - half : centre + half + 1, centre - half : centre + half + 1]
    gain = window[half, half + 1] / kernel[half, half + 1]
    torch.testing.assert_close(window / gain, kernel, atol=1e-2, rtol=0)
    model = _model()
    focal_pitch = float(model.fourier_lens.pixel_size_out.reshape(-1)[0])
    assert module.kernel_pixel_size[1] == pytest.approx(focal_pitch, rel=1e-6)


def test_the_light_at_the_wraps_is_the_beam_on_the_wrapped_edges() -> None:
    phase = _grating()
    module = ZerothOrderField(FOCAL_LENGTH, wrap_weight=WRAP_WEIGHT)

    added = _added_focal_field(module, phase)

    edges = _wrapped_edges(torch.remainder(-phase / (2 * math.pi), 1.0))
    lens = _model().fourier_lens
    at_wraps = ComplexAmplitude(
        WRAP_WEIGHT * _beam() * edges, torch.tensor(WAVELENGTH), (PITCH, PITCH)
    )
    expected = lens(at_wraps).as_tensor().squeeze()
    # The model's field less the plain model's carries the rounding of the whole field.
    torch.testing.assert_close(
        added, expected, atol=1e-4 * float(expected.abs().max()), rtol=0
    )


def test_the_added_phase_moves_the_edges_to_those_the_device_shows() -> None:
    phase = _grating(period=8.0)
    shift = torch.full((RESOLUTION, RESOLUTION), 2 * math.pi * 3.5 / 8.0)
    module = ZerothOrderField(FOCAL_LENGTH, wrap_weight=1.0, added_phase=shift)

    model = _model()
    model.virtual_slm.set_phase(phase)
    module.update_wraps(model)

    shown = model.virtual_slm.phase_response.fraction_at(phase + shift)
    # The beam on the edges is the calibrated beam without the added phase.
    edges = _wrapped_edges(torch.round(shown * 256) % 256 / 256)
    expected = _beam() * torch.exp(-1j * shift) * edges
    torch.testing.assert_close(module.beam_at_wraps, expected)
    unshifted = ZerothOrderField(FOCAL_LENGTH, wrap_weight=1.0)
    unshifted.update_wraps(model)
    assert not torch.equal(unshifted.beam_at_wraps.ne(0), module.beam_at_wraps.ne(0))


def test_a_batch_of_holograms_gets_one_set_of_edges_each() -> None:
    phases = torch.stack([_grating(7.0), _grating(11.0)])
    module = ZerothOrderField(FOCAL_LENGTH, wrap_weight=WRAP_WEIGHT)

    batched = _added_focal_field(module, phases)

    for index in range(2):
        alone = _added_focal_field(
            ZerothOrderField(FOCAL_LENGTH, wrap_weight=WRAP_WEIGHT), phases[index]
        )
        torch.testing.assert_close(batched[index], alone)


def test_the_light_at_the_wraps_holds_until_update_wraps_runs_again() -> None:
    model = _model()
    model.virtual_slm.set_phase(_grating(7.0))
    module = ZerothOrderField(FOCAL_LENGTH, wrap_weight=WRAP_WEIGHT)
    model.insert_after("virtual_slm", "zeroth_order_field", module)
    module.update_wraps(model)
    held = module.beam_at_wraps.clone()

    model.virtual_slm.set_phase(_grating(11.0))
    with torch.no_grad():
        model()
    assert torch.equal(module.beam_at_wraps, held)

    module.update_wraps(model)
    assert not torch.equal(module.beam_at_wraps, held)


def _on_both_grids(module_arguments: dict) -> tuple[torch.Tensor, torch.Tensor]:
    """The focal field a module adds on the SLM's grid, and the centre of the one it
    adds on a crosstalk grid twice as fine, where the plane is twice as wide at the
    same pitch.
    """
    phase = _grating()
    coarse = _added_focal_field(
        ZerothOrderField(FOCAL_LENGTH, **module_arguments), phase
    )
    with pytest.warns(UserWarning, match="upscaling"):
        fine = _added_focal_field(
            ZerothOrderField(FOCAL_LENGTH, **module_arguments), phase, crosstalk=True
        )
    side = coarse.shape[-1]
    return coarse, fine[side // 2 : side // 2 + side, side // 2 : side // 2 + side]


def test_a_crosstalk_grid_keeps_the_spectrum_of_the_light_at_the_wraps() -> None:
    coarse, fine = _on_both_grids({"wrap_weight": WRAP_WEIGHT})

    # The resampling keeps the coarse grid's spectrum, the focal field at every second
    # pixel from the zeroth order. The pixels between interpolate it, and the two grids
    # interpolate a field as sharp as the wrap edges differently.
    scale = float(coarse.abs().max())
    torch.testing.assert_close(
        fine[::2, ::2], coarse[::2, ::2], atol=1e-4 * scale, rtol=0
    )


def test_a_crosstalk_grid_keeps_the_field_of_the_kernel() -> None:
    coarse, fine = _on_both_grids({"kernel": _gaussian_kernel()})

    # The coarse plane repeats beyond its edge, which adds the kernel's far tail there.
    scale = float(coarse.abs().max())
    torch.testing.assert_close(fine, coarse, atol=1e-3 * scale, rtol=0)


def test_light_at_the_wraps_without_its_edges_raises() -> None:
    model = _model()
    module = ZerothOrderField(FOCAL_LENGTH, wrap_weight=0.1)
    model.insert_after("virtual_slm", "zeroth_order_field", module)

    with pytest.raises(RuntimeError, match="update_wraps"):
        model()


# --- Transforms, gradients and checkpoints --------------------------------------


def test_the_kernel_field_adjoint_satisfies_the_inner_product_identity() -> None:
    module = ZerothOrderField(FOCAL_LENGTH, kernel=_gaussian_kernel())
    model = _model()
    model.insert_after("virtual_slm", "zeroth_order_field", module)
    with torch.no_grad():
        model()
    generator = torch.Generator().manual_seed(0)
    kernel = torch.randn(
        *module.kernel.shape, dtype=torch.complex64, generator=generator
    )
    field = torch.randn(
        RESOLUTION, RESOLUTION, dtype=torch.complex64, generator=generator
    )

    forward = torch.vdot(module.kernel_field(kernel).flatten(), field.flatten())
    backward = torch.vdot(
        kernel.flatten(), module.kernel_field_adjoint(field).flatten()
    )

    torch.testing.assert_close(forward, backward, rtol=1e-4, atol=1e-6)


def test_the_gradient_reaches_the_levels_through_the_field() -> None:
    model = _model()
    model.virtual_slm.set_phase(_grating())
    module = ZerothOrderField(
        FOCAL_LENGTH, wrap_weight=WRAP_WEIGHT, kernel=_gaussian_kernel()
    )
    model.insert_after("virtual_slm", "zeroth_order_field", module)
    module.update_wraps(model)
    model.virtual_slm.levels.requires_grad_(True)

    model().as_tensor().abs().square().sum().backward()

    assert model.virtual_slm.levels.grad is not None
    assert float(model.virtual_slm.levels.grad.abs().max()) > 0.0


def test_a_checkpoint_keeps_the_weight_the_kernel_and_the_added_phase(tmp_path) -> None:
    shift = torch.full((RESOLUTION, RESOLUTION), 0.7)
    module = ZerothOrderField(
        FOCAL_LENGTH,
        wrap_weight=WRAP_WEIGHT,
        kernel=_gaussian_kernel(),
        added_phase=shift,
    )
    model = _model()
    model.insert_after("virtual_slm", "zeroth_order_field", module)
    module.update_wraps(model)
    with torch.no_grad():
        model()
    path = tmp_path / "zeroth_order_field.pt"

    module.save(str(path))
    reopened = ZerothOrderField.from_file(str(path))

    assert complex(reopened.wrap_weight) == pytest.approx(WRAP_WEIGHT, abs=1e-7)
    torch.testing.assert_close(reopened.kernel, module.kernel)
    torch.testing.assert_close(reopened.added_phase, shift)
    torch.testing.assert_close(
        reopened.kernel_field(reopened.kernel), module.kernel_field(module.kernel)
    )
