"""Pixel crosstalk: the four models of Schroff et al., Optics Express 32, 48957 (2024).

The properties that decide whether a fitted kernel means anything are checked here:
a flat phase survives, the models conserve optical power, the phase the model believes
the SLM displays is the phase the gray levels sent to the device impose, and the whole
sub-pixel grid leaves the focal plane exactly where it was.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from hologradpy.fourier_transforms import fft_resample
from hologradpy.hardware import SimulatedCameraTorch, SimulatedSLMTorch
from hologradpy.optics.complex_amplitude import ComplexAmplitude, FieldGeometry
from hologradpy.optics.modules import (
    FreeKernelCrosstalk,
    GridAdapter,
    NeighbourDifferenceCrosstalk,
    PiecewiseSuperGaussianCrosstalk,
    PixelwiseSLMField,
    SuperGaussianCrosstalk,
    VirtualSLM,
)
from hologradpy.optics.systems import SLMCZT, SLMFFT, SLMNUFFTAffine
from hologradpy.phase_levels import LinearResponse, LookupResponse

pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")

RESOLUTION: tuple[int, int] = (16, 16)
PITCH: float = 10e-6
FOCAL_LENGTH: float = 0.1
WAVELENGTH: float = 800e-9

UPSCALE_FACTORS = (1, 2, 3, 5)


def _models(upscale_factor: int, extent: int = 3) -> dict[str, object]:
    return {
        "super_gaussian": SuperGaussianCrosstalk(upscale_factor, extent),
        "piecewise": PiecewiseSuperGaussianCrosstalk(upscale_factor, extent),
        "free_kernel": FreeKernelCrosstalk(upscale_factor, extent),
        "neighbour": NeighbourDifferenceCrosstalk(upscale_factor, extent),
    }


MODEL_IDS = tuple(_models(1))
CONVOLUTIONAL_IDS = ("super_gaussian", "piecewise", "free_kernel")


def _geometry(resolution: tuple[int, int] = RESOLUTION) -> FieldGeometry:
    return FieldGeometry(
        torch.tensor(WAVELENGTH),
        torch.tensor([PITCH, PITCH]),
        resolution,
    )


def _system(crosstalk=None, **kwargs) -> SLMFFT:
    return SLMFFT(
        input_geometry=_geometry(),
        virtual_slm=VirtualSLM(phase_scaling=1.0, pixel_crosstalk=crosstalk),
        slm_field=PixelwiseSLMField(),
        focal_length=FOCAL_LENGTH,
        padded_resolution=(32, 32),
        **kwargs,
    )


# --- the models themselves -------------------------------------------------------


@pytest.mark.parametrize("upscale_factor", UPSCALE_FACTORS)
@pytest.mark.parametrize("name", MODEL_IDS)
@pytest.mark.parametrize("shape", [(6, 6), (4, 6, 6), (2, 6, 6)])
def test_a_flat_phase_comes_through_flat(
    name: str, upscale_factor: int, shape: tuple[int, ...]
) -> None:
    """Nothing to smear, so nothing changes, right up to the border.

    Exact for the neighbour model whatever its weights, since every difference is
    zero, and exact for the convolutional ones through the normalized kernel and the
    replicated edge.
    """
    model = _models(upscale_factor)[name]
    smeared = model(torch.full(shape, 0.7))

    expected = (*shape[:-2], 6 * upscale_factor, 6 * upscale_factor)
    assert tuple(smeared.shape) == expected
    torch.testing.assert_close(smeared, torch.full(expected, 0.7), atol=1e-5, rtol=0)


@pytest.mark.parametrize("upscale_factor", UPSCALE_FACTORS)
@pytest.mark.parametrize("name", CONVOLUTIONAL_IDS)
def test_the_kernel_is_the_right_size_and_sums_to_one(
    name: str, upscale_factor: int
) -> None:
    """A kernel summing to one can neither add nor remove optical power."""
    model = _models(upscale_factor)[name]
    kernel = model.kernel()

    side = 3 * upscale_factor
    assert tuple(kernel.shape) == (side, side)
    assert float(kernel.sum()) == pytest.approx(1.0, abs=1e-6)


@pytest.mark.parametrize("name", MODEL_IDS)
def test_one_pixel_reaches_only_as_far_as_the_fringing_field(name: str) -> None:
    """A phase step on one SLM pixel leaves the rest of the grid untouched.

    The kernel spans three SLM pixels, so the reach beyond that pixel's own block is
    half a kernel, and everything past it stays exactly as the flat background.
    """
    upscale_factor = 3
    model = _models(upscale_factor)[name]
    if name == "neighbour":
        # Zero weights would make this trivially true.
        with torch.no_grad():
            model.transitions.fill_(0.1)

    resolution = 9
    phase = torch.zeros(resolution, resolution)
    phase[4, 4] = 1.0

    changed = (model(phase) - model(torch.zeros_like(phase))).abs() > 1e-9
    rows = changed.any(dim=-1).nonzero().reshape(-1)
    columns = changed.any(dim=-2).nonzero().reshape(-1)

    block_start, block_stop = 4 * upscale_factor, 5 * upscale_factor - 1
    reach = (3 * upscale_factor) // 2
    for touched in (rows, columns):
        assert int(touched.min()) >= block_start - reach
        assert int(touched.max()) <= block_stop + reach


@pytest.mark.parametrize("name", MODEL_IDS)
@pytest.mark.parametrize("leading", [(5,), (3,)])
def test_a_batch_is_the_same_as_one_pattern_at_a_time(
    name: str, leading: tuple[int, ...]
) -> None:
    """Every leading axis is carried through untouched, with content that would show a
    mix-up.

    The models collapse the leading axes to reach ``conv2d`` and put them back
    afterwards. A flat phase, or a kernel left at its identity, hides a wrong
    reshape, so this uses random patterns and a random kernel. The two leading ranks
    are the batch of patterns and the per-wavelength phase ``ZernikeSLM`` produces.
    """
    generator = torch.Generator().manual_seed(4)
    patterns = torch.rand((*leading, 7, 7), generator=generator) * 2 * torch.pi

    model = {
        "super_gaussian": lambda: SuperGaussianCrosstalk(3, 3),
        "piecewise": lambda: PiecewiseSuperGaussianCrosstalk(3, 3),
        "free_kernel": lambda: FreeKernelCrosstalk(
            3, 3, init_kernel=torch.rand((9, 9), generator=generator)
        ),
        "neighbour": lambda: NeighbourDifferenceCrosstalk(
            3, 3, init_transitions=torch.rand((8, 3, 3), generator=generator) * 0.2
        ),
    }[name]()

    together = model(patterns)
    apart = torch.stack([model(patterns[i]) for i in range(patterns.shape[0])])

    torch.testing.assert_close(together, apart, atol=1e-6, rtol=0)


def test_model_one_matches_the_published_kernel_construction() -> None:
    """Model I against a plain transcription of Eq. 9, at the fitted values of the
    paper. Pins the frequency grid, the magnitude, and the normalization."""

    def published(pitch: float, upscale: int, extent: int, order, width) -> np.ndarray:
        size = upscale * extent
        maximum = 1 / (pitch * extent)
        line = np.arange(-size / 2, size / 2, 1.0) * maximum
        rows, columns = np.meshgrid(line, line, indexing="ij")
        sigma = width / pitch
        profile = np.exp(-((np.abs(rows) / sigma) ** order)) * np.exp(
            -((np.abs(columns) / sigma) ** order)
        )
        kernel = np.abs(np.fft.fftshift(np.fft.ifft2(profile)))
        return kernel / kernel.sum()

    # The pitch cancels out of the construction, so the same kernel comes out of both.
    for pitch in (12.5e-6, 8e-6):
        for upscale, order, width in [(3, 1.20, 2.03), (5, 2.3, 0.92), (7, 1.8, 1.24)]:
            model = SuperGaussianCrosstalk(upscale, 3, order=order, width=width)
            np.testing.assert_allclose(
                model.kernel().detach().numpy(),
                published(pitch, upscale, 3, order, width),
                atol=1e-6,
            )


def test_the_piecewise_model_with_equal_half_axes_is_the_symmetric_one() -> None:
    """Model II collapses onto Model I when the four half-axes agree."""
    symmetric = SuperGaussianCrosstalk(3, 3, order=1.4, width=1.7)
    piecewise = PiecewiseSuperGaussianCrosstalk(
        3,
        3,
        order_y=(1.4, 1.4),
        order_x=(1.4, 1.4),
        width_y=(1.7, 1.7),
        width_x=(1.7, 1.7),
    )
    torch.testing.assert_close(piecewise.kernel(), symmetric.kernel())


def test_the_piecewise_model_puts_its_widths_on_the_axes_it_names() -> None:
    """A narrow horizontal half-axis in frequency spreads the kernel horizontally.

    A transposed frequency grid would put ``width_x`` on the rows and go unnoticed
    everywhere else, mirroring any asymmetric kernel loaded from a fit.
    """

    def spreads(width_y, width_x) -> tuple[float, float]:
        model = PiecewiseSuperGaussianCrosstalk(
            3,
            3,
            order_y=(2.0, 2.0),
            order_x=(2.0, 2.0),
            width_y=(width_y, width_y),
            width_x=(width_x, width_x),
        )
        kernel = model.kernel().detach().double()
        offset = torch.arange(kernel.shape[0], dtype=torch.float64)
        offset = offset - kernel.shape[0] // 2
        return (
            float((kernel.sum(dim=-1) * offset**2).sum()),
            float((kernel.sum(dim=-2) * offset**2).sum()),
        )

    down_rows, across_columns = spreads(width_y=2.0, width_x=0.35)
    assert across_columns > down_rows

    # Swapping the two axes swaps the spread, and nothing else.
    swapped = spreads(width_y=0.35, width_x=2.0)
    assert swapped == pytest.approx((across_columns, down_rows), rel=1e-6)


def test_the_neighbour_model_starts_as_the_plain_upscale() -> None:
    """Zero weights leave nothing but the nearest-neighbour spread."""
    model = NeighbourDifferenceCrosstalk(3, 3)
    generator = torch.Generator().manual_seed(0)
    phase = torch.rand((5, 5), generator=generator) * 2 * torch.pi

    torch.testing.assert_close(model(phase), model.repeat_pixels(phase))


def test_a_neighbour_weight_moves_phase_from_the_pixel_it_names() -> None:
    """Turning on one transition matrix pulls each sub-pixel towards that neighbour
    alone, by the fraction the matrix says."""
    upscale_factor = 3
    model = NeighbourDifferenceCrosstalk(upscale_factor, 3)
    index = model.neighbour_offsets.index((0, 1))
    with torch.no_grad():
        model.transitions[index, 1, 2] = 0.25

    phase = torch.zeros(4, 4)
    phase[1, 2] = 1.0
    smeared = model(phase)

    # Pixel (1, 1) sits one column left of the raised pixel, so its sub-pixel (1, 2)
    # moves a quarter of the way there. Its other sub-pixels do not.
    assert float(smeared[1 * 3 + 1, 1 * 3 + 2]) == pytest.approx(0.25)
    assert float(smeared[1 * 3 + 0, 1 * 3 + 2]) == pytest.approx(0.0)
    # Pixel (1, 3) sits on the other side, so this weight does nothing for it.
    assert float(smeared[1 * 3 + 1, 3 * 3 + 2]) == pytest.approx(0.0)


@pytest.mark.parametrize("name", MODEL_IDS)
def test_gradients_reach_every_parameter(name: str) -> None:
    """The fitting work that follows needs something to optimize."""
    model = _models(3)[name]
    generator = torch.Generator().manual_seed(1)
    phase = (torch.rand((5, 5), generator=generator) * 2 * torch.pi).requires_grad_()

    model(phase).pow(2).sum().backward()

    assert torch.isfinite(phase.grad).all()
    for parameter_name, parameter in model.named_parameters():
        assert parameter.grad is not None, parameter_name
        assert torch.isfinite(parameter.grad).all(), parameter_name
        assert float(parameter.grad.abs().sum()) > 0.0, parameter_name


def test_gradients_reach_the_slm_plane_through_the_sub_pixel_grid() -> None:
    """The grid adapter has to hand its output on as an on-graph field.

    The stage after it multiplies through ``__torch_dispatch__``, so an adapter that
    wrapped its result as an autograd leaf would strand every SLM-plane parameter with
    no gradient and no error at all. The wavefront fit turns on exactly those
    parameters, so it would silently optimize nothing.
    """

    def slm_field_gradients(crosstalk):
        system = _system(crosstalk)
        with torch.no_grad():
            system()
        generator = torch.Generator().manual_seed(8)
        system.virtual_slm.set_phase(
            torch.rand(RESOLUTION, generator=generator) * 2 * torch.pi
        )
        for parameter in system.slm_field.parameters():
            parameter.requires_grad_(True)
        image = system().intensity
        # Weighted by position, so where the light lands matters and the phase counts.
        weights = torch.linspace(0.0, 1.0, image.shape[-1])
        (image * weights).sum().backward()
        return system.slm_field.amplitude.grad, system.slm_field.phase.grad

    for crosstalk in (None, SuperGaussianCrosstalk(3, 3), FreeKernelCrosstalk(2, 3)):
        amplitude, phase = slm_field_gradients(crosstalk)
        label = type(crosstalk).__name__
        assert amplitude is not None, label
        assert phase is not None, label
        assert float(amplitude.abs().sum()) > 0.0, label
        assert float(phase.abs().sum()) > 0.0, label


def test_a_free_kernel_warm_starts_from_a_fitted_parametric_one() -> None:
    fitted = SuperGaussianCrosstalk(3, 3, order=1.4, width=1.7)
    free = FreeKernelCrosstalk.from_parametric(fitted)

    torch.testing.assert_close(free.kernel(), fitted.kernel())


def test_crosstalk_grows_with_the_spatial_frequency_of_the_pattern() -> None:
    """The claim the paper rests on: a steeper grating is smeared more."""
    model = SuperGaussianCrosstalk(5, 3)
    columns = torch.arange(32.0)

    smearing = []
    for cycles in (2.0, 8.0, 16.0):
        phase = (2 * torch.pi * cycles / 32 * columns).expand(32, 32)
        wrapped = phase % (2 * torch.pi)
        smearing.append(
            float((model(wrapped) - model.repeat_pixels(wrapped)).pow(2).mean().sqrt())
        )

    assert smearing[0] < smearing[1] < smearing[2]


# --- resampling ------------------------------------------------------------------


@pytest.mark.parametrize("resolution", [7, 8, 9, 16])
@pytest.mark.parametrize("factor", [2, 3])
def test_fft_resample_is_exact_for_what_the_coarse_grid_resolves(
    resolution: int, factor: int
) -> None:
    """A mode the coarse grid already carries lands on the fine grid untouched."""
    fine = resolution * factor

    def mode(size: int) -> torch.Tensor:
        rows = torch.arange(size, dtype=torch.float64).reshape(-1, 1) / size
        columns = torch.arange(size, dtype=torch.float64).reshape(1, -1) / size
        return torch.cos(2 * torch.pi * (2 * rows - columns)) + 0.3

    resampled = fft_resample(mode(resolution), (fine, fine))

    assert not resampled.is_complex()
    torch.testing.assert_close(resampled, mode(fine), atol=1e-12, rtol=0)


@pytest.mark.parametrize("resolution", [7, 8, 16])
def test_fft_resample_returns_a_complex_field_to_where_it_started(
    resolution: int,
) -> None:
    generator = torch.Generator().manual_seed(2)
    field = torch.randn(
        resolution, resolution, generator=generator, dtype=torch.float64
    ) + 1j * torch.randn(
        resolution, resolution, generator=generator, dtype=torch.float64
    )

    upscaled = fft_resample(field, (resolution * 3, resolution * 3))
    torch.testing.assert_close(
        fft_resample(upscaled, (resolution, resolution)), field, atol=1e-12, rtol=0
    )


@pytest.mark.parametrize("factor", [1, 2, 3])
def test_the_grid_adapter_keeps_the_plane_the_same_size(factor: int) -> None:
    adapter = GridAdapter(factor=factor)
    field = ComplexAmplitude(
        torch.ones(RESOLUTION, dtype=torch.complex64),
        torch.tensor(WAVELENGTH),
        (PITCH, PITCH),
    )

    resampled = adapter(field)

    assert tuple(resampled.resolution) == tuple(
        length * factor for length in RESOLUTION
    )
    torch.testing.assert_close(
        resampled.pixel_size * torch.tensor(resampled.resolution, dtype=torch.float32),
        field.pixel_size * torch.tensor(RESOLUTION, dtype=torch.float32),
    )
    if factor == 1:
        assert resampled is field


def test_the_grid_adapter_caches_only_when_asked() -> None:
    field = ComplexAmplitude(
        torch.ones(RESOLUTION, dtype=torch.complex64),
        torch.tensor(WAVELENGTH),
        (PITCH, PITCH),
    )

    cached = GridAdapter(factor=2, cache=True)
    first = cached(field)
    assert cached(field) is first

    cached.clear_cache()
    again = cached(field)
    assert again is not first
    torch.testing.assert_close(again.as_tensor(), first.as_tensor())

    uncached = GridAdapter(factor=2)
    assert uncached(field) is not uncached(field)


# --- the wrap has to be the one the hardware uses --------------------------------


@pytest.mark.parametrize("phase_scaling", [0.6, 1.0, 1.5, 2.0])
def test_the_model_and_the_device_wrap_on_the_same_pixels(
    phase_scaling: float,
) -> None:
    """The phase a simulation believes it displays is the phase the levels sent to the
    device impose.

    Without this the crosstalk convolution smears across a step the SLM puts somewhere
    else, and a fitted kernel means nothing.
    """
    virtual_slm = VirtualSLM(phase_scaling=phase_scaling)
    virtual_slm(
        ComplexAmplitude(
            torch.ones(RESOLUTION, dtype=torch.complex64),
            torch.tensor(WAVELENGTH),
            (PITCH, PITCH),
        )
    )

    # Several cycles, as a blazed grating covers.
    generator = torch.Generator().manual_seed(3)
    virtual_slm.set_phase(
        (torch.rand(RESOLUTION, generator=generator) * 8 - 4) * torch.pi
    )

    believed = virtual_slm.get_phase().double()
    sent = virtual_slm.phase_to_levels(believed)
    imposed = virtual_slm.levels_to_phase(torch.as_tensor(sent.astype(np.float64)))

    # One gray level of slack, since the levels are whole numbers.
    step = 2 * torch.pi * phase_scaling / virtual_slm.phase_response.number_of_levels
    assert float((imposed - believed).abs().max()) < step

    # And the fraction of full scale really is one the device can show.
    fraction = virtual_slm.displayed_levels()
    assert float(fraction.min()) >= 0.0
    assert float(fraction.max()) < 1.0


def test_a_measured_response_wraps_where_its_table_ends() -> None:
    levels = np.arange(256)
    response = LookupResponse(
        bitdepth=8,
        phases=-2.4 * np.pi * (0.5 - 0.5 * np.cos(np.pi * levels / levels[-1])),
    )
    virtual_slm = VirtualSLM(phase_response=response)
    virtual_slm(
        ComplexAmplitude(
            torch.ones(RESOLUTION, dtype=torch.complex64),
            torch.tensor(WAVELENGTH),
            (PITCH, PITCH),
        )
    )
    generator = torch.Generator().manual_seed(4)
    virtual_slm.set_phase(
        (torch.rand(RESOLUTION, generator=generator) * 8 - 4) * torch.pi
    )

    fraction = virtual_slm.displayed_levels()
    assert float(fraction.min()) >= 0.0
    assert float(fraction.max()) <= 1.0


@pytest.mark.parametrize("phase_scaling", [0.6, 1.0, 1.5])
def test_a_level_means_the_phase_the_response_says(phase_scaling: float) -> None:
    """Full scale reaches ``phase_scaling`` cycles, which is what the whole package
    documents the number to mean."""
    response = LinearResponse(bitdepth=8, phase_scaling=phase_scaling)

    assert float(response.phase_at(np.array(1.0))) == pytest.approx(
        -2 * np.pi * phase_scaling
    )
    assert float(response.phase_at(np.array(0.0))) == pytest.approx(0.0)


def test_quantizing_holds_the_phase_on_whole_levels() -> None:
    virtual_slm = VirtualSLM(phase_scaling=1.0, quantize=True)
    virtual_slm(
        ComplexAmplitude(
            torch.ones(RESOLUTION, dtype=torch.complex64),
            torch.tensor(WAVELENGTH),
            (PITCH, PITCH),
        )
    )
    generator = torch.Generator().manual_seed(5)
    virtual_slm.set_phase(torch.rand(RESOLUTION, generator=generator) * 2 * torch.pi)

    quantized = virtual_slm.apply_phase_transforms(virtual_slm.get_phase())
    levels = virtual_slm.phase_response.to_levels(quantized)

    torch.testing.assert_close(levels, levels.round(), atol=1e-4, rtol=0)
    # Applying it again changes nothing.
    torch.testing.assert_close(
        virtual_slm.apply_phase_transforms(quantized), quantized, atol=1e-5, rtol=0
    )


# --- the SLM stage on a sub-pixel grid -------------------------------------------


def test_the_slm_keeps_one_pattern_value_per_real_pixel() -> None:
    upscale_factor = 4
    virtual_slm = VirtualSLM(
        phase_scaling=1.0,
        pixel_crosstalk=FreeKernelCrosstalk(upscale_factor=upscale_factor),
    )
    fine = tuple(length * upscale_factor for length in RESOLUTION)
    virtual_slm(
        ComplexAmplitude(
            torch.ones(fine, dtype=torch.complex64),
            torch.tensor(WAVELENGTH),
            (PITCH / upscale_factor, PITCH / upscale_factor),
        )
    )

    assert virtual_slm.slm_resolution == RESOLUTION
    assert tuple(virtual_slm.levels.shape) == RESOLUTION
    assert tuple(virtual_slm.resolution_out) == fine

    virtual_slm.set_phase(torch.zeros(RESOLUTION))
    with pytest.raises(ValueError, match="does not match the SLM resolution"):
        virtual_slm.set_phase(torch.zeros(fine))

    # The pattern grid is the SLM's own, coarser than the field's by the factor.
    grid_x, _ = virtual_slm.get_slm_grid()
    assert tuple(grid_x.shape) == RESOLUTION
    fine_x, _ = virtual_slm.get_spatial_grid_input()
    assert tuple(fine_x.shape) == fine


def test_an_slm_stage_refuses_a_field_it_cannot_divide() -> None:
    virtual_slm = VirtualSLM(
        phase_scaling=1.0, pixel_crosstalk=FreeKernelCrosstalk(upscale_factor=3)
    )
    with pytest.raises(ValueError, match="GridAdapter"):
        virtual_slm(
            ComplexAmplitude(
                torch.ones((16, 16), dtype=torch.complex64),
                torch.tensor(WAVELENGTH),
                (PITCH, PITCH),
            )
        )


def test_an_slm_stage_refuses_a_factor_that_divides_but_disagrees() -> None:
    """A factor that happens to divide the resolution still has to land on the real
    pitch, or the model is describing a different SLM."""

    class _Device:
        pixel_size = (PITCH, PITCH)
        phase_response = LinearResponse(bitdepth=8)

    virtual_slm = VirtualSLM.from_slm_data(_Device())
    virtual_slm.pixel_crosstalk = FreeKernelCrosstalk(upscale_factor=2)

    with pytest.raises(ValueError, match="does not match the crosstalk model"):
        virtual_slm(
            ComplexAmplitude(
                torch.ones((16, 16), dtype=torch.complex64),
                torch.tensor(WAVELENGTH),
                (PITCH, PITCH),
            )
        )


# --- whole systems ---------------------------------------------------------------


@pytest.mark.parametrize("name", MODEL_IDS)
@pytest.mark.parametrize("upscale_factor", [2, 3])
def test_the_focal_plane_stays_put_under_the_sub_pixel_grid(
    name: str, upscale_factor: int
) -> None:
    """A blazed grating lands in the same place with the sub-pixel grid as without it.

    The sub-pixel grid makes the focal plane larger, so the position is measured from
    its center. The check that the padded resolution scaled with the upscale factor:
    without that the focal-plane pixel would grow and the spot would move in.
    """

    def first_order(crosstalk) -> tuple[int, int]:
        system = _system(crosstalk)
        system()
        grid_x, _ = system.virtual_slm.get_slm_grid()
        system.virtual_slm.set_phase(
            2 * torch.pi * (4.0 / (RESOLUTION[1] * PITCH)) * grid_x
        )
        image = system().intensity.detach()
        height, width = image.shape[-2:]
        index = int(torch.argmax(image.reshape(-1)))
        row, column = divmod(index, width)
        return row - height // 2, column - width // 2

    assert first_order(_models(upscale_factor)[name]) == first_order(None)


@pytest.mark.parametrize("name", MODEL_IDS)
def test_no_optical_power_is_created_or_lost(name: str) -> None:
    system = _system(_models(3)[name])
    system()
    generator = torch.Generator().manual_seed(6)
    system.virtual_slm.set_phase(
        torch.rand(RESOLUTION, generator=generator) * 2 * torch.pi
    )

    plain = _system(None)
    plain()
    plain.virtual_slm.set_phase(system.virtual_slm.get_phase())

    assert float(system().power()) == pytest.approx(float(plain().power()), rel=1e-4)


def test_the_reorder_leaves_a_system_without_crosstalk_alone() -> None:
    """The beam and the displayed phase are diagonal multiplies, so running the beam
    first is the same calculation."""
    system = _system(None)
    system()
    generator = torch.Generator().manual_seed(7)
    phase = torch.rand(RESOLUTION, generator=generator) * 2 * torch.pi
    system.virtual_slm.set_phase(phase)

    beam = system.slm_field.get_transmission()
    displayed = torch.exp(1j * system.virtual_slm.get_phase())
    expected = system.fourier_lens(
        ComplexAmplitude(
            (displayed * beam).reshape(RESOLUTION),
            torch.tensor(WAVELENGTH),
            (PITCH, PITCH),
        )
    )

    torch.testing.assert_close(
        system().as_tensor().reshape(32, 32),
        expected.as_tensor().reshape(32, 32),
        atol=1e-5,
        rtol=1e-4,
    )


@pytest.mark.parametrize("system_class", [SLMCZT, SLMNUFFTAffine])
def test_the_other_lenses_take_the_sub_pixel_grid(system_class) -> None:
    system = system_class(
        input_geometry=_geometry(),
        virtual_slm=VirtualSLM(
            phase_scaling=1.0, pixel_crosstalk=FreeKernelCrosstalk(3, 3)
        ),
        slm_field=PixelwiseSLMField(),
        camera_resolution=RESOLUTION,
        camera_pixel_size=(5e-6, 5e-6),
        focal_length=FOCAL_LENGTH,
    )

    output = system()

    assert tuple(output.resolution) == RESOLUTION
    torch.testing.assert_close(
        output.pixel_size[0], torch.tensor([5e-6, 5e-6]), atol=1e-12, rtol=0
    )


def test_the_crosstalk_follows_the_field_onto_its_device_and_dtype() -> None:
    """The kernel is sized at construction, so it starts on the CPU in single
    precision and has to catch up with the field on the first forward.

    A system built for CUDA is never moved with ``.to(device)`` in the examples: its
    geometry carries the device and every module builds its state to match. A kernel
    left behind makes ``conv2d`` raise on the first frame.
    """
    virtual_slm = VirtualSLM(
        phase_scaling=1.0, pixel_crosstalk=FreeKernelCrosstalk(2, 3)
    )
    virtual_slm(
        ComplexAmplitude(
            torch.ones((32, 32), dtype=torch.complex128),
            torch.tensor(WAVELENGTH, dtype=torch.float64),
            (PITCH / 2, PITCH / 2),
        )
    )

    assert virtual_slm.pixel_crosstalk.kernel().dtype == torch.float64
    assert virtual_slm.get_phase().dtype == torch.float64


def test_a_system_with_crosstalk_survives_a_checkpoint(tmp_path) -> None:
    system = _system(FreeKernelCrosstalk(3, 3))
    system()
    with torch.no_grad():
        system.virtual_slm.pixel_crosstalk.weights.add_(0.01)

    path = tmp_path / "crosstalk.pt"
    system.save(str(path))
    reopened = SLMFFT.load(str(path))

    torch.testing.assert_close(
        reopened.virtual_slm.pixel_crosstalk.kernel(),
        system.virtual_slm.pixel_crosstalk.kernel(),
    )
    torch.testing.assert_close(reopened().as_tensor(), system().as_tensor())


# --- the simulated camera carries it, the retrieval model does not ---------------


def _simulated_pair(crosstalk_upscale_factor: int | None):
    """An SLM and a simulated camera sharing it, as the feedback example builds them."""
    geometry = _geometry()
    slm = SimulatedSLMTorch(input_geometry=geometry, bitdepth=8)
    beam = ComplexAmplitude(
        torch.ones(RESOLUTION) + 0j,
        wavelength=geometry.wavelength,
        pixel_size=geometry.pixel_size,
        power=1e-3,
    )
    model = SLMCZT(
        input_geometry=geometry,
        virtual_slm=slm.virtual_slm,
        camera_resolution=RESOLUTION,
        camera_pixel_size=(5e-6, 5e-6),
        focal_length=FOCAL_LENGTH,
        slm_field=PixelwiseSLMField(beam),
        padded_resolution=(24, 24),
    )
    camera = SimulatedCameraTorch(
        slm_camera_model=model,
        bitdepth=12,
        nd_filter_optical_density=6,
        noise_level=0,
        crosstalk_upscale_factor=crosstalk_upscale_factor,
    )
    return slm, camera, model


def test_the_camera_leaves_the_model_alone_without_crosstalk_arguments() -> None:
    _, camera, model = _simulated_pair(None)

    assert camera.slm_camera_model is model
    assert camera.slm_camera_model.virtual_slm.pixel_crosstalk is None
    assert camera.slm_camera_model.grid_adapter.factor == 1


def test_the_camera_carries_crosstalk_the_retrieval_model_knows_nothing_about() -> None:
    slm, camera, model = _simulated_pair(3)
    crosstalk = camera.slm_camera_model.virtual_slm.pixel_crosstalk

    # Rebuilt onto the sub-pixel grid, sharing the SLM it was handed.
    assert camera.slm_camera_model is not model
    assert isinstance(crosstalk, SuperGaussianCrosstalk)
    assert crosstalk.upscale_factor == 3
    assert camera.slm_camera_model.grid_adapter.factor == 3
    assert camera.slm_camera_model.virtual_slm is slm.virtual_slm

    # A model built separately from the same SLM has none, so a retriever
    # optimizing against it never sees the crosstalk.
    retrieval = SLMCZT(
        input_geometry=_geometry(),
        virtual_slm=VirtualSLM.from_slm(slm),
        camera_resolution=RESOLUTION,
        camera_pixel_size=(5e-6, 5e-6),
        focal_length=FOCAL_LENGTH,
        slm_field=PixelwiseSLMField(),
        padded_resolution=(24, 24),
    )
    retrieval()
    assert retrieval.virtual_slm.pixel_crosstalk is None
    assert retrieval.virtual_slm is not slm.virtual_slm


def test_the_rebuilt_camera_reports_the_geometry_it_would_have_anyway() -> None:
    _, plain, _ = _simulated_pair(None)
    _, smeared, _ = _simulated_pair(3)

    assert tuple(smeared.resolution) == tuple(plain.resolution)
    np.testing.assert_allclose(smeared.pixel_size, plain.pixel_size)
    assert smeared.bitdepth == plain.bitdepth
    assert smeared.max_pixel_value == plain.max_pixel_value


def test_a_phase_set_before_any_capture_reaches_the_sub_pixel_camera() -> None:
    """``SimulatedSLMTorch`` builds its virtual SLM from the SLM plane, which is
    coarser than the field the stage reads once crosstalk is fitted."""
    slm, camera, _ = _simulated_pair(3)

    grid_x, _unused = slm.get_spatial_grid()
    slm.set_phase(2 * torch.pi * (3.0 / (RESOLUTION[1] * PITCH)) * grid_x)

    assert tuple(slm.virtual_slm.levels.shape) == RESOLUTION
    assert camera.get_image().shape == tuple(camera.resolution)
    assert camera.static_slm_field.shape == RESOLUTION


def test_crosstalk_costs_a_blazed_grating_its_efficiency() -> None:
    """The mechanism the paper describes: crosstalk takes light out of the first order
    of a blazed grating and leaves it undiffracted, and the loss grows with the spatial
    frequency of the grating.

    Measured as the excess over the same system without crosstalk, since a finite
    aperture puts light at the center whatever the SLM does.
    """
    slm_resolution = (96, 120)
    camera_resolution = (240, 240)

    def undiffracted_fraction(crosstalk, cycles: float) -> float:
        system = SLMCZT(
            input_geometry=_geometry(slm_resolution),
            virtual_slm=VirtualSLM(phase_scaling=1.0, pixel_crosstalk=crosstalk),
            camera_resolution=camera_resolution,
            camera_pixel_size=(20e-6, 20e-6),
            focal_length=FOCAL_LENGTH,
            slm_field=PixelwiseSLMField(),
            padded_resolution=(120, 144),
        )
        system()
        grid_x, _unused = system.virtual_slm.get_slm_grid()
        system.virtual_slm.set_phase(
            2 * torch.pi * (cycles / (slm_resolution[1] * PITCH)) * grid_x
        )
        image = system().intensity.detach().reshape(camera_resolution)
        center = camera_resolution[0] // 2, camera_resolution[1] // 2
        undiffracted = image[
            center[0] - 2 : center[0] + 3, center[1] - 2 : center[1] + 3
        ].sum()
        return float(undiffracted / image.sum())

    # The square pixel on its own leaves the undiffracted order where it was. Only the
    # smearing feeds it, so the excess over that baseline is the crosstalk.
    def excess(cycles: float) -> float:
        smeared = undiffracted_fraction(SuperGaussianCrosstalk(3, 3), cycles)
        square_pixel = undiffracted_fraction(FreeKernelCrosstalk(3, 3), cycles)
        assert smeared > square_pixel
        return smeared - square_pixel

    assert excess(2.0) < excess(6.0) < excess(12.0)
