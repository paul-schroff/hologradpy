"""The chirp-z (Bluestein) partial affine of the spectrum.

``ChirpZPartialAffine`` is the *exact* DFT sampled on a scaled, shifted and rotated
window, so it must match the full FFT (and a padded FFT for a true zoom), have a correct
conjugate-transpose adjoint, and be differentiable. Its rotation must turn by the right
angle in the right direction (validated with an *asymmetric* pattern, since a symmetric
one is rotation-invariant) and conserve power.
"""

from __future__ import annotations

import math

import pytest
import torch

from hologradpy.utils import to_canvas
from hologradpy.fourier_transforms import (
    fft_2d,
    ChirpZPartialAffine,
)
from hologradpy.roi import ROI


pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")


def _random_field(n: int) -> torch.Tensor:
    torch.manual_seed(0)
    return (torch.randn(n, n) + 1j * torch.randn(n, n)).to(torch.complex64)


def _elliptical_gaussian(n: int, sigma_x: float, sigma_y: float, theta: float):
    grid_y, grid_x = torch.meshgrid(
        torch.arange(n) - n // 2, torch.arange(n) - n // 2, indexing="ij"
    )
    cos, sin = math.cos(theta), math.sin(theta)
    rotated_x = cos * grid_x + sin * grid_y
    rotated_y = -sin * grid_x + cos * grid_y
    return torch.exp(
        -0.5 * ((rotated_x / sigma_x) ** 2 + (rotated_y / sigma_y) ** 2)
    ).to(torch.complex64)


def _relative_error(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a - b).abs().max() / b.abs().max())


# %% Chirp-z zoom correctness
def test_czt_matches_fft_at_unit_magnification() -> None:
    field = _random_field(48)
    czt = ChirpZPartialAffine((48, 48), (48, 48), (1.0, 1.0))
    reference = fft_2d(field, norm="backward", fft_shift=True)
    assert _relative_error(czt.forward(field), reference) < 1e-3


def test_czt_zoom_matches_padded_fft() -> None:
    n, magnification = 48, 2.0
    field = _random_field(n)
    czt = ChirpZPartialAffine((n, n), (n, n), (magnification, magnification))

    padded = int(n * magnification)
    offset = (padded - n) // 2
    padded_field = torch.zeros(padded, padded, dtype=torch.complex64)
    padded_field[offset:offset + n, offset:offset + n] = field
    full = fft_2d(padded_field, norm="backward", fft_shift=True)
    center = padded // 2
    window = full[center - n // 2:center + n - n // 2,
                  center - n // 2:center + n - n // 2]
    assert _relative_error(czt.forward(field), window) < 1e-3


def test_czt_adjoint_is_conjugate_transpose() -> None:
    field = _random_field(48)
    other = (torch.randn(48, 48) + 1j * torch.randn(48, 48)).to(torch.complex64)
    czt = ChirpZPartialAffine(
        (48, 48), (48, 48), (1.5, 1.5), shift=(0.3, -0.2),
        angle=math.radians(20),
    )
    forward_inner = (czt.forward(field) * other.conj()).sum()
    adjoint_inner = (field * czt.adjoint(other).conj()).sum()
    assert float((forward_inner - adjoint_inner).abs() / forward_inner.abs()) < 1e-4


def test_czt_is_differentiable() -> None:
    field = _random_field(32).requires_grad_(True)
    czt = ChirpZPartialAffine((32, 32), (32, 32), (1.5, 1.5))
    czt.forward(field).abs().sum().backward()
    assert field.grad is not None


# %% Framing


def test_pad_crop_centers_a_field_in_a_larger_frame() -> None:
    """The center sample has to land on the center sample. Any other offset is a phase
    ramp in the conjugate plane, so it reads as a tilt across the focal plane rather
    than as a shifted image.
    """
    field = torch.zeros((64, 80), dtype=torch.complex64)
    field[64 // 2, 80 // 2] = 1.0

    placed = to_canvas(field, (96, 112))

    assert tuple(placed.shape) == (96, 112)
    assert int(placed.abs().argmax()) == (96 // 2) * 112 + 112 // 2


def test_pad_crop_crops_as_well_as_grows() -> None:
    """The same call shrinks a frame, which is what lets a rotation out into a larger
    frame and the rotation back into the smaller one be the same operation.
    """
    field = _elliptical_gaussian(96, 10.0, 6.0, 0.0)

    assert tuple(to_canvas(field, (48, 64)).shape) == (48, 64)
    assert tuple(to_canvas(field, (96, 96)).shape) == (96, 96)


def test_growing_a_frame_is_the_transpose_of_shrinking_it() -> None:
    """``FourierLensCZT`` grows the frame on the way out and shrinks it on the way back,
    and its adjoint is a true conjugate transpose only if those two are transposes.
    """
    small, large = (32, 40), (48, 56)
    generator = torch.Generator().manual_seed(0)
    source = torch.randn(small, generator=generator, dtype=torch.float64)
    probe = torch.randn(large, generator=generator, dtype=torch.float64)

    left = float((to_canvas(source, large) * probe).sum())
    right = float((source * to_canvas(probe, small)).sum())
    assert left == pytest.approx(right, rel=1e-12)


# %% Rotation, which now lives inside the transform


def test_chirpz_rotation_conserves_power() -> None:
    """The rotation is area preserving, so it must not cost the field any power. A
    compact field, since anything the shear carries past the frame edge is cropped and
    that loss is framing, not rotation.
    """
    field = _elliptical_gaussian(96, 10.0, 6.0, 0.0)
    settings = ((96, 96), (96, 96), (1.0, 1.0))

    plain = ChirpZPartialAffine(*settings).forward(field)
    rotated = ChirpZPartialAffine(*settings, angle=math.radians(25)).forward(field)

    ratio = float(rotated.abs().pow(2).sum() / plain.abs().pow(2).sum())
    assert ratio == pytest.approx(1.0, abs=1e-3)


@pytest.mark.parametrize("angle", [0.0, 1e-6, 0.05, 0.3, -0.4])
def test_chirpz_angle_gradient_matches_finite_differences(angle: float) -> None:
    """The gradient with respect to the angle is right, including at exactly zero.

    Zero is the value a calibration starts from, and it is where a transform length that
    steps discontinuously with the angle would corrupt the gradient: the shear's padding
    is sized from the largest shift, so rounding it up rather than flooring it would
    single out a shift of exactly zero.

    In double precision throughout, since a central difference of this size is swamped
    by float32 rounding, which would mask the very defect being checked. The default
    dtype has to move too: the frequency grids are built from it, and a float32 grid
    caps the transform near 1e-5 however precise the field is.
    """
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        field = _elliptical_gaussian(32, 6.0, 4.0, 0.0).to(torch.complex128)
        weight = _elliptical_gaussian(32, 9.0, 3.0, 0.7).to(torch.complex128)

        def loss(value: torch.Tensor) -> torch.Tensor:
            transform = ChirpZPartialAffine(
                (32, 32), (32, 32), (1.0, 1.0), angle=value
            )
            return (transform.forward(field) * weight).real.sum()

        parameter = torch.tensor(angle, dtype=torch.float64, requires_grad=True)
        loss(parameter).backward()

        step = 1e-5
        numeric = (
            loss(torch.tensor(angle + step, dtype=torch.float64))
            - loss(torch.tensor(angle - step, dtype=torch.float64))
        ) / (2 * step)

        torch.testing.assert_close(
            parameter.grad, numeric.detach(), rtol=1e-4, atol=1e-6
        )
    finally:
        torch.set_default_dtype(previous)


def test_chirpz_rotates_in_the_direction_it_claims() -> None:
    """An elongated gaussian, so the direction is unmistakable: its transform must match
    the transform of the analytically rotated ellipse and clearly not the opposite
    rotation. A symmetric field would pass either way.
    """
    theta = math.radians(25)
    plain = ChirpZPartialAffine((96, 96), (96, 96), (1.0, 1.0))
    rotated = ChirpZPartialAffine((96, 96), (96, 96), (1.0, 1.0), angle=theta).forward(
        _elliptical_gaussian(96, 13.0, 5.0, 0.0)
    )

    same = plain.forward(_elliptical_gaussian(96, 13.0, 5.0, theta))
    opposite = plain.forward(_elliptical_gaussian(96, 13.0, 5.0, -theta))

    assert _relative_error(rotated, same) < 1e-2
    assert _relative_error(rotated, opposite) > 0.1


def test_chirpzzoom_rotation_matches_the_rotated_sample_grid() -> None:
    """The rotation, against a direct sum over the points it claims to visit.

    Both are exact, so they agree to floating point. This is the test that would catch a
    sign or an axis swap in the folded rotation, which would otherwise show up only as a
    wavefront recovered the wrong way round.
    """
    # The chirp phases grow as step * n**2, so a float32 frequency grid caps the whole
    # transform near 1e-5 however precise the field is. Exactness needs float64 ramps.
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        _rotation_matches_the_rotated_sample_grid()
    finally:
        torch.set_default_dtype(previous)


def _rotation_matches_the_rotated_sample_grid() -> None:
    resolution, resolution_out = (48, 64), (40, 56)
    rows = torch.arange(resolution[0], dtype=torch.float64) - resolution[0] // 2
    columns = torch.arange(resolution[1], dtype=torch.float64) - resolution[1] // 2
    y, x = torch.meshgrid(rows, columns, indexing="ij")
    # Compact, so the shear that is left carries nothing past the frame edge, where it
    # would be cropped. That loss is a property of shearing a field that fills its
    # frame, not of the factorisation, and it would otherwise set the tolerance here.
    field = torch.exp(-0.5 * ((x / 3.0) ** 2 + (y / 3.0) ** 2)).to(torch.complex128)

    for degrees in (-19.0, 0.0, 7.0, 25.0):
        transform = ChirpZPartialAffine(
            resolution, resolution_out, (1.3, 0.8), (0.21, -0.13),
            math.radians(degrees),
        )
        # frequencies carries the points it says it samples, (2, H_out * W_out).
        omega = transform.frequencies
        phase = torch.exp(
            -1j
            * (
                omega[0][:, None, None] * columns[None, None, :]
                + omega[1][:, None, None] * rows[None, :, None]
            )
        )
        expected = (phase * field).sum(dim=(-2, -1)).reshape(resolution_out)

        assert _relative_error(transform.forward(field), expected) < 1e-10, degrees


@pytest.fixture
def float64_default():
    """Double precision for the field and for the frequency grids, which are built in
    the default dtype.
    """
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(previous)


def _random_phase_field(n: int, seed: int = 0) -> torch.Tensor:
    """A unit field with a random phase on every sample. It fills its frame and holds
    the Nyquist frequency, where the shear of a rotation spreads a field furthest.
    """
    generator = torch.Generator().manual_seed(seed)
    return torch.exp(2j * math.pi * torch.rand(n, n, generator=generator))


def _direct_sum(transform: ChirpZPartialAffine, field: torch.Tensor) -> torch.Tensor:
    """The DFT of ``field`` at the points the transform samples, as
    ``(H_out * W_out,)``.
    """
    rows = torch.arange(field.shape[-2]) - field.shape[-2] // 2
    columns = torch.arange(field.shape[-1]) - field.shape[-1] // 2
    omega = transform.frequencies
    along_columns = torch.exp(-1j * omega[0][:, None] * columns[None, :])
    along_rows = torch.exp(-1j * omega[1][:, None] * rows[None, :])
    return torch.einsum("kc,kr,rc->k", along_columns, along_rows, field)


@pytest.mark.parametrize("degrees", [0.5, 3.0, -20.0])
def test_rotation_of_a_field_that_fills_its_frame_is_exact(
    float64_default, degrees: float
) -> None:
    """The window covers part of the order, so the shear is windowed to the row
    frequencies the transform samples, and the transform matches a direct sum over
    its points.
    """
    field = _random_phase_field(64)
    transform = ChirpZPartialAffine(
        (64, 64), (48, 48), (1.7, 1.7), (0.21, -0.13), math.radians(degrees)
    )
    expected = _direct_sum(transform, field).reshape(48, 48)

    assert _relative_error(transform.forward(field), expected) < 1e-10


def test_the_windowed_rotation_keeps_the_adjoint_its_transpose(float64_default) -> None:
    generator = torch.Generator().manual_seed(1)
    field = _random_phase_field(48)
    samples = torch.randn(40, 40, generator=generator) + 1j * torch.randn(
        40, 40, generator=generator
    )
    transform = ChirpZPartialAffine(
        (48, 48), (40, 40), (1.5, 1.5), (0.3, -0.2), math.radians(12.0)
    )

    forward_inner = (transform.forward(field) * samples.conj()).sum()
    adjoint_inner = (field * transform.adjoint(samples).conj()).sum()

    assert abs(complex(forward_inner - adjoint_inner)) < 1e-12 * abs(
        complex(forward_inner)
    )


@pytest.mark.parametrize("angle", [0.0, 0.05, -0.3])
def test_windowed_rotation_angle_gradient_matches_finite_differences(
    float64_default, angle: float
) -> None:
    """The window is set from the angle without a gradient. It is one at every row
    frequency the transform samples, so the gradient through the shifts is exact.
    """
    field = _random_phase_field(32)
    generator = torch.Generator().manual_seed(2)
    weight = torch.randn(24, 24, generator=generator) + 1j * torch.randn(
        24, 24, generator=generator
    )

    def loss(value: torch.Tensor) -> torch.Tensor:
        transform = ChirpZPartialAffine((32, 32), (24, 24), (1.7, 1.7), angle=value)
        return (transform.forward(field) * weight).real.sum()

    parameter = torch.tensor(angle, requires_grad=True)
    loss(parameter).backward()
    step = 1e-5
    numeric = (
        loss(torch.tensor(angle + step)) - loss(torch.tensor(angle - step))
    ) / (2 * step)

    torch.testing.assert_close(parameter.grad, numeric.detach(), rtol=1e-6, atol=1e-9)


# %% An output region of interest
def test_an_output_roi_samples_the_whole_grid_there() -> None:
    """The rectangle is a slice of the uniformly spaced output frequencies, so it holds
    the values of the whole grid there, turned and shifted alike.
    """
    field = _random_field(48)
    roi = ROI(10, 17, 12, 20)
    for angle in (0.0, 0.15):
        whole = ChirpZPartialAffine((48, 48), (60, 72), (1.7, 1.7), (0.3, -0.2), angle)
        window = ChirpZPartialAffine(
            (48, 48), (60, 72), (1.7, 1.7), (0.3, -0.2), angle, output_roi=roi
        )
        output = window.forward(field)
        assert tuple(output.shape) == (12, 20)
        assert _relative_error(output, roi.crop(whole.forward(field))) < 1e-3


def test_an_output_roi_keeps_the_adjoint_its_transpose() -> None:
    torch.manual_seed(1)
    field = _random_field(32)
    samples = (torch.randn(10, 14) + 1j * torch.randn(10, 14)).to(torch.complex64)
    czt = ChirpZPartialAffine(
        (32, 32), (40, 40), (1.3, 1.3), angle=0.1, output_roi=ROI(5, 9, 10, 14)
    )
    forward_inner = (czt.forward(field) * samples.conj()).sum()
    adjoint_inner = (field * czt.adjoint(samples).conj()).sum()
    assert abs(complex(forward_inner - adjoint_inner)) < 1e-3 * abs(
        complex(forward_inner)
    )


def test_an_output_roi_outside_the_grid_is_refused() -> None:
    with pytest.raises(ValueError, match="output region of interest"):
        ChirpZPartialAffine(
            (32, 32), (40, 40), (1.0, 1.0), output_roi=ROI(35, 0, 10, 10)
        )
