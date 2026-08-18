"""Sub-sample translation by the Fourier shift theorem.

What it is for: a target whose distance from the zeroth order matters cannot be placed
by rounding to the nearest sample, which is wrong by up to half of one. A phase ramp in
the Fourier domain moves it by any fraction, exactly for a band-limited pattern.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from hologradpy.fourier_transforms import fft_translate, translate_intensity


def _gaussian(size: int = 64, center: float | None = None, width: float = 60.0):
    """A compact, smooth pattern: what the shift theorem is exact for."""
    center = (size - 1) / 2 if center is None else center
    axis = torch.arange(float(size))
    return torch.exp(-(((axis[:, None] - center) ** 2 + (axis[None, :] - center) ** 2)
                       / width))


def _centroid_column(image) -> float:
    image = np.asarray(image, dtype=float)
    return float((image.sum(0) * np.arange(image.shape[-1])).sum() / image.sum())


def test_a_whole_sample_shift_is_a_roll() -> None:
    """The fractional case is the point, but the whole-sample case has a known answer,
    so it pins the sign convention: positive moves towards increasing index."""
    spike = torch.zeros(32, 32)
    spike[10, 12] = 1.0

    shifted = fft_translate(spike, (3, -2)).real

    expected = torch.roll(torch.roll(spike, 3, 0), -2, 1)
    assert torch.allclose(shifted, expected, atol=1e-5)


def test_a_band_limited_pattern_shifts_exactly() -> None:
    """No resampling kernel, so no interpolation error to bound: for a pattern the
    sampling actually resolves, the shift is the identity up to round-off."""
    size = 64
    shift = (0.25, 0.5)
    smooth = _gaussian(size)
    axis = torch.arange(float(size))
    center = (size - 1) / 2
    expected = torch.exp(
        -(((axis[:, None] - center - shift[0]) ** 2
           + (axis[None, :] - center - shift[1]) ** 2) / 60)
    )

    assert torch.allclose(fft_translate(smooth, shift).real, expected, atol=1e-6)


@pytest.mark.parametrize("fraction", [0.1, 0.25, 0.5, 0.75, 0.9])
def test_a_smooth_target_lands_where_it_was_asked_to(fraction: float) -> None:
    """The measurement that matters: the pattern's position, not its samples."""
    smooth = _gaussian()
    moved = _centroid_column(translate_intensity(smooth, (0.0, fraction)))

    assert moved - _centroid_column(smooth) == pytest.approx(fraction, abs=1e-3)


@pytest.mark.parametrize("fraction", [0.1, 0.25, 0.5, 0.75, 0.9])
@pytest.mark.parametrize("width", [7, 8])
def test_even_a_hard_edge_lands_where_it_was_asked_to(
    fraction: float, width: int
) -> None:
    """A top hat is not band limited and its edges ring, but going through the
    amplitude keeps the pattern where it was put, to a few hundredths of a sample
    against the half a sample rounding would cost.

    Both parities: a hat of even width straddles a sample boundary and one of odd width
    is centered on a sample, and the two land to different accuracies (roughly 1e-3 and
    3e-2). Testing only the flattering one would pin a number that is about the pattern
    rather than about the method.
    """
    hat = torch.zeros(64, 64)
    start = 32 - width // 2
    hat[start : start + width, start : start + width] = 1.0

    shifted = translate_intensity(hat, (0.0, fraction))
    moved = _centroid_column(shifted) - _centroid_column(hat)

    assert moved == pytest.approx(fraction, abs=0.05)
    assert abs(moved - fraction) < abs(round(fraction) - fraction)


@pytest.mark.parametrize("fraction", [0.25, 0.5])
def test_moving_an_intensity_does_not_change_how_much_there_is(
    fraction: float,
) -> None:
    """Squaring a magnitude cannot go negative, so the ringing needs no clipping. Were
    the intensity moved directly and its overshoot clipped away, a top hat would gain
    several percent of energy from being put half a sample to the left."""
    hat = torch.zeros(64, 64)
    hat[28:36, 28:36] = 1.0

    moved = translate_intensity(hat, (0.0, fraction))

    assert float(moved.sum()) == pytest.approx(float(hat.sum()), rel=1e-4)


def test_shifting_back_returns_the_pattern() -> None:
    """A translation is unitary, so it has an exact inverse. Rounding does not."""
    smooth = _gaussian()
    there = fft_translate(smooth, (0.3, -0.7))
    back = fft_translate(there, (-0.3, 0.7))

    assert torch.allclose(back.real, smooth, atol=1e-6)


def test_no_shift_changes_nothing() -> None:
    """Asked for nothing, it does nothing, rather than paying a transform to return
    almost the same numbers."""
    smooth = _gaussian()
    assert torch.equal(fft_translate(smooth, (0.0, 0.0)).real, smooth)


def test_an_intensity_stays_non_negative() -> None:
    """Not by clipping, but because a squared magnitude has nowhere else to be."""
    hat = torch.zeros(48, 48)
    hat[20:28, 20:28] = 1.0

    # Moving the intensity itself would ring below zero.
    assert fft_translate(hat, (0.0, 0.5)).real.min() < 0
    # Moving its amplitude and squaring cannot.
    assert translate_intensity(hat, (0.0, 0.5)).min() >= 0.0



def test_a_pattern_is_not_wrapped_around_the_frame() -> None:
    """The transform is cyclic, so without padding a pattern near one edge would
    reappear at the other."""
    edge = torch.zeros(32, 32)
    edge[:, -1] = 1.0

    shifted = translate_intensity(edge, (0.0, 2.0))

    # Nothing but float32 round-off arrives at the opposite edge, against a source of 1.
    assert shifted[:, :4].max() < 1e-6
