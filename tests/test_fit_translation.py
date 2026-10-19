"""fit_translation: the translation, scale and offset that fit a pattern to a frame."""

from __future__ import annotations

import pytest
import torch

from hologradpy.analysis.fitting import fit_scale, fit_translation
from hologradpy.fourier_transforms import cross_correlation_peak, fft_translate
from hologradpy.grids import get_spatial_grid
from hologradpy.profiles.amplitude import super_gaussian

RESOLUTION = (64, 64)


def _pattern() -> torch.Tensor:
    """A flat top with a soft edge, well inside the frame."""
    grid = get_spatial_grid(RESOLUTION, (1.0, 1.0))
    return super_gaussian(*grid, 0.0, 0.0, 6, 6, 12.0, 9.0).double()


def _region() -> torch.Tensor:
    region = torch.zeros(RESOLUTION, dtype=torch.bool)
    region[8:56, 8:56] = True
    return region


def test_recovers_a_subpixel_translation_and_scale() -> None:
    pattern = _pattern()
    shift = (1.3, -2.7)
    generator = torch.Generator().manual_seed(0)
    noise = torch.randn(RESOLUTION, generator=generator, dtype=torch.float64)
    frame = 3.0 * fft_translate(pattern, shift).real + 0.01 * noise

    fit = fit_translation(frame, pattern, _region())

    assert fit.shift == pytest.approx(shift, abs=0.01)
    assert fit.scale == pytest.approx(3.0, rel=1e-2)
    assert fit.offset == 0.0


def test_recovers_an_offset_on_request() -> None:
    pattern = _pattern()
    shift = (1.3, -2.7)
    generator = torch.Generator().manual_seed(0)
    noise = torch.randn(RESOLUTION, generator=generator, dtype=torch.float64)
    frame = 3.0 * fft_translate(pattern, shift).real + 5.0 + 0.01 * noise

    fit = fit_translation(frame, pattern, _region(), fit_offset=True)

    assert fit.shift == pytest.approx(shift, abs=0.01)
    assert fit.scale == pytest.approx(3.0, rel=1e-2)
    assert fit.offset == pytest.approx(5.0, abs=0.02)


def test_finds_a_shift_of_many_samples() -> None:
    """The peak of the cross-correlation starts the steps near a large shift."""
    pattern = _pattern()
    shift = (7.0, -11.4)
    frame = fft_translate(pattern, shift).real

    fit = fit_translation(frame, pattern, _region())

    assert fit.shift == pytest.approx(shift, abs=0.01)


def test_light_outside_the_region_plays_no_part() -> None:
    """The bright spot outside the region represents the zeroth order on a bench."""
    pattern = _pattern()
    frame = fft_translate(pattern, (0.4, 0.6)).real
    frame[0:4, 0:4] = 100.0

    fit = fit_translation(frame, pattern, _region())

    assert fit.shift == pytest.approx((0.4, 0.6), abs=0.01)


def test_the_cross_correlation_peak_is_the_whole_sample_shift() -> None:
    """In an integer image, a fractional shift is found at the nearest whole sample,
    with its sign.
    """
    pattern = _pattern()
    frame = fft_translate(pattern, (5.0, -9.4)).real

    peak = cross_correlation_peak(
        (100 * frame).round().to(torch.int32), pattern, _region()
    )

    assert peak == (5.0, -9.0)


def test_a_pattern_moved_3_columns_right_and_1_row_down_fits_3_1() -> None:
    """The shift is ``(columns, rows)``, the order ``fft_translate`` takes.

    The pattern is wider than it is tall, and ``torch.roll`` moves it, so the order is
    pinned without ``fft_translate``.
    """
    pattern = _pattern()
    frame = torch.roll(pattern, shifts=(1, 3), dims=(-2, -1))

    fit = fit_translation(frame, pattern, _region())

    assert fit.shift == pytest.approx((3.0, 1.0), abs=0.01)


def test_a_pattern_without_structure_is_refused() -> None:
    """A dark pattern has no scale, and a constant one no scale beside an offset."""
    with pytest.raises(ValueError, match="constant"):
        fit_scale(torch.rand(RESOLUTION), torch.zeros(RESOLUTION), _region())
    with pytest.raises(ValueError, match="constant"):
        fit_scale(
            torch.rand(RESOLUTION), torch.ones(RESOLUTION), _region(), fit_offset=True
        )
