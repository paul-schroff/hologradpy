"""The progress helpers in :mod:`hologradpy.utils`.

Both exist so a long loop can show a bar without the loop having to know whether one is
wanted, so what matters is that turning the bar off changes nothing except the output.
"""

from __future__ import annotations

import pytest

from hologradpy.utils import ProgressBar, progress


def test_progress_yields_every_element_either_way() -> None:
    assert list(progress(range(5), verbose=False)) == [0, 1, 2, 3, 4]
    assert list(progress(range(5), verbose=True, description="counting")) == [
        0,
        1,
        2,
        3,
        4,
    ]


def test_progress_returns_the_iterable_untouched_when_off() -> None:
    """Not merely equal: the same object, so nothing is consumed or copied on the way
    through."""
    values = [3, 1, 4]
    assert progress(values, verbose=False) is values


def test_progress_bar_is_a_no_op_when_off() -> None:
    with ProgressBar(total=2, description="quiet", verbose=False) as bar:
        assert bar._bar is None
        bar.update(loss=1.0)
        bar.update()
    assert bar._bar is None


def test_progress_bar_updates_and_closes() -> None:
    """disable=False because pytest captures stderr, where tqdm would otherwise
    suppress the bar and leave its counter at zero. This is about the mechanics."""
    with ProgressBar(total=3, description="loud", verbose=True, disable=False) as bar:
        for value in (1e10, 1e8, 1e6):
            bar.update(loss=value)
        assert bar._bar.n == 3
    assert bar._bar is None


def test_progress_bar_closes_on_exception() -> None:
    """A search that raises should not leave a bar open and the terminal wedged."""
    bar = ProgressBar(total=3, description="raising", verbose=True)
    with pytest.raises(RuntimeError):
        with bar:
            bar.update()
            raise RuntimeError("optimiser blew up")
    assert bar._bar is None


def test_progress_bar_formats_numeric_postfix() -> None:
    """A loss running from 1e10 to 1e6 keeps a constant width, so the bar does not
    jitter."""
    with ProgressBar(total=1, verbose=True, disable=False) as bar:
        bar.update(loss=1.23456789e10, label="text")
        postfix = bar._bar.postfix
    assert "1.235e+10" in postfix
    assert "text" in postfix


# --- Suppression where a bar cannot render -----------------------------------------


def test_bars_are_left_for_tqdm_to_suppress() -> None:
    """Bars pass disable=None, which is tqdm's own non-terminal rule.

    tqdm then suppresses a text bar when the stream is not a terminal and keeps a widget
    bar regardless. Doing this ourselves is what the removed nested_progress_supported()
    was for, and tqdm already knows the answer.
    """
    with ProgressBar(total=2, verbose=True) as bar:
        assert bar._bar.disable is not None
