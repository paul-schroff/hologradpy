"""A virtual clock that stands in for the ``time`` module inside the slmsuite drivers.

slmsuite settles an SLM with ``time.sleep``. With the clock installed, time moves only
when a driver reads the clock or sleeps on it, so a settle wait is deterministic and
takes no real time.

The clock reaches a module through the module's global name ``time``, so it reaches
only modules that ``import time`` and call through that name.
:func:`install_virtual_clock` checks this for every patched module. A HoloGradPy module
that sleeps joins :data:`DRIVER_TIME_MODULES` on the same terms.
"""

from __future__ import annotations

import importlib
import time
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import pytest

# The driver modules that read or sleep on the time module, by import name.
DRIVER_TIME_MODULES: tuple[str, ...] = (
    "slmsuite.hardware.slms.slm",
    "slmsuite.hardware.cameras.camera",
)


class VirtualClock:
    """A clock that advances by a fixed tick on every reading and by the full length of
    every sleep.

    It stands in for the ``time`` module, so :meth:`time`, :meth:`perf_counter`,
    :meth:`monotonic` and :meth:`sleep` share one timeline. Every other name is read
    from the real ``time`` module.

    Args:
        tick_s: The advance of the clock on each reading, in seconds. It is positive,
            so a loop that waits on the clock ends.
        start_s: The reading of the clock in seconds before its first tick.

    Raises:
        ValueError: The tick is not positive.
    """

    def __init__(self, tick_s: float = 1e-3, start_s: float = 0.0) -> None:
        if not tick_s > 0.0:
            raise ValueError(f"The tick has to be positive, got {tick_s} s.")
        self.tick_s = float(tick_s)
        self._now_s = float(start_s)

    @property
    def now(self) -> float:
        """The current time in seconds, read without advancing the clock."""
        return self._now_s

    def advance(self, seconds: float) -> float:
        """Move the clock forward.

        Args:
            seconds: How far to move it, in seconds.

        Returns:
            float: The new time in seconds.

        Raises:
            ValueError: ``seconds`` is negative.
        """
        if seconds < 0:
            raise ValueError(f"A clock only moves forward, got {seconds} s.")
        self._now_s += float(seconds)
        return self._now_s

    def time(self) -> float:
        """Advance by one tick and return the new time, as ``time.time`` reads."""
        return self.advance(self.tick_s)

    def perf_counter(self) -> float:
        """Advance by one tick and return the new time, as ``time.perf_counter``."""
        return self.advance(self.tick_s)

    def monotonic(self) -> float:
        """Advance by one tick and return the new time, as ``time.monotonic``."""
        return self.advance(self.tick_s)

    def sleep(self, seconds: float) -> None:
        """Advance by ``seconds`` at once, as ``time.sleep`` waits.

        Args:
            seconds: The length of the sleep in seconds.

        Raises:
            ValueError: ``seconds`` is negative, which ``time.sleep`` also refuses.
        """
        self.advance(seconds)

    def __getattr__(self, name: str) -> Any:
        return getattr(time, name)


def install_virtual_clock(
    monkeypatch: pytest.MonkeyPatch,
    clock: VirtualClock,
    module_names: tuple[str, ...] = DRIVER_TIME_MODULES,
) -> None:
    """Replace the ``time`` module inside each driver module with ``clock``.

    ``monkeypatch`` undoes the replacement at the end of the test.

    Args:
        monkeypatch: The pytest fixture that records the patches.
        clock: The clock read by the drivers from here on.
        module_names: The modules to patch, by import name.

    Raises:
        AssertionError: A module holds something other than the ``time`` module under
            the name ``time``, so the clock cannot reach its calls.
    """
    for module_name in module_names:
        module = importlib.import_module(module_name)
        held = getattr(module, "time", None)
        if held is not time:
            raise AssertionError(
                f"{module_name} holds {held!r} under the name 'time', where the "
                "virtual clock expects the time module. Import time as a module "
                "there, or leave the module out of the patch."
            )
        monkeypatch.setattr(module, "time", clock)
