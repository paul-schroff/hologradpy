"""Opt-in recording of per-forward values for OpticsModules.

A mix-in that gives any :class:`torch.nn.Module` a toggleable recording buffer. While
recording is on, a persistent forward hook captures the named tensors a module declares
in :meth:`RecordingMixin.recordables` -- one row per forward pass -- so callers can
inspect quantities a module produces (sampled random values, learnable parameters,
intermediate results) without wrapping the module or the caller.

Mix it in *before* ``nn.Module`` so ``nn.Module.__init__`` runs first and the forward
hook can be registered, e.g. ``class Foo(RecordingMixin, nn.Module)``. Subclasses
declare what to capture by overriding :meth:`RecordingMixin.recordables`; the default
records nothing (so a module that never overrides it costs one boolean check per
forward).
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

import torch
from torch import Tensor


class RecordingMixin:
    """Adds ``record`` / ``record_samples`` / ``history`` to an ``nn.Module``.

    Subclasses override :meth:`recordables` to declare the named tensors to capture
    each forward; the base records nothing. Enable recording with :meth:`record` (or
    scope it with :meth:`record_samples`) and read the stacked result from
    :attr:`history`.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._recording: bool = False
        self._history: dict[str, list[Tensor]] = {}
        # Persistent forward hook: appends the declared recordables each forward while
        # recording is on (a no-op otherwise).
        self.register_forward_hook(RecordingMixin._record_forward)

    def recordables(self) -> dict[str, Tensor]:
        """Named tensors to capture each forward while recording is on. Override to
        declare them (e.g. ``{"angle": ...}``); the base records nothing."""
        return {}

    @staticmethod
    def _record_forward(module: RecordingMixin, inputs, output) -> None:
        if not module._recording:
            return
        for name, value in module.recordables().items():
            module._history.setdefault(name, []).append(value.detach())

    def record(self, enabled: bool = True) -> None:
        """Toggle recording of the per-forward :meth:`recordables`.

        Enabling clears any previously recorded history, so each recording starts over. 
        Read the result from :attr:`history`.
        """
        self._recording = enabled
        if enabled:
            self._history = {}

    @contextmanager
    def record_samples(self) -> Iterator[RecordingMixin]:
        """Record :meth:`recordables` for the duration of the ``with`` block
        (recording is turned off again on exit). Read them from :attr:`history`."""
        self.record(True)
        try:
            yield self
        finally:
            self.record(False)

    @property
    def history(self) -> dict[str, Tensor]:
        """Everything recorded, as ``{name: (n, ...) tensor}`` (one row per forward).
        Empty ``{}`` if nothing was recorded."""
        return {
            name: torch.stack(values)
            for name, values in self._history.items()
            if values
        }
