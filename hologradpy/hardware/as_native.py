"""Coerce any camera / SLM instance to the native interface.

``as_camera`` / ``as_slm`` take an already existing device and return it as a native
device: a native device (a :class:`~hologradpy.hardware.camera.Camera` /
:class:`~hologradpy.hardware.slm.SLM` subclass, including the simulator) passes through,
and a registered driver family is wrapped in its adapter.

This layer is backend-agnostic: it knows nothing about any specific driver. A backend
plugs itself in by registering an adapter, e.g.
``as_camera.register(SomeDriver)(MyAdapter)`` (the slmsuite bridge in
:mod:`hologradpy.hardware.slmsuite` does exactly that on import). Construction from a
driver class or a registered backend name lives in
:mod:`hologradpy.hardware.factory`.
"""

from __future__ import annotations

from functools import singledispatch

from .camera import Camera
from .slm import SLM


@singledispatch
def as_camera(device) -> Camera:
    """Return a native-camera interface for ``device`` (idempotent).

    Dispatches on the device type so consumers accept *any* device and normalise
    here, and the caller never builds an adapter by hand. A native device (a
    :class:`~hologradpy.hardware.camera.Camera` subclass, including the simulator) is
    passed through. A registered driver family is wrapped in its adapter. Teach it a
    new backend with ``as_camera.register(SomeDriver)(MyAdapter)``.
    """
    raise TypeError(
        f"No native camera for {type(device).__name__!r}. Register one with "
        "as_camera.register(...) or subclass hardware.camera.Camera."
    )


@as_camera.register(Camera)
def _(device):
    return device  # Already native (a Camera subclass or an adapter).


@singledispatch
def as_slm(device) -> SLM:
    """Return a native-SLM interface for ``device`` (see :func:`as_camera`)."""
    raise TypeError(
        f"No native SLM for {type(device).__name__!r}. Register one with "
        "as_slm.register(...) or subclass hardware.slm.SLM."
    )


@as_slm.register(SLM)
def _(device):
    return device  # Already native (an SLM subclass or an adapter).
