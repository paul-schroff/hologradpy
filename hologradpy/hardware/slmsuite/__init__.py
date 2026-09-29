"""slmsuite bridge: the adapters and conversions that make a real slmsuite ``Camera`` /
``SLM`` satisfy the native device interface, plus the registration that plugs slmsuite
into the backend-agnostic dispatch in :mod:`hologradpy.hardware.as_native`.

Importing this package registers the slmsuite handlers on
:func:`~hologradpy.hardware.as_native.as_camera` /
:func:`~hologradpy.hardware.as_native.as_slm` as a side effect, so a raw slmsuite device
is wrapped from then on.

Each call builds a new, cheap adapter around the raw device. Anything loaded or set
through an adapter is stored on the raw device, so every adapter of one device shares
it. An SLM shares its corrections, its phase response and its resolution. A camera
shares its region of interest, its excluded pixels, its frame timeout and whether its
last frame was overexposed.
"""

from __future__ import annotations

from slmsuite.hardware.cameras.camera import Camera as SLMSuiteCamera
from slmsuite.hardware.slms.slm import SLM as SLMSuiteSLM

from ..as_native import as_camera, as_slm
from .adapter import SLMSuiteCameraAdapter, SLMSuiteSLMAdapter
from .backends import register_slmsuite_backends


# Plug slmsuite into the native dispatch: a raw slmsuite device is wrapped in its
# adapter, while native devices keep passing through untouched.
as_camera.register(SLMSuiteCamera)(SLMSuiteCameraAdapter)
as_slm.register(SLMSuiteSLM)(SLMSuiteSLMAdapter)

__all__ = [
    "SLMSuiteCameraAdapter",
    "SLMSuiteSLMAdapter",
    "register_slmsuite_backends",
]
