"""Named shortcuts for slmsuite's drivers, so they can be opened by string.

:func:`register_slmsuite_backends` registers each slmsuite camera / SLM driver under a
short name as a lazy ``"module:Attr"`` import spec, so ``open_camera("thorlabs", ...)``
works without importing any vendor SDK until that backend is actually opened. It is
*opt-in*: call it once (e.g. in your setup) to enable the by-name form. Opening a driver
by class or instance never needs it.

The tables mirror slmsuite's ``hardware.cameras`` / ``hardware.slms`` layout, so they
drift if slmsuite renames a driver module or class (:func:`open_camera` then raises a
clear ImportError / AttributeError pointing at the offending backend).
"""

from ..factory import register_camera_backend, register_slm_backend

# name -> "module:class" import spec (imported lazily on first open).
_SLMSUITE_CAMERAS: dict[str, str] = {
    "thorlabs": "slmsuite.hardware.cameras.thorlabs:ThorCam",
    "basler": "slmsuite.hardware.cameras.basler:Basler",
    "flir": "slmsuite.hardware.cameras.flir:FLIR",
    "alliedvision": "slmsuite.hardware.cameras.alliedvision:AlliedVision",
    "imagingsource": "slmsuite.hardware.cameras.imagingsource:ImagingSource",
    "mindvision": "slmsuite.hardware.cameras.mindvision:MindVision",
    "instrumental": "slmsuite.hardware.cameras.instrumental:Instrumental",
    "pylablib": "slmsuite.hardware.cameras.pylablib:PyLabLib",
    "mmcore": "slmsuite.hardware.cameras.mmcore:MMCore",
    "xenics": "slmsuite.hardware.cameras.xenics:Cheetah640",
    "webcam": "slmsuite.hardware.cameras.webcam:Webcam",
}

_SLMSUITE_SLMS: dict[str, str] = {
    "hamamatsu": "slmsuite.hardware.slms.hamamatsu:Hamamatsu",
    "holoeye": "slmsuite.hardware.slms.holoeye:Holoeye",
    "meadowlark": "slmsuite.hardware.slms.meadowlark:Meadowlark",
    "santec": "slmsuite.hardware.slms.santec:Santec",
    "screenmirrored": "slmsuite.hardware.slms.screenmirrored:ScreenMirrored",
    "texasinstruments": "slmsuite.hardware.slms.texasinstruments:PLM",
}


def register_slmsuite_backends() -> None:
    """Register every slmsuite driver under its short name for
    :func:`~hologradpy.hardware.factory.open_camera` /
    :func:`~hologradpy.hardware.factory.open_slm`.

    Opt-in and lazy: call it once, then e.g. ``open_camera("thorlabs", serial=...)``.
    No vendor SDK is imported until a given backend is actually opened.
    """
    for name, spec in _SLMSUITE_CAMERAS.items():
        register_camera_backend(name, spec)
    for name, spec in _SLMSUITE_SLMS.items():
        register_slm_backend(name, spec)
