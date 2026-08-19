"""Device construction: build a camera / SLM from a driver class or a registered name.

``open_camera`` / ``open_slm`` construct a device and coerce it to native in one call,
so ``open_camera(Driver, ...)`` is the one-call form of ``as_camera(Driver(...))``.
``register_camera_backend`` / ``register_slm_backend`` give a driver class a short name
so it can be opened by string (``open_camera("thorlabs", ...)``). The coercion itself
lives in :mod:`hologradpy.hardware.as_native`.
"""

from __future__ import annotations

import importlib
from typing import Any, Callable, ParamSpec, TypeVar, overload

from .camera import Camera
from .slm import SLM
from .as_native import as_camera, as_slm


# Each backend entry is a driver class, or a lazy ``"package.module:Attr"`` import spec
# resolved on first :func:`open_camera` / :func:`open_slm` so a driver's (heavy, often
# uninstalled) vendor SDK is imported only when that backend is actually opened.
_CAMERA_BACKENDS: dict[str, type | str] = {}
_SLM_BACKENDS: dict[str, type | str] = {}


def register_camera_backend(name: str, driver: type | str) -> None:
    """Register a camera driver under ``name`` for :func:`open_camera`.

    ``driver`` is the driver class, or a ``"package.module:Attr"`` (or plain dotted)
    import spec that is imported lazily on the first ``open_camera(name, ...)``, so a
    vendor SDK loads only when that backend is opened.
    """
    _CAMERA_BACKENDS[name] = driver


def register_slm_backend(name: str, driver: type | str) -> None:
    """Register an SLM driver under ``name`` for :func:`open_slm` (see
    :func:`register_camera_backend` for the lazy ``"module:Attr"`` spec form).
    """
    _SLM_BACKENDS[name] = driver


def available_camera_backends() -> list[str]:
    """Return the names :func:`open_camera` accepts, sorted.

    slmsuite's drivers appear here only after
    :func:`~hologradpy.hardware.slmsuite.backends.register_slmsuite_backends`, since
    that registration is opt-in.
    """
    return sorted(_CAMERA_BACKENDS)


def available_slm_backends() -> list[str]:
    """Return the names :func:`open_slm` accepts, sorted (see
    :func:`available_camera_backends` for the opt-in slmsuite caveat).
    """
    return sorted(_SLM_BACKENDS)


_P = ParamSpec("_P")
_CameraT = TypeVar("_CameraT", bound=Camera)
_SLMT = TypeVar("_SLMT", bound=SLM)


@overload
def open_camera(
    driver: Callable[_P, _CameraT], *args: _P.args, **kwargs: _P.kwargs
) -> _CameraT: ...
@overload
def open_camera(
    driver: Callable[_P, Any], *args: _P.args, **kwargs: _P.kwargs
) -> Camera: ...
@overload
def open_camera(driver: str, *args: Any, **kwargs: Any) -> Camera: ...
def open_camera(driver: Callable[..., Any] | str, *args: Any, **kwargs: Any) -> Camera:
    """Construct a camera and return it as a native HoloGradPy device, in one call.

    ``driver`` is a camera driver class (whose constructor signature is preserved for
    editors and type checkers), or a name registered with
    :func:`register_camera_backend`. It is built with ``*args`` / ``**kwargs`` and
    normalized via :func:`~hologradpy.hardware.as_native.as_camera`, the one-call
    form of ``as_camera(Driver(...))``.

    A driver that is already a native :class:`~hologradpy.hardware.camera.Camera`
    subclass (such as :class:`SimulatedCameraTorch`) passes through ``as_camera``
    unchanged, so the concrete subclass type is preserved for the caller. A raw
    slmsuite driver is wrapped in an adapter, so it comes back typed as the base
    ``Camera``, and a name lookup returns the base ``Camera`` as well.
    """
    driver = _resolve_backend(driver, _CAMERA_BACKENDS, "camera")
    return as_camera(driver(*args, **kwargs))


@overload
def open_slm(
    driver: Callable[_P, _SLMT], *args: _P.args, **kwargs: _P.kwargs
) -> _SLMT: ...
@overload
def open_slm(
    driver: Callable[_P, Any], *args: _P.args, **kwargs: _P.kwargs
) -> SLM: ...
@overload
def open_slm(driver: str, *args: Any, **kwargs: Any) -> SLM: ...
def open_slm(driver: Callable[..., Any] | str, *args: Any, **kwargs: Any) -> SLM:
    """Construct an SLM and return it as a native HoloGradPy device (see
    :func:`open_camera`, including how a native subclass keeps its concrete type
    while a wrapped slmsuite driver comes back as the base ``SLM``).
    """
    driver = _resolve_backend(driver, _SLM_BACKENDS, "SLM")
    return as_slm(driver(*args, **kwargs))


def _resolve_backend(
    driver: Callable[..., Any] | str, registry: dict[str, type | str], kind: str
) -> Callable[..., Any]:
    """Resolve a driver argument to a driver class.

    A class is returned as is. A registered ``name`` is looked up, and a registry entry
    that is a ``"module:Attr"`` import spec is imported lazily (so the vendor SDK loads
    only here, on the actual open).
    """
    if isinstance(driver, str):
        if driver not in registry:
            raise KeyError(
                f"Unknown {kind} backend {driver!r}. Registered: {sorted(registry)}."
            )
        entry = registry[driver]
    else:
        entry = driver
    if isinstance(entry, str):
        return _import_spec(entry, driver, kind)
    return entry


def _import_spec(spec: str, name: str, kind: str) -> type:
    """Import a ``"package.module:Attr"`` (or plain dotted) spec to its class."""
    module_path, separator, attribute = spec.partition(":")
    if not separator:  # plain dotted path: split off the last component
        module_path, _, attribute = spec.rpartition(".")
    try:
        module = importlib.import_module(module_path)
    except ImportError as error:
        raise ImportError(
            f"{kind.capitalize()} backend {name!r} needs {module_path!r}, which failed "
            f"to import ({error}). Its vendor SDK is likely not installed."
        ) from error
    try:
        return getattr(module, attribute)
    except AttributeError as error:
        raise AttributeError(
            f"{kind.capitalize()} backend {name!r}: {module_path!r} has no "
            f"{attribute!r} (the driver may have been renamed)."
        ) from error
