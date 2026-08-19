"""OpticsModule, the base class every optical element subclasses (backend core)."""

from __future__ import annotations
import functools
import inspect
from typing import Callable, TypedDict, Any  # , Self

import torch
from torch import nn, Tensor

from ..complex_amplitude import ComplexAmplitude, FieldGeometry
from ...grids import get_spatial_grid
from .recording import RecordingMixin


class SaveDict(TypedDict):
    """What a checkpoint of one module holds.

    ``class_name`` identifies what wrote it, so a state dict cannot be loaded into an
    unrelated module. ``kwargs`` are the constructor arguments :func:`capture_init`
    recorded, so the file can be reopened.
    """

    class_name: str
    kwargs: dict[str, Any]
    state_dict: dict[str, Any]
    input_geometry: FieldGeometry
    resolution_out: tuple[int, int]
    pixel_size_out: Tensor


def capture_init(init: Callable[..., None]) -> Callable[..., None]:
    """Decorator for a concrete ``__init__`` that records the bound constructor
    arguments into ``self._init_kwargs``.

    Shared by :class:`OpticsModule` and
    :class:`~hologradpy.optics.systems.OpticalSystem`: both need to rebuild an object
    from a checkpoint, and neither should carry a hand-written list of what its
    constructor took. The original ``__init__`` runs first, so ``nn.Module`` is fully
    initialized before anything is recorded.

    A ``**kwargs`` parameter is flattened into the recorded arguments, so passing them
    back as keywords reproduces the original call. ``*args`` cannot be reproduced by
    keyword and is therefore not recorded.
    """
    signature = inspect.signature(init)

    @functools.wraps(init)
    def wrapper(self, *args, **kwargs) -> None:
        init(self, *args, **kwargs)
        bound = signature.bind(self, *args, **kwargs)
        bound.apply_defaults()

        captured: dict[str, Any] = {}
        for name, value in bound.arguments.items():
            if name == "self":
                continue
            kind = signature.parameters[name].kind
            if kind is inspect.Parameter.VAR_KEYWORD:
                captured.update(value)
            elif kind is not inspect.Parameter.VAR_POSITIONAL:
                captured[name] = value
        self._init_kwargs = captured

    return wrapper


def _check_checkpoint_class(state: dict, expected: type, path: str) -> None:
    """Refuse a checkpoint written by a different module class.

    Mirrors what :meth:`hologradpy.optics.systems.OpticalSystem.load` does for a whole
    system. A checkpoint written before the class name was recorded carries none, and is
    accepted.
    """
    written_by = state.get("class_name")
    if written_by is not None and written_by != expected.__name__:
        raise TypeError(
            f"{path} was saved from a {written_by}, not a {expected.__name__}."
        )


def _auto_init_forward(forward: Callable) -> Callable:
    """Wrap a subclass ``forward`` so the module lazily initializes itself from
    its input field on first use (replaces the old forward pre-hook).
    """

    @functools.wraps(forward)
    def wrapper(self, complex_amplitude, *args, **kwargs):
        if not self.initialized:
            self._lazy_initialize(complex_amplitude)
        return forward(self, complex_amplitude, *args, **kwargs)

    wrapper._optics_wrapped = True
    return wrapper


def _auto_init_adjoint(adjoint: Callable) -> Callable:
    """Wrap a subclass ``adjoint`` so it requires the module to be initialized
    first -- its input is an output-plane field, so it cannot infer the input
    geometry and must have seen a forward() / initialize_from_geometry().
    """

    @functools.wraps(adjoint)
    def wrapper(self, complex_amplitude, *args, **kwargs):
        self._ensure_initialized()
        return adjoint(self, complex_amplitude, *args, **kwargs)

    wrapper._optics_wrapped = True
    return wrapper


class OpticsModule(RecordingMixin, nn.Module):
    """Base class for an optical element (a differentiable field transform).

    Subclassing contract:

    - Implement ``forward(field) -> field`` and, if invertible, ``adjoint(field)``.
    - Initialization is **automatic**: the module lazily initializes from the input
      field on the first ``forward``; ``adjoint`` requires a prior ``forward`` (or
      ``initialize_from_geometry``) because its input carries the *output*-plane
      geometry. You never call an init/guard yourself.
    - Build any device/geometry-dependent state (buffers, Parameters, transforms)
      by overriding ``lazy_init(field)`` -- it runs once, before the first
      forward/adjoint, with the input geometry available via ``pixel_size_in`` /
      ``resolution_in`` / ``input_geometry``.
    - Output sampling defaults to the input (sampling-preserving). To change it,
      pass ``pixel_size_out`` / ``resolution_out`` to ``super().__init__()`` (when
      known at construction) or call ``self.set_output_geometry(...)`` in
      ``lazy_init``.
    """

    def __init_subclass__(cls, **kwargs) -> None:
        # Auto-wrap the subclass's forward/adjoint so initialization is automatic
        # and symmetric. Only wrap methods the subclass defines itself, and never
        # double-wrap.
        super().__init_subclass__(**kwargs)
        forward = cls.__dict__.get("forward")
        if forward is not None and not getattr(forward, "_optics_wrapped", False):
            cls.forward = _auto_init_forward(forward)
        adjoint = cls.__dict__.get("adjoint")
        if adjoint is not None and not getattr(adjoint, "_optics_wrapped", False):
            cls.adjoint = _auto_init_adjoint(adjoint)

    def __init__(
        self,
        pixel_size_out: tuple[float, float] | None = None,
        resolution_out: tuple[int, int] | None = None,
    ) -> None:
        """
        Args:
            pixel_size_out: Output pixel size in metres ``(height, width)`` when known
                at construction. Defaults to None, which preserves the input sampling,
                or set it later with :meth:`set_output_geometry`.
            resolution_out: Output resolution in pixels ``(height, width)`` when known
                at construction. Defaults to None.
        """
        super().__init__()
        self._pixel_size_out_init = pixel_size_out
        self._resolution_out: tuple[int, int] | None = resolution_out
        self._input_geometry: FieldGeometry | None = None
        self._pixel_size_out: Tensor | None = None
        self.initialized = False

    def lazy_init(self, complex_amplitude: ComplexAmplitude) -> None:
        """Build device/geometry-dependent state (buffers, Parameters, transforms).

        Called once, automatically, before the first forward()/adjoint(), with the
        input geometry available via self.pixel_size_in / resolution_in /
        input_geometry. The output geometry is already set (to the input, or to the
        pixel_size_out/resolution_out passed at construction); call
        self.set_output_geometry(...) here to change it. The base builds nothing.
        """
        return

    def set_output_geometry(
        self,
        resolution: tuple[int, int] | None = None,
        pixel_size: tuple[float, float] | Tensor | None = None,
    ) -> None:
        """Declare the module's output-plane sampling from within lazy_init().

        Call this when the module changes resolution and/or pixel size; omitted
        arguments keep the current default (the input geometry). ``pixel_size`` may
        be a ``(height, width)`` tuple or a tensor.
        """
        if resolution is not None:
            self._resolution_out = resolution
        if pixel_size is not None:
            if isinstance(pixel_size, Tensor):
                self._pixel_size_out = pixel_size
            else:
                self._pixel_size_out = torch.tensor(
                    pixel_size,
                    device=self.pixel_size_in.device,
                    dtype=self.pixel_size_in.dtype,
                )

    def _set_default_output_geometry(self) -> None:
        """Default output geometry: the values passed at construction, else the
        input geometry (sampling-preserving). Runs before lazy_init().
        """
        if self._pixel_size_out_init is None:
            self._pixel_size_out = self.pixel_size_in
        else:
            self._pixel_size_out = torch.tensor(
                self._pixel_size_out_init,
                device=self.pixel_size_in.device,
                dtype=self.pixel_size_in.dtype,
            )
        if self._resolution_out is None:
            self._resolution_out = self.resolution_in

    def _finalize_output_geometry(self) -> None:
        """Validate the output geometry is set and broadcast the output pixel size
        across wavelengths.
        """
        if self._resolution_out is None or self._pixel_size_out is None:
            raise ValueError(
                f"{type(self).__name__}: output geometry is unset after "
                "lazy_init(). Pass pixel_size_out/resolution_out to __init__ or "
                "call set_output_geometry() in lazy_init()."
            )
        if self._pixel_size_out.ndim == 1:
            self._pixel_size_out = self._pixel_size_out.unsqueeze(0)
        if (
            self._pixel_size_out.shape[0] == 1
            and self.input_geometry.number_of_wavelengths > 1
        ):
            self._pixel_size_out = self._pixel_size_out.repeat(
                self.input_geometry.number_of_wavelengths, 1
            )

    def _lazy_initialize(self, complex_amplitude: ComplexAmplitude) -> None:
        """Initialize from an input-plane field (or probe): record the input
        geometry, set the default output geometry, let the subclass build its state
        in lazy_init(), then validate + broadcast the output geometry.
        """
        self._input_geometry = complex_amplitude.geometry
        self.initialized = True
        self._set_default_output_geometry()
        self.lazy_init(complex_amplitude)
        self._finalize_output_geometry()

    def initialize_from_geometry(
        self,
        geometry: FieldGeometry,
        dtype: torch.dtype = torch.complex64,
    ) -> None:
        """Initialize the module from an input-plane geometry, without a
        forward pass.

        Lets ``forward()`` and ``adjoint()`` be called in any order: the
        adjoint receives an output-plane field that does not carry the
        input-plane geometry, so the module must be told it explicitly here.

        Args:
            geometry: Input-plane (e.g. SLM-plane) field geometry.
            dtype: Complex dtype of the fields the module will process.
        """
        if self.initialized:
            return

        number_of_wavelengths = geometry.number_of_wavelengths
        shape = (
            geometry.resolution
            if number_of_wavelengths == 1
            else (number_of_wavelengths, *geometry.resolution)
        )
        # Probe carries only geometry/dtype/device into lazy_init, which never
        # reads field values, so the (uninitialized) contents are unused.
        probe = ComplexAmplitude(
            torch.empty(shape, dtype=dtype, device=geometry.wavelength.device),
            geometry.wavelength,
            geometry.pixel_size,
        )
        self._lazy_initialize(probe)

    def _ensure_initialized(self) -> None:
        """Raise if the module has not been lazily initialized yet."""
        if not self.initialized:
            raise RuntimeError(
                f"{type(self).__name__} must be initialized before this "
                "call; run forward() once or call "
                "initialize_from_geometry(input_geometry)."
            )

    @property
    def is_stochastic(self) -> bool:
        """Whether :meth:`forward` draws randomness, so two passes can differ. Anything
        that draws once in :meth:`lazy_init` into a registered buffer stays False, since
        the draw is then part of the state dict.
        """
        return False

    @property
    def input_geometry(self) -> FieldGeometry:
        if not self.initialized:
            raise ValueError(
                "OpticsModule.input_geometry is not available until the first "
                "forward() call."
            )
        return self._input_geometry

    @property
    def pixel_size_in(self) -> Tensor:
        """Input pixel size in metres.

        Returns:
            Tensor: Input pixel size (height, width).
        """
        return self.input_geometry.pixel_size

    @property
    def resolution_in(self) -> tuple[int, int]:
        """Input resolution in pixels.

        Returns:
            tuple[int, int]: Input resolution (height, width).
        """
        return self.input_geometry.resolution

    @property
    def pixel_size_out(self) -> Tensor:
        """Output pixel size in metres.

        Note: this is ``None`` until the module is lazily initialized (the tensor
        is built in ``lazy_init``). This is intentional and relied upon -- callers
        that need it before a forward pass fall back to the constructor value
        ``_pixel_size_out_init`` (see ``SimulatedCameraTorch``).

        Returns:
            Tensor: Output pixel size (height, width), or ``None`` before init.
        """
        return self._pixel_size_out

    @property
    def resolution_out(self) -> tuple[int, int]:
        """Output resolution in pixels.

        Returns:
            tuple[int, int]: Output resolution (height, width).
        """
        return self._resolution_out

    def forward(self, complex_amplitude: ComplexAmplitude) -> ComplexAmplitude:
        raise NotImplementedError(
            "Subclasses of OpticsModule must implement forward() method."
        )

    def adjoint(self, complex_amplitude: ComplexAmplitude) -> ComplexAmplitude:
        """Adjoint propagation from output plane back to input plane."""
        raise NotImplementedError(
            "Subclasses of OpticsModule must implement adjoint() method."
        )

    def save(self, path: str) -> None:
        save_dict: SaveDict = {
            "class_name": type(self).__name__,
            "kwargs": getattr(self, "_init_kwargs", None),
            "state_dict": self.state_dict(),
            "input_geometry": self.input_geometry,
            "resolution_out": self.resolution_out,
            "pixel_size_out": self.pixel_size_out,
        }
        torch.save(save_dict, path)

    def load_weights(self, path: str) -> None:
        """Load weights from a checkpoint written by :meth:`save`.

        Raises:
            TypeError: The checkpoint was written by a different class.
        """
        state = torch.load(path, weights_only=False)
        _check_checkpoint_class(state, type(self), path)
        self.load_state_dict(state["state_dict"])

    @classmethod
    def from_file(cls, path: str, device: torch.device = "cpu") -> OpticsModule:
        """Rebuild a module saved by :meth:`save`.

        The constructor arguments are replayed, the module is initialized from the input
        geometry the checkpoint recorded (so its lazily built state exists), and the
        weights are loaded on top.

        Raises:
            TypeError: The checkpoint was written by a different class.
            NotImplementedError: The checkpoint holds no constructor arguments, because
                the class does not wear :func:`capture_init`.
        """
        state: SaveDict = torch.load(path, map_location=device, weights_only=False)
        _check_checkpoint_class(state, cls, path)

        kwargs = state.get("kwargs")
        if kwargs is None:
            raise NotImplementedError(
                f"{path} holds no constructor arguments, so a {cls.__name__} cannot be "
                f"rebuilt from it. Decorate {cls.__name__}.__init__ with @capture_init "
                "and save it again."
            )

        # map_location has already moved the geometry's tensors, so the module builds
        # its lazy state on the device the caller asked for.
        module = cls(**kwargs)
        module.initialize_from_geometry(state["input_geometry"])
        module.load_state_dict(state["state_dict"])
        return module

    def get_spatial_grid_input(self, index: int = 0) -> tuple[Tensor, Tensor]:
        return get_spatial_grid(
            resolution=self.resolution_in,
            pixel_size=self.pixel_size_in.tolist()[index],
            device=self.pixel_size_in.device,
        )

    def get_spatial_grid_output(self, index: int = 0) -> tuple[Tensor, Tensor]:
        return get_spatial_grid(
            resolution=self.resolution_out,
            pixel_size=self.pixel_size_out.tolist()[index],
            device=self.pixel_size_out.device,
        )
