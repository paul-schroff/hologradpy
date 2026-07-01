from __future__ import annotations
import functools
import inspect
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TypeVar, Type

import torch
import torch.nn as nn
from torch import Tensor

from ..virtual_slms.abstract import VirtualSLM

from ..optics_module import OpticsModule
from ..complex_amplitude import ComplexAmplitude, FieldGeometry
from ..diagonal_elements import StaticSLMField
from ..pointing_instability import PointingInstability
from ..recording import RecordingMixin


T = TypeVar("T", bound=nn.Module)


@dataclass(frozen=True)
class OpticalSystemCheckpoint:
    class_name: str
    spec: dict[str, object]
    state_dict: dict[str, object]


def capture_init(init: Callable[..., None]) -> Callable[..., None]:
    """Decorator for a concrete ``OpticalSystem.__init__`` that records the bound
    constructor arguments into ``self._init_kwargs``, so the base
    ``get_checkpoint_spec`` can reproduce them without a hand-written method.

    The original ``__init__`` runs first (so ``nn.Module`` is fully initialised);
    the captured dict holds the live arguments verbatim (a ``torch.Generator`` is
    serialized only later, by ``get_checkpoint_spec``).
    """
    signature = inspect.signature(init)

    @functools.wraps(init)
    def wrapper(self, *args, **kwargs) -> None:
        init(self, *args, **kwargs)
        bound = signature.bind(self, *args, **kwargs)
        bound.apply_defaults()
        self._init_kwargs = {
            name: value
            for name, value in bound.arguments.items()
            if name != "self"
        }

    return wrapper


class OpticalSystem(nn.Module):
    """
    Sequential container for optical systems with named layers.
    """

    def __init__(self, input_geometry: FieldGeometry, **modules: OpticsModule) -> None:
        super().__init__()
        self.input_geometry = input_geometry
        self.device = input_geometry.pixel_size.device

        self.init_field = ComplexAmplitude(
            data=torch.ones(
                input_geometry.resolution,
                dtype=torch.complex64,
                device=self.device,
            ),
            wavelength=input_geometry.wavelength,
            pixel_size=input_geometry.pixel_size,
        )

        self._order: list[str] = []

        for name, module in modules.items():
            self.add(name, module)

    def add(self, name: str, module: OpticsModule) -> None:
        if hasattr(self, name):
            raise ValueError(f"Layer '{name}' already exists")

        setattr(self, name, module)
        self._order.append(name)

    def insert_after(
        self,
        reference: str | Type[OpticsModule],
        name: str,
        module: OpticsModule,
    ) -> None:
        """Insert ``module`` immediately after an existing layer.

        ``reference`` is either a layer name or the type of an existing layer
        (the first match is used).
        """
        if hasattr(self, name):
            raise ValueError(f"Layer '{name}' already exists")

        if isinstance(reference, str):
            if reference not in self._order:
                raise KeyError(f"No layer named '{reference}'")
            index = self._order.index(reference)
        else:
            index = next(
                (
                    i
                    for i, existing in enumerate(self._order)
                    if isinstance(getattr(self, existing), reference)
                ),
                None,
            )
            if index is None:
                raise KeyError(f"No module of type {reference.__name__}")

        setattr(self, name, module)
        self._order.insert(index + 1, name)

    def forward(
        self, complex_amplitude: ComplexAmplitude | None = None
    ) -> ComplexAmplitude:
        out = complex_amplitude

        if out is None:
            out = self.init_field

        for name in self._order:
            out = getattr(self, name)(out)
        return out

    def layers(self) -> dict[str, OpticsModule]:
        return {name: getattr(self, name) for name in self._order}

    def get(self, module_type: Type[T]) -> T:
        for name in self._order:
            module = getattr(self, name)
            if isinstance(module, module_type):
                return module
        raise KeyError(f"No module of type {module_type.__name__}")

    def has(self, module_type: Type[OpticsModule]) -> bool:
        return any(isinstance(getattr(self, n), module_type) for n in self._order)

    def record(self, enabled: bool = True) -> None:
        """Toggle recording on every layer that supports it (see
        :class:`~hologradpy.propagation.recording.RecordingMixin`). Read the
        per-layer result from :attr:`history`."""
        for name in self._order:
            module = getattr(self, name)
            if isinstance(module, RecordingMixin):
                module.record(enabled)

    @contextmanager
    def record_samples(self) -> Iterator[OpticalSystem]:
        """Record every layer's declared values for the duration of the ``with``
        block (recording is turned off again on exit). Read them from
        :attr:`history`."""
        self.record(True)
        try:
            yield self
        finally:
            self.record(False)

    @property
    def history(self) -> dict[str, dict[str, Tensor]]:
        """Per-layer recording, ``{layer_name: {value_name: (n, ...) tensor}}``,
        including only layers that recorded something."""
        result: dict[str, dict[str, Tensor]] = {}
        for name in self._order:
            module = getattr(self, name)
            if isinstance(module, RecordingMixin):
                layer_history = module.history
                if layer_history:
                    result[name] = layer_history
        return result

    def summary(self):
        for i, name in enumerate(self._order):
            module = getattr(self, name)
            print(f"{i:02d}  {name:15} {module.__class__.__name__}")

    def power_report(
        self, input_field: ComplexAmplitude | None = None
    ) -> dict[str, object]:
        """Track optical power through the system, module by module.

        Runs the forward chain one module at a time, recording the field
        ``power()`` after each named module and its efficiency relative to the
        field entering it. Returns ``{"input_power", "modules": [{"module",
        "type", "power", "efficiency"}, ...]}``.

        Notes:
        - Power is absolute (watts) only when the input field / ``StaticSLMField``
          beam carries an absolute power and the FFT lens is ``power_normalized``.
          The diagonal-stage (phase/amplitude) efficiencies are scale-invariant
          ratios and meaningful regardless.
        - For a ``power_normalized`` FFT system the ``fourier_lens`` stage is
          ~1.0 by Parseval. The **camera-FOV capture fraction** is NOT the
          ``affine_transform`` stage efficiency: that resampling conserves the
          discrete L2 norm ``sum|E|^2`` (via ``1/sqrt(scale.prod())``), which
          equals physical power ``sum|E|^2 * pixel_area`` only when the pixel
          size is unchanged. When the affine changes the pixel size (FFT plane
          -> camera) the physical power scales by ``(pixel_out/pixel_in)**2``,
          so the stage ratio is ``1/scale**2``, not the capture fraction. A
          NUFFT lens likewise only computes the camera window. The physical
          capture fraction comes from integrating ``|E_fft|^2`` over the FOV at
          the FFT-plane pixel size (crop the power-normalized FFT plane) -- see
          the NUFFT power diagnosis.
        """
        field = self.init_field if input_field is None else input_field
        input_power = field.power().detach()

        modules: list[dict[str, object]] = []
        previous_power = input_power
        for name in self._order:
            field = getattr(self, name)(field)
            current_power = field.power().detach()
            modules.append(
                {
                    "module": name,
                    "type": type(getattr(self, name)).__name__,
                    "power": current_power,
                    "efficiency": current_power / previous_power,
                }
            )
            previous_power = current_power

        return {"input_power": input_power, "modules": modules}

    def get_checkpoint_spec(self) -> dict[str, object]:
        """Return reconstructible keyword arguments for this system.

        The concrete constructor is captured by the ``@capture_init`` decorator into
        ``self._init_kwargs`` -- all of which are plain, picklable values.
        """
        spec = getattr(self, "_init_kwargs", None)
        if spec is None:
            raise NotImplementedError(
                f"{self.__class__.__name__} must decorate __init__ with "
                "@capture_init (or override get_checkpoint_spec) to support "
                "checkpointing."
            )
        return dict(spec)

    def get_init_kwargs(self) -> dict[str, object]:
        """Backward-compatible alias for the checkpoint spec."""
        return dict(self.get_checkpoint_spec())

    @classmethod
    def from_checkpoint_spec(
        cls,
        spec: dict[str, object],
    ) -> OpticalSystem:
        """Reconstruct an optical system from a checkpoint spec."""
        return cls(**spec)

    def save(self, filename: str) -> None:
        """Save model parameters and constructor metadata to a checkpoint."""
        # Ensure lazily initialised modules have created their parameters.
        _ = self()

        checkpoint = OpticalSystemCheckpoint(
            class_name=self.__class__.__name__,
            spec=self.get_checkpoint_spec(),
            state_dict=self.state_dict(),
        )
        torch.save(checkpoint, filename)

    @classmethod
    def load(
        cls,
        filename: str,
        map_location: str | torch.device | None = None,
        strict: bool = True,
        **kwargs,
    ) -> OpticalSystem:
        """Load a checkpoint created by :meth:`save`.

        Any explicit ``kwargs`` override the saved constructor arguments.
        """
        # These checkpoints intentionally store constructor metadata objects,
        # so we need the legacy full unpickling path here.
        checkpoint = torch.load(
            filename,
            map_location=map_location,
            weights_only=False,
        )

        if isinstance(checkpoint, OpticalSystemCheckpoint):
            checkpoint_class_name = checkpoint.class_name
            spec = dict(checkpoint.spec)
            state_dict = checkpoint.state_dict
        else:
            if "state_dict" not in checkpoint:
                raise KeyError("Checkpoint missing required key 'state_dict'.")

            checkpoint_class_name = checkpoint.get("class_name")
            spec = dict(checkpoint.get("spec", checkpoint.get("init_kwargs", {})))
            state_dict = checkpoint["state_dict"]

        if checkpoint_class_name is not None and checkpoint_class_name != cls.__name__:
            raise ValueError(
                f"Checkpoint was saved from '{checkpoint_class_name}', "
                f"but '{cls.__name__}.load' was called."
            )

        spec.update(kwargs)

        model = cls.from_checkpoint_spec(spec)
        # Ensure lazy modules are initialised before loading saved weights.
        _ = model()
        model.load_state_dict(state_dict, strict=strict)
        return model

    def __getitem__(self, key: int | str) -> OpticsModule:
        if isinstance(key, int):
            return getattr(self, self._order[key])
        return getattr(self, key)

    def __len__(self):
        return len(self._order)

    def __repr__(self):
        lines = ["OpticalSystem("]
        for name in self._order:
            lines.append(f"  ({name}): {repr(getattr(self, name))}")
        lines.append(")")
        return "\n".join(lines)

    # Fixes for type checking and IDE support
    def __call__(self, *args, **kwargs) -> ComplexAmplitude:
        return super().__call__(*args, **kwargs)


class SLMFourierLensModel(OpticalSystem):
    virtual_slm: VirtualSLM
    fourier_lens: OpticsModule

    def __init__(
        self,
        input_geometry: FieldGeometry,
        *,
        focal_length: float | None = None,
        pointing_focal_shift_std: float | tuple[float, float] | None = None,
        pointing_seed: int | None = None,
        **modules: OpticsModule,
    ) -> None:
        """Build the named SLM -> Fourier-lens chain.

        If ``pointing_focal_shift_std`` is given, a :class:`PointingInstability` is
        built from it via :meth:`PointingInstability.from_focal_shift` (using this
        model's ``focal_length``) and inserted immediately after the
        ``StaticSLMField`` stage, so the (static) SLM-plane field carries a freshly
        sampled beam-pointing tilt on every forward pass. ``pointing_seed`` seeds
        that sampling for reproducibility.
        """
        super().__init__(input_geometry, **modules)
        self.pointing_focal_shift_std = pointing_focal_shift_std
        if pointing_focal_shift_std is not None:
            if focal_length is None:
                raise ValueError(
                    "focal_length is required to build a PointingInstability "
                    "from pointing_focal_shift_std."
                )
            pointing = PointingInstability.from_focal_shift(
                pointing_focal_shift_std,
                focal_length,
                seed=pointing_seed,
            )
            self.insert_after(StaticSLMField, "pointing_instability", pointing)

    def forward(
        self, complex_amplitude: ComplexAmplitude | None = None
    ) -> ComplexAmplitude:
        return super().forward(complex_amplitude)
