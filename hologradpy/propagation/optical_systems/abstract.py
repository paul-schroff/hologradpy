from __future__ import annotations
from dataclasses import dataclass
from typing import TypeVar, Type

import torch
import torch.nn as nn

from ..virtual_slms.abstract import VirtualSLM

from ..optics_module import OpticsModule
from ..complex_amplitude import ComplexAmplitude, FieldGeometry


T = TypeVar("T", bound=nn.Module)


@dataclass(frozen=True)
class OpticalSystemCheckpoint:
    class_name: str
    spec: dict[str, object]
    state_dict: dict[str, object]


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

    def summary(self):
        for i, name in enumerate(self._order):
            module = getattr(self, name)
            print(f"{i:02d}  {name:15} {module.__class__.__name__}")

    def get_checkpoint_spec(self) -> dict[str, object]:
        """Return reconstructible keyword arguments for this system."""
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement get_checkpoint_spec()."
        )

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

    def forward(
        self, complex_amplitude: ComplexAmplitude | None = None
    ) -> ComplexAmplitude:
        return super().forward(complex_amplitude)
