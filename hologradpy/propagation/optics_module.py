from __future__ import annotations
from typing import TypedDict, Any  # , Self

import torch
from torch import nn, Tensor

from .complex_amplitude import ComplexAmplitude, FieldGeometry
from .fourier import get_spatial_grid


class SaveDict(TypedDict):
    state_dict: dict[str, Any]
    input_geometry: FieldGeometry
    resolution_out: tuple[int, int]
    pixel_size_out: Tensor


class OpticsModule(nn.Module):
    def __init__(
        self,
        pixel_size_out: tuple[float, float] | None = None,
        resolution_out: tuple[int, int] | None = None,
    ) -> None:
        """Base class to simulate the propagation of light.
        OpticsModule can be lazily initialised, meaning without
        prior knowledge of the dimensions of the input ElectricField.
        The input and output dimensions are initialised before the first
        forward() call.

        Args:
            pixel_size_out (Tuple[float, float] | None, optional):
                Output pixel size in meters (height, width). Defaults to
                None.
            resolution_out (Tuple[int, int] | None, optional): Output
                resolution in pixels (height, width). Defaults to None.
        """
        super().__init__()

        self._pixel_size_out_init = pixel_size_out

        self._resolution_out: tuple[int, int] | None = resolution_out

        self._input_geometry: FieldGeometry | None = None

        self._pixel_size_out: Tensor | None = None

        self._init_hook = self.register_forward_pre_hook(self._initialize_from_input)

        self.initialized = False

    def lazy_init(self, complex_amplitude: ComplexAmplitude) -> None:
        """Needs to assign values to:
        - self._pixel_size_out
        - self._resolution_out
        """
        if self._pixel_size_out_init is None:
            self._pixel_size_out = self.pixel_size_in
        else:
            self._pixel_size_out = torch.tensor(
                self._pixel_size_out_init,
                device=complex_amplitude.pixel_size.device,
                dtype=complex_amplitude.pixel_size.dtype,
            )

        if self._resolution_out is None:
            self._resolution_out = self.resolution_in

    def _lazy_initialize(self, complex_amplitude: ComplexAmplitude) -> None:
        """Run lazy initialization from an input-plane field (or probe)."""
        self._input_geometry = complex_amplitude.geometry
        self.initialized = True

        self.lazy_init(complex_amplitude)

        if self._resolution_out is None or self._pixel_size_out is None:
            raise ValueError(
                "OpticsModule subclasses must set _pixel_size_out and "
                "_resolution_out in lazy_init()."
            )

        if self._pixel_size_out.ndim == 1:
            self._pixel_size_out = self._pixel_size_out.unsqueeze(0)
        if (
            self._pixel_size_out.shape[0] == 1
            and complex_amplitude.geometry.number_of_wavelengths > 1
        ):
            self._pixel_size_out = self._pixel_size_out.repeat(
                complex_amplitude.geometry.number_of_wavelengths, 1
            )

        self._init_hook.remove()

    def _initialize_from_input(self, module, inputs) -> None:
        self._lazy_initialize(inputs[0])

    def initialize_from_geometry(
        self,
        geometry: FieldGeometry,
        dtype: torch.dtype = torch.complex64,
    ) -> None:
        """Initialise the module from an input-plane geometry, without a
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
        # reads field values — so the (uninitialised) contents are unused.
        probe = ComplexAmplitude(
            torch.empty(shape, dtype=dtype, device=geometry.wavelength.device),
            geometry.wavelength,
            geometry.pixel_size,
        )
        self._lazy_initialize(probe)

    def _ensure_initialized(self) -> None:
        """Raise if the module has not been lazily initialised yet."""
        if not self.initialized:
            raise RuntimeError(
                f"{type(self).__name__} must be initialised before this "
                "call; run forward() once or call "
                "initialize_from_geometry(input_geometry)."
            )

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
        """Input pixel size in meters.

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
        """Output pixel size in meters.

        Returns:
            Tensor: Output pixel size (height, width).
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
            "state_dict": self.state_dict(),
            "input_geometry": self.input_geometry,
            "resolution_out": self.resolution_out,
            "pixel_size_out": self.pixel_size_out,
        }
        torch.save(save_dict, path)

    def load_weights(self, path: str) -> None:
        state = torch.load(path, weights_only=False)
        self.load_state_dict(state["state_dict"])

    # TODO: Add Self type hint once upgraded to Python>=3.11
    @classmethod
    def from_file(cls, path: str, device: torch.device = "cpu"):
        raise NotImplementedError(
            "OpticsModule subclasses must implement from_file() method."
        )

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
