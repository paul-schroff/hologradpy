from __future__ import annotations
from dataclasses import dataclass

from collections.abc import Mapping

import torch
from torch import Tensor
from torch.utils._pytree import tree_map, tree_flatten
from torch._prims_common import (
    corresponding_complex_dtype,
    corresponding_real_dtype,
)

import numpy as np
from numpy.typing import NDArray

from .utils.fourier_utils import get_spatial_grid


@dataclass(frozen=True)
class FieldGeometry:
    wavelength: Tensor
    pixel_size: Tensor
    resolution: tuple[int, int]

    @property
    def number_of_wavelengths(self) -> int:
        return self.wavelength.numel()

    @property
    def wavenumber(self) -> Tensor:
        return 2 * torch.pi / self.wavelength

    @property
    def spatial_extent(self) -> Tensor:
        resolution = torch.as_tensor(
            self.resolution,
            device=self.wavelength.device,
            dtype=self.wavelength.dtype,
        )
        return self.pixel_size * resolution
    
    def get_spatial_grid(self, index: int = 0) -> tuple[Tensor, Tensor]:
        return get_spatial_grid(
            resolution=self.resolution,
            pixel_size=self.pixel_size[index].tolist(),
            device=self.wavelength.device,
        )


class ComplexAmplitude(Tensor):
    __torch_function__ = torch._C._disabled_torch_function_impl

    @staticmethod
    def __new__(
        cls: type[ComplexAmplitude],
        data: Tensor | ComplexAmplitude,
        wavelength: float | Tensor,
        pixel_size: tuple[float, float] | Tensor,
    ):
        if isinstance(data, cls):
            # Keep the outer wrapper's requires_grad flag so that gradients
            # flowing through the ComplexAmplitude dispatch mechanism are not
            # silently dropped when re-wrapping.
            requires_grad = data.requires_grad
            inner = data._data
        elif not isinstance(data, Tensor):
            inner = torch.as_tensor(data)
            requires_grad = inner.requires_grad
        else:
            inner = data
            requires_grad = inner.requires_grad

        wavelength, pixel_size = cls._sanitize_inputs(
            inner, wavelength, pixel_size
        )

        return Tensor._make_wrapper_subclass(
            cls,
            size=inner.shape,
            dtype=inner.dtype,
            layout=inner.layout,
            device=inner.device,
            strides=inner.stride(),
            storage_offset=inner.storage_offset(),
            requires_grad=requires_grad,
        )

    def __init__(
        self,
        data: Tensor | ComplexAmplitude,
        wavelength: float | Tensor,
        pixel_size: tuple[float, float] | Tensor,
    ):
        if isinstance(data, ComplexAmplitude):
            # Unwrap to the raw inner tensor for storage. The autograd graph
            # (grad_fn / requires_grad) lives on the outer 
            # _make_wrapper_subclass wrapper.
            data = data._data
        elif not isinstance(data, Tensor):
            data = torch.as_tensor(data)

        wavelength, pixel_size = self._sanitize_inputs(
            data, wavelength, pixel_size
        )
        self._data: Tensor = data
        self.geometry: FieldGeometry = FieldGeometry(
            wavelength, pixel_size, data.shape[-2:]
        )

    @staticmethod
    def _sanitize_inputs(
        data: Tensor,
        wavelength: float | Tensor,
        pixel_size: tuple[float, float] | Tensor,
    ) -> tuple[Tensor, Tensor]:
        if isinstance(wavelength, float):
            wavelength = torch.tensor(
                [wavelength], device=data.device, dtype=torch.float32
            )
        elif isinstance(wavelength, Tensor):
            if wavelength.ndim == 0:
                wavelength = wavelength.unsqueeze(0)
            if wavelength.ndim != 1:
                raise TypeError("Wavelength tensor must be a scalar or 1D.")
        else:
            raise TypeError(
                "Wavelength must be either float or a scalar or 1D tensor."
            )

        if isinstance(pixel_size, tuple):
            pixel_size = torch.tensor(
                [pixel_size], device=data.device, dtype=torch.float32
            )
        elif isinstance(pixel_size, Tensor):
            if pixel_size.ndim == 1:
                pixel_size = pixel_size.unsqueeze(0)
            if pixel_size.ndim != 2 or pixel_size.shape[1] != 2:
                raise TypeError(
                    "Pixel size tensor must be a tuple or a 2D tensor of "
                    "shape (1, 2) or (N, 2)."
                )
        else:
            raise TypeError(
                "Pixel size must be either a tuple or a tensor with shape "
                "(1, 2) or (N, 2)."
            )
        if pixel_size.shape[0] not in (1, wavelength.numel()):
            raise ValueError(
                "pixel_size must have shape (1, 2) or (n_wavelength, 2)"
            )
        if pixel_size.shape[0] == 1 and wavelength.shape[0] > 1:
            pixel_size = pixel_size.expand(wavelength.shape[0], 2)

        if data.ndim < 2:
            raise ValueError("Data must have at least 2 dimensions (H, W).")
        elif data.ndim == 2:
            if wavelength.numel() > 1:
                raise ValueError(
                    "If data is 2D, wavelength must be a single value."
                )
        else:
            if data.shape[-3] != wavelength.shape[-1]:
                raise ValueError(
                    "The third-last dimension of data must match the number "
                    "of wavelengths: (..., wavelength, H, W)."
                )
        return wavelength, pixel_size

    @property
    def wavelength(self) -> Tensor:
        return self.geometry.wavelength
    
    @property
    def number_of_wavelengths(self) -> int:
        return self.geometry.number_of_wavelengths

    @property
    def wavenumber(self) -> Tensor:
        return self.geometry.wavenumber
    @property
    def pixel_size(self) -> Tensor:
        return self.geometry.pixel_size

    @property
    def resolution(self) -> tuple[int, int]:
        return self.geometry.resolution

    @property
    def spatial_extent(self) -> Tensor:
        return self.geometry.spatial_extent
    
    def get_spatial_grid(self) -> tuple[Tensor, Tensor]:
        return self.geometry.get_spatial_grid()

    @property
    def phase(self) -> Tensor:
        return torch.angle(self._data)

    @property
    def amplitude(self) -> Tensor:
        return torch.abs(self._data)

    @property
    def intensity(self) -> Tensor:
        return self.amplitude ** 2

    def numpy(self) -> NDArray[np.complex_]:
        return self._data.detach().cpu().numpy()
    
    @property
    def dtype_r(self: ComplexAmplitude) -> torch.dtype:
        if self.dtype.is_complex:
            return corresponding_real_dtype(self.dtype)
        else:
            return self.dtype

    @property
    def dtype_c(self: ComplexAmplitude) -> torch.dtype:
        if self.dtype.is_complex:
            return self.dtype
        else:
            return corresponding_complex_dtype(self.dtype)

    @property
    def eps(self: ComplexAmplitude) -> float:
        return torch.finfo(self.dtype_r).eps

    def __repr__(self):
        return (
            f"ComplexAmplitude(shape={tuple(self._data.shape)}, "
            f"dtype={self._data.dtype}, "
            f"wavelength={self.wavelength}, "
            f"pixel_size={self.pixel_size})"
        )

    def with_geometry(
        self,
        wavelength: float | Tensor | None = None,
        pixel_size: tuple[float, float] | Tensor | None = None,
    ) -> ComplexAmplitude:
        """Return this ComplexAmplitude with updated wavelength / pixel_size
        metadata, preserving the autograd graph.

        This is the preferred alternative to constructing a new
        ``ComplexAmplitude(result, new_wavelength, new_pixel_size)`` from an
        intermediate result, which would sever the autograd graph.  Instead
        this method replaces only the ``geometry`` attribute (pure metadata)
        while keeping the existing tensor wrapper – and therefore the
        ``grad_fn`` – intact.

        Args:
            wavelength: New wavelength(s). If *None*, the existing value is
                kept.
            pixel_size: New pixel size(s). If *None*, the existing value is
                kept.

        Returns:
            ComplexAmplitude
                The same object with updated geometry.
        """
        if wavelength is None:
            wavelength = self.geometry.wavelength
        elif isinstance(wavelength, float):
            wavelength = torch.tensor(
                [wavelength], device=self.device, dtype=torch.float32
            )
        elif isinstance(wavelength, Tensor) and wavelength.ndim == 0:
            wavelength = wavelength.unsqueeze(0)

        if pixel_size is None:
            pixel_size = self.geometry.pixel_size
        elif isinstance(pixel_size, tuple):
            pixel_size = torch.tensor(
                [pixel_size], device=self.device, dtype=torch.float32
            )
        elif isinstance(pixel_size, Tensor) and pixel_size.ndim == 1:
            pixel_size = pixel_size.unsqueeze(0)

        new_geometry = FieldGeometry(
            wavelength, pixel_size, self.geometry.resolution
        )
        # Use object.__setattr__ to bypass any tensor attribute-setting
        # restrictions while keeping the autograd graph intact.
        object.__setattr__(self, "geometry", new_geometry)
        return self

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        """This method ensures that ComplexAmplitude is treated like
        torch.Tensor, including propagation of gradients. Useful
        references on how this works:
        - https://docs.google.com/presentation/d/1piuv9nBzyoqdH49D1SoE5OZUPSMpOOFqfSKOhr-ab2c/edit#slide=id.p1
        - https://github.com/albanD/subclass_zoo
        - https://dev-discuss.pytorch.org/t/what-and-why-is-torch-dispatch/557
        """
        kwargs = kwargs or {}

        flat_args = tree_flatten((args, kwargs))[0]
        fields = [x for x in flat_args if isinstance(x, cls)]

        if not fields:
            # No ComplexAmplitude found, call the function as usual
            return func(*args, **kwargs)

        geometry = fields[0].geometry
        # During backward, torch.is_grad_enabled() is False.  Backward
        # kernels operate on gradient tensors that may carry mismatched
        # geometry metadata (e.g. SLM-plane vs camera-plane pixel size).
        # The geometry check is skipped in those cases and only enforced
        # during the forward pass where it matters for correctness.
        if torch.is_grad_enabled():
            for field in fields[1:]:
                if not torch.allclose(field.wavelength, geometry.wavelength):
                    raise ValueError(
                        "ComplexAmplitude arguments must have the same "
                        "wavelength."
                    )
                if not torch.allclose(field.pixel_size, geometry.pixel_size):
                    raise ValueError(
                        "ComplexAmplitude arguments must have the same pixel "
                        "size."
                    )
                if field.resolution != geometry.resolution:
                    raise ValueError(
                        "ComplexAmplitude arguments must have the same "
                        "resolution."
                    )

        def unwrap(x):
            return x._data if isinstance(x, cls) else x

        # Special handling for slicing to update wavelength
        if func == torch.ops.aten.slice.Tensor:
            # Handling when step is not provided (defaults to None)
            if len(args) < 5:
                input_tensor, dim, start, end = args
                step = None
            else:
                input_tensor, dim, start, end, step = args

            dim = dim if dim >= 0 else dim + input_tensor.ndim

            out = func(*tree_map(unwrap, args), **kwargs)

            # Wavelength dimension
            if dim == input_tensor.ndim - 3:
                new_wavelength = geometry.wavelength[start:end:step]
                new_pixel_size = geometry.pixel_size[start:end:step]
            else:
                new_wavelength = geometry.wavelength
                new_pixel_size = geometry.pixel_size

            return cls(out, new_wavelength, new_pixel_size)
        elif func == torch.ops.aten.select.int:
            input_tensor, dim, index = args

            dim = dim if dim >= 0 else dim + input_tensor.ndim

            out = func(*tree_map(unwrap, args), **kwargs)

            if dim == input_tensor.ndim - 3:
                new_wavelength = geometry.wavelength[index].unsqueeze(0)
                new_pixel_size = geometry.pixel_size[index].unsqueeze(0)
            else:
                new_wavelength = geometry.wavelength
                new_pixel_size = geometry.pixel_size

            return cls(out, new_wavelength, new_pixel_size)
        elif func in (
            torch.ops.aten.to.device,
            torch.ops.aten._to_copy.default,
        ):
            out = func(*tree_map(unwrap, args), **kwargs)
            new_wavelength = geometry.wavelength.to(out.device)
            new_pixel_size = geometry.pixel_size.to(out.device)
            return cls(out, new_wavelength, new_pixel_size)
        else:
            pass

        out = func(*tree_map(unwrap, args), **tree_map(unwrap, kwargs))

        n_wavelength = geometry.wavelength.shape[0]

        def should_wrap_tensor(x: Tensor) -> bool:
            if isinstance(x, cls):
                return False
            if x.ndim < 2:
                return False
            if x.ndim == 2:
                return n_wavelength == 1
            return x.shape[-3] == n_wavelength

        def wrap_output(x):
            if isinstance(x, Tensor):
                if should_wrap_tensor(x):
                    return cls(x, geometry.wavelength, geometry.pixel_size)
                return x

            if isinstance(x, tuple):
                wrapped = tuple(wrap_output(item) for item in x)
                if hasattr(x, "_fields"):
                    return type(x)(*wrapped)
                return wrapped

            if isinstance(x, list):
                return [wrap_output(item) for item in x]

            if isinstance(x, Mapping):
                return type(x)((key, wrap_output(val)) for key, val in x.items())

            return x

        return wrap_output(out)
