"""The electric field object, a core data structure of the optics backend."""

from __future__ import annotations
from dataclasses import dataclass

from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor
from torch.autograd.function import FunctionCtx
from torch.utils._pytree import tree_map, tree_flatten
from torch._ops import OpOverload
from torch._prims_common import (
    corresponding_complex_dtype,
    corresponding_real_dtype,
)

import numpy as np
from numpy.typing import NDArray

from ..grids import get_spatial_grid
from ..utils import unsqueeze_to


def _real_dtype(dtype: torch.dtype) -> torch.dtype:
    """Real dtype for geometry metadata (wavelength / pixel_size) that matches a
    field of ``dtype``: the corresponding real dtype for a complex field (so a
    ``complex128`` field gets ``float64`` geometry, ``complex64`` gets
    ``float32``), the dtype itself if already real-floating, else the default.
    """
    if dtype.is_complex:
        return corresponding_real_dtype(dtype)
    if dtype.is_floating_point:
        return dtype
    return torch.get_default_dtype()


def broadcast_wavelength_operand(operand: Tensor, field_ndim: int) -> Tensor:
    """Align a per-wavelength operand to a field of the given rank.

    The operand is laid out as ``(n_wavelengths, H, W)``. A 2D field carries no
    wavelength axis (it is single-wavelength), so the singleton wavelength axis
    is dropped. Otherwise leading singleton batch axes are added so it
    broadcasts against ``(*batch, n_wavelengths, H, W)`` without changing rank.
    """
    if field_ndim == 2:
        return operand.squeeze(0)
    return unsqueeze_to(operand, field_ndim)


@dataclass(frozen=True)
class BatchSpec:
    """Records the leading-dimension layout of a :class:`ComplexAmplitude`.

    A field is canonically laid out as ``(*batch, wavelength, H, W)`` where the
    wavelength axis is always at ``dim=-3`` and everything before it is batch.
    To run a fixed-rank operation (e.g. kornia ``warp_perspective`` or
    ``torchkbnufft``) the batch dimensions are collapsed into a single leading
    axis, giving canonical ``(N, n_wavelengths, H, W)``. ``BatchSpec`` captures
    enough information to restore the original rank afterwards.
    """

    leading_shape: tuple[int, ...]
    original_ndim: int


class _WrapperToTensor(torch.autograd.Function):
    """Convert a :class:`ComplexAmplitude` to a plain ``Tensor`` on-graph.

    When a field is produced by an ``OpticsModule`` (i.e. via
    ``__torch_dispatch__``), the autograd graph lives on the outer wrapper
    while the inner ``_data`` tensor is detached. Reading ``_data`` directly
    would therefore silently break gradient flow. This ``Function`` returns
    the inner values as a plain tensor in forward and routes the incoming
    gradient back through the wrapper (re-wrapped with the field geometry) in
    backward, so the result is a genuine real/complex tensor that still
    participates in optimization.
    """

    @staticmethod
    def forward(ctx: FunctionCtx, field: ComplexAmplitude) -> Tensor:
        ctx.geometry = field.geometry
        return field._data

    @staticmethod
    def backward(ctx: FunctionCtx, grad: Tensor) -> ComplexAmplitude:
        geometry = ctx.geometry
        return ComplexAmplitude(grad, geometry.wavelength, geometry.pixel_size)


class _TensorToWrapper(torch.autograd.Function):
    """Wrap a plain ``Tensor`` as a :class:`ComplexAmplitude`, on-graph.

    The mirror image of :class:`_WrapperToTensor`, and the reason it is needed:
    ``_make_wrapper_subclass`` produces an autograd **leaf**. Building a wrapper
    straight from a graph-carrying tensor therefore creates a wrapper with no
    edge back to it, so a module that goes on to work through
    ``__torch_dispatch__`` records its own gradients against that leaf and the
    gradient never reaches the tensor the field was built from.

    Routing the crossing through this ``Function`` supplies the missing edge, so
    the graph survives in both directions. This is the pattern PyTorch's own
    wrapper subclasses use (compare ``DTensor._FromTorchTensor``).
    """

    @staticmethod
    def forward(
        ctx: FunctionCtx,
        data: Tensor,
        wavelength: Tensor,
        pixel_size: Tensor,
    ) -> ComplexAmplitude:
        # The inner tensor is detached: the graph belongs on the wrapper, which
        # autograd links back to ``data`` through this Function.
        return ComplexAmplitude(data.detach(), wavelength, pixel_size)

    @staticmethod
    def backward(ctx: FunctionCtx, grad: ComplexAmplitude) -> tuple[Tensor, None, None]:
        inner = grad._data if isinstance(grad, ComplexAmplitude) else grad
        # wavelength / pixel_size are geometry metadata and never differentiable.
        return inner, None, None


def pixel_area(pixel_size: Tensor) -> Tensor:
    """The area of one pixel per wavelength, from a ``(n_wavelengths, 2)`` pitch.
    Always float64, whatever the field's dtype.
    """
    return (pixel_size[:, 0] * pixel_size[:, 1]).to(torch.float64).reshape(-1)


def _power_factor(
    current_power: Tensor, power: float | Tensor, ndim: int, device: torch.device
) -> Tensor:
    """The amplitude scale taking a field of ``current_power`` to ``power``."""
    target_power = torch.as_tensor(power, dtype=torch.float64, device=device)
    factor = torch.sqrt(target_power / current_power)
    return factor[..., None, None] if ndim > 2 else factor


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
    """An electric field: complex values carrying the geometry that gives them meaning.

    A tensor subclass, so a field can be multiplied, propagated and differentiated like
    any other tensor, while :attr:`wavelength` and :attr:`pixel_size` travel with it.
    That is what lets an :class:`~hologradpy.optics.modules.abstract.OpticsModule` read
    the sampling of its input rather than being told it.

    Operations are intercepted through ``__torch_dispatch__``, which keeps the geometry
    attached across them. The wrapper is an autograd leaf, so a field built straight
    from a graph-carrying tensor would strand that graph. Use :meth:`from_tensor` to
    cross into a field and :meth:`as_tensor` to cross back out, since both route through
    an autograd function that preserves the edge.
    """

    __torch_function__ = torch._C._disabled_torch_function_impl

    @staticmethod
    def __new__(
        cls: type[ComplexAmplitude],
        data: Tensor | ComplexAmplitude,
        wavelength: float | Tensor,
        pixel_size: tuple[float, float] | Tensor,
        power: float | Tensor | None = None,
    ) -> ComplexAmplitude:
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

        wavelength, pixel_size = cls._sanitize_inputs(inner, wavelength, pixel_size)

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
        power: float | Tensor | None = None,
    ) -> None:
        """Wrap ``data`` as a field with the given geometry.

        Note on autograd: this builds the wrapper with
        ``_make_wrapper_subclass``, which is a *leaf*. If ``data`` carries a
        graph, the field is **not** connected to it, so a gradient flowing back
        through any ``__torch_dispatch__`` operation stops at this wrapper and
        never reaches ``data``. Use :meth:`from_tensor` whenever the input may
        be on-graph (it falls back to this constructor when it is not).
        """
        if isinstance(data, ComplexAmplitude):
            # Unwrap to the raw inner tensor for storage. The autograd graph
            # (grad_fn / requires_grad) lives on the outer
            # _make_wrapper_subclass wrapper.
            data = data._data
        elif not isinstance(data, Tensor):
            data = torch.as_tensor(data)

        wavelength, pixel_size = self._sanitize_inputs(data, wavelength, pixel_size)

        # Optionally scale the field to an absolute power (watts) at construction.
        if power is not None:
            data = self._scale_to_power(data, pixel_size, power)

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
        geometry_dtype = _real_dtype(data.dtype)
        if isinstance(wavelength, float):
            wavelength = torch.tensor(
                [wavelength], device=data.device, dtype=geometry_dtype
            )
        elif isinstance(wavelength, Tensor):
            if wavelength.ndim == 0:
                wavelength = wavelength.unsqueeze(0)
            if wavelength.ndim != 1:
                raise TypeError("Wavelength tensor must be a scalar or 1D.")
        else:
            raise TypeError("Wavelength must be either float or a scalar or 1D tensor.")

        if isinstance(pixel_size, tuple):
            pixel_size = torch.tensor(
                [pixel_size], device=data.device, dtype=geometry_dtype
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
            raise ValueError("pixel_size must have shape (1, 2) or (n_wavelength, 2)")
        if pixel_size.shape[0] == 1 and wavelength.shape[0] > 1:
            pixel_size = pixel_size.expand(wavelength.shape[0], 2)

        if data.ndim < 2:
            raise ValueError("Data must have at least 2 dimensions (H, W).")
        elif data.ndim == 2:
            if wavelength.numel() > 1:
                raise ValueError("If data is 2D, wavelength must be a single value.")
        else:
            if data.shape[-3] != wavelength.shape[-1]:
                raise ValueError(
                    "The third-last dimension of data must match the number "
                    "of wavelengths: (..., wavelength, H, W)."
                )
        return wavelength, pixel_size

    @classmethod
    def from_geometry(
        cls: type[ComplexAmplitude],
        geometry: FieldGeometry,
        data: Tensor | None = None,
        dtype: torch.dtype = torch.complex64,
        power: float | Tensor | None = None,
    ) -> ComplexAmplitude:
        """Create a field that matches a :class:`FieldGeometry`.

        Args:
            geometry: Target geometry (wavelength, pixel size, resolution).
            data: Field values ``(..., H, W)``. If ``None``, a uniform
                unit-amplitude field (ones) is created at the geometry's
                resolution, with a leading wavelength axis when there is more
                than one wavelength.
            dtype: Dtype of the default field when ``data`` is ``None``.
            power: If given, scale the field to this absolute power (watts).

        Returns:
            ComplexAmplitude: A field carrying the geometry's wavelength and
            pixel size.
        """
        if data is None:
            number_of_wavelengths = geometry.number_of_wavelengths
            shape = (
                geometry.resolution
                if number_of_wavelengths == 1
                else (number_of_wavelengths, *geometry.resolution)
            )
            data = torch.ones(
                shape,
                dtype=dtype,
                device=geometry.wavelength.device,
            )
        return cls(data, geometry.wavelength, geometry.pixel_size, power=power)

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
    def batch_shape(self) -> tuple[int, ...]:
        """Leading batch dimensions preceding ``(wavelength, H, W)``.

        Returns an empty tuple for 2D ``(H, W)`` and 3D ``(wavelength, H, W)``
        fields, which carry no batch dimensions.
        """
        if self.ndim <= 3:
            return ()
        return tuple(self.shape[:-3])

    @property
    def spatial_extent(self) -> Tensor:
        return self.geometry.spatial_extent

    def get_spatial_grid(self) -> tuple[Tensor, Tensor]:
        return self.geometry.get_spatial_grid()

    def as_tensor(self) -> Tensor:
        """Return the underlying complex field as a plain ``torch.Tensor``,
        preserving the autograd graph wherever it lives.

        Prefer this over ``._data`` when building a differentiable loss: a
        field produced by an ``OpticsModule`` keeps its graph on the wrapper,
        so ``._data`` is detached and would break gradient flow.
        """
        if self._data.requires_grad:
            # Field built directly from graph-carrying data, so the inner
            # tensor is already on the graph.
            return self._data
        if self.requires_grad:
            # Field produced via dispatch, so the graph lives on the wrapper.
            return _WrapperToTensor.apply(self)
        return self._data

    @property
    def phase(self) -> Tensor:
        """Real-valued phase ``arg(E)``, differentiable and on-graph."""
        return torch.angle(self.as_tensor())

    @property
    def amplitude(self) -> Tensor:
        """Real-valued amplitude ``|E|``, differentiable and on-graph."""
        return self.as_tensor().abs()

    @property
    def intensity(self) -> Tensor:
        """Real-valued intensity ``|E|**2``, differentiable and on-graph.

        Computed as ``real**2 + imag**2`` to avoid the gradient singularity of
        ``abs()`` at zero field.
        """
        field = self.as_tensor()
        return field.real**2 + field.imag**2

    @staticmethod
    def _integrate_power(intensity: Tensor, pixel_size: Tensor) -> Tensor:
        """Integrate intensity over area -> optical power per
        ``(*batch, wavelength)`` (a scalar for a 2D field).

        Reduces over ``(H, W)`` in float64 for precision (the per-pixel values
        are fine in float32, but summing ~1e6 of them is not). ``pixel_size`` is
        ``(n_wavelengths, 2)``.
        """
        area = pixel_area(pixel_size)
        summed = intensity.to(torch.float64).sum(dim=(-2, -1))
        if intensity.ndim == 2:
            return summed * area.squeeze(0)
        return summed * area

    @classmethod
    def _scale_to_power(
        cls, data: Tensor, pixel_size: Tensor, power: float | Tensor
    ) -> Tensor:
        """Return ``data`` scaled so its integrated power equals ``power`` (W),
        preserving phase. The scale ratio is computed in float64.
        """
        intensity = data.real**2 + data.imag**2
        current_power = cls._integrate_power(intensity, pixel_size)
        factor = _power_factor(current_power, power, data.ndim, data.device)
        return data * factor.to(corresponding_real_dtype(data.dtype))

    def power(self) -> Tensor:
        """Total optical power = integral of intensity over area
        (``sum(|E|^2) * pixel_area``), returned per ``(*batch, wavelength)`` (a
        scalar for a 2D field).

        In SI this is watts when the field amplitude is in ``sqrt(W/m^2)`` and
        ``pixel_size`` in metres. The reduction is performed in float64.
        """
        return self._integrate_power(self.intensity, self.pixel_size)

    def with_power(self, power: float | Tensor) -> ComplexAmplitude:
        """Return this field scaled so ``power() == power``, preserving phase
        and the autograd graph.

        Like :meth:`with_geometry`, this is the graph-safe way to rescale a
        field produced by an OpticsModule: the multiply goes through the
        dispatch mechanism, keeping ``grad_fn`` intact. ``power`` is matched
        per ``(*batch, wavelength)``.
        """
        factor = _power_factor(self.power(), power, self.ndim, self.device)
        return self * factor.to(self.dtype_r)

    def numpy(self) -> NDArray[np.complex128]:
        """The field as a numpy array, detached and on the host."""
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

    def __repr__(self) -> str:
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

        Only the ``geometry`` attribute (pure metadata) is replaced, so the
        existing tensor wrapper, and therefore the ``grad_fn``, stays intact.
        This is the preferred way to retag an intermediate result with a new
        wavelength or pixel size.

        Args:
            wavelength: New wavelength(s). If *None*, the existing value is
                kept.
            pixel_size: New pixel size(s). If *None*, the existing value is
                kept.

        Returns:
            ComplexAmplitude: The same object with updated geometry.
        """
        if wavelength is None:
            wavelength = self.geometry.wavelength
        elif isinstance(wavelength, float):
            wavelength = torch.tensor(
                [wavelength], device=self.device, dtype=self.dtype_r
            )
        elif isinstance(wavelength, Tensor) and wavelength.ndim == 0:
            wavelength = wavelength.unsqueeze(0)

        if pixel_size is None:
            pixel_size = self.geometry.pixel_size
        elif isinstance(pixel_size, tuple):
            pixel_size = torch.tensor(
                [pixel_size], device=self.device, dtype=self.dtype_r
            )
        elif isinstance(pixel_size, Tensor) and pixel_size.ndim == 1:
            pixel_size = pixel_size.unsqueeze(0)

        new_geometry = FieldGeometry(wavelength, pixel_size, self.geometry.resolution)
        # Use object.__setattr__ to bypass any tensor attribute-setting
        # restrictions while keeping the autograd graph intact.
        object.__setattr__(self, "geometry", new_geometry)
        return self

    def flatten_batch(self) -> tuple[Tensor, BatchSpec]:
        """Collapse all batch dimensions into a single leading axis.

        Returns the underlying tensor reshaped to canonical
        ``(N, n_wavelengths, H, W)`` form together with a :class:`BatchSpec`
        describing the original layout. This is the entry point for ND batch
        support in fixed-rank ``OpticsModule`` implementations: flatten, run
        the fixed-rank operation, then restore with :meth:`unflatten_batch`.

        Returns:
            tuple[Tensor, BatchSpec]: The ``(N, n_wavelengths, H, W)`` tensor
            (sharing storage with ``self._data``) and the spec needed to
            restore rank.
        """
        height, width = self.resolution
        n_wavelengths = self.number_of_wavelengths
        data = self.as_tensor()

        if self.ndim == 2:
            spec = BatchSpec(leading_shape=(), original_ndim=2)
            return data.reshape(1, 1, height, width), spec

        spec = BatchSpec(leading_shape=tuple(self.shape[:-3]), original_ndim=self.ndim)
        return data.reshape(-1, n_wavelengths, height, width), spec

    @classmethod
    def from_tensor(
        cls,
        data: Tensor,
        wavelength: float | Tensor,
        pixel_size: tuple[float, float] | Tensor,
    ) -> ComplexAmplitude:
        """Build a field from a plain tensor, preserving the autograd graph.

        Prefer this over calling the constructor directly whenever ``data`` may
        carry a graph. The constructor goes through ``_make_wrapper_subclass``,
        which produces an autograd *leaf*, so the resulting field would be
        disconnected from ``data`` and any gradient flowing back through a
        ``__torch_dispatch__`` operation would stop at the wrapper. See
        :class:`_TensorToWrapper`.

        For a graph-free tensor this is exactly the constructor.
        """
        if isinstance(data, Tensor) and data.requires_grad:
            wavelength, pixel_size = cls._sanitize_inputs(
                data, wavelength, pixel_size
            )
            return _TensorToWrapper.apply(data, wavelength, pixel_size)
        return cls(data, wavelength, pixel_size)

    @classmethod
    def unflatten_batch(
        cls,
        data: Tensor,
        spec: BatchSpec,
        wavelength: float | Tensor,
        pixel_size: tuple[float, float] | Tensor,
    ) -> ComplexAmplitude:
        """Restore a canonical ``(N, n_wavelengths, H, W)`` tensor to the rank
        recorded in ``spec`` and wrap it as a :class:`ComplexAmplitude`.

        Inverse of :meth:`flatten_batch`. The output spatial resolution is
        taken from ``data`` so it may differ from the input (e.g. after a
        resampling propagator), while batch and wavelength dimensions are
        preserved.

        Args:
            data: Tensor shaped ``(N, n_wavelengths, H_out, W_out)``.
            spec: Layout captured by :meth:`flatten_batch`.
            wavelength: Output wavelength(s).
            pixel_size: Output pixel size(s).

        Returns:
            ComplexAmplitude: Field with the same rank as the original input.
        """
        n_wavelengths = data.shape[1]
        height_out, width_out = data.shape[-2:]

        if spec.original_ndim == 2:
            out = data.reshape(height_out, width_out)
        elif spec.original_ndim == 3:
            out = data.reshape(n_wavelengths, height_out, width_out)
        else:
            out = data.reshape(
                *spec.leading_shape, n_wavelengths, height_out, width_out
            )

        # from_tensor, not the constructor: a resampling module reaches here with
        # an on-graph tensor, and the constructor would strand it on a leaf.
        return cls.from_tensor(out, wavelength, pixel_size)

    @classmethod
    def __torch_dispatch__(
        cls,
        func: OpOverload,
        types: tuple[type, ...],
        args: tuple = (),
        kwargs: dict | None = None,
    ) -> Any:
        """Ensure that ComplexAmplitude is treated like a ``torch.Tensor``,
        including propagation of gradients.

        Useful references on how this works:

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
                        "ComplexAmplitude arguments must have the same wavelength."
                    )
                if not torch.allclose(field.pixel_size, geometry.pixel_size):
                    raise ValueError(
                        "ComplexAmplitude arguments must have the same pixel size."
                    )
                if field.resolution != geometry.resolution:
                    raise ValueError(
                        "ComplexAmplitude arguments must have the same resolution."
                    )

        def unwrap(x: Any) -> Any:
            if not isinstance(x, cls):
                return x
            inner = x._data
            # ``_make_wrapper_subclass`` does not carry the lazy conjugate / negative
            # bit, so a conj/neg *view* (as complex autograd's ``mul`` backward
            # produces) would be re-wrapped without its bit, silently dropping the
            # conjugation and corrupting gradients of complex fields. Materialize it.
            if inner.is_conj():
                inner = inner.resolve_conj()
            if inner.is_neg():
                inner = inner.resolve_neg()
            return inner

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

        def wrap_output(x: Any) -> Any:
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
