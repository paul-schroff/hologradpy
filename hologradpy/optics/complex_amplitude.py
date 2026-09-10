"""The electric field object, a core data structure of the optics backend."""

from __future__ import annotations
from dataclasses import dataclass, replace

from collections.abc import Mapping, Sequence
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
from jaxtyping import Float

from ..grids import get_spatial_grid


# The two reserved axes. A field is always ``(*batch, component, wavelength, H, W)``.
COMPONENT_DIM = -4
WAVELENGTH_DIM = -3

# Allowed lengths of the component axis.
SCALAR = 1
VECTOR = 3


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


def _slice_arguments(
    args: tuple, kwargs: dict
) -> tuple[Tensor, int, int | None, int | None, int]:
    """The five arguments of ``aten.slice.Tensor`` with the defaults filled in.

    The schema is ``slice.Tensor(self, dim=0, start=None, end=None, step=1)`` and a
    caller such as ``torch.gradient`` passes only the first three.
    """
    names = ("self", "dim", "start", "end", "step")
    defaults = (None, 0, None, None, 1)
    values = list(args)
    for name, default in zip(names[len(args) :], defaults[len(args) :]):
        values.append(kwargs.get(name, default))
    return tuple(values)


def _to_canonical_layout(data: Tensor, number_of_wavelengths: int) -> Tensor:
    """``data`` in the canonical ``(*batch, component, wavelength, H, W)`` layout.

    ``(H, W)`` is one component of one wavelength, and ``(n_wavelengths, H, W)`` is
    one component of each. Rank four or more is expected to be canonical already.

    Args:
        data: The field values.
        number_of_wavelengths: How many wavelengths the geometry carries.

    Returns:
        Tensor: A view of ``data`` with both reserved axes present.

    Raises:
        ValueError: The layout cannot be read as a field.
    """
    if data.ndim < 2:
        raise ValueError(
            f"A field needs at least the two spatial axes (H, W), got shape "
            f"{tuple(data.shape)}."
        )
    if data.ndim == 2:
        if number_of_wavelengths != 1:
            raise ValueError(
                f"A plane of shape {tuple(data.shape)} carries one wavelength, but the "
                f"geometry carries {number_of_wavelengths}. Add a wavelength axis: "
                "(n_wavelengths, H, W)."
            )
        return data.reshape(1, 1, *data.shape)
    if data.ndim == 3:
        if data.shape[0] != number_of_wavelengths:
            raise ValueError(
                f"A field of rank 3 is read as (n_wavelengths, H, W), but "
                f"{tuple(data.shape)} does not start with the "
                f"{number_of_wavelengths} wavelengths of its geometry."
            )
        return data.reshape(1, *data.shape)
    if data.shape[WAVELENGTH_DIM] != number_of_wavelengths:
        raise ValueError(
            f"The wavelength axis of {tuple(data.shape)} does not match the "
            f"{number_of_wavelengths} wavelengths of its geometry. The layout is "
            "(*batch, component, wavelength, H, W)."
        )
    if data.shape[COMPONENT_DIM] not in (SCALAR, VECTOR):
        raise ValueError(
            f"The component axis of {tuple(data.shape)} has length "
            f"{data.shape[COMPONENT_DIM]}, and only {SCALAR} (a scalar field) or "
            f"{VECTOR} (a field vector) are meaningful. The layout is "
            "(*batch, component, wavelength, H, W), so a batch of fields needs a "
            "component axis of its own."
        )
    return data


def broadcast_wavelength_operand(operand: Tensor, field_ndim: int) -> Tensor:
    """Align a per-wavelength operand to a field.

    The operand is laid out as ``(n_wavelengths, H, W)`` or ``(H, W)``, and a field as
    ``(*batch, component, n_wavelengths, H, W)``.

    Args:
        operand: The per-wavelength values.
        field_ndim: Rank of the field the operand multiplies.

    Returns:
        Tensor: The operand, ready to broadcast.
    """
    if operand.ndim > 3:
        raise ValueError(
            f"A per-wavelength operand is (n_wavelengths, H, W) or (H, W), got "
            f"{tuple(operand.shape)}."
        )
    return operand


@dataclass(frozen=True)
class BatchSpec:
    """Records the leading layout of a :class:`ComplexAmplitude`.

    A field is canonically laid out as ``(*batch, component, wavelength, H, W)``. The
    component axis is always at ``dim=-4`` and holds one value for a scalar field or
    three for a field vector, the wavelength axis is always at ``dim=-3``, and
    everything before them is batch.
    """

    batch_shape: tuple[int, ...]
    number_of_components: int
    original_ndim: int


class _WrapperToTensor(torch.autograd.Function):
    """Convert a :class:`ComplexAmplitude` to a plain ``Tensor`` on-graph.

    When a field is produced by an ``OpticsModule`` (i.e. via ``__torch_dispatch__``),
    the autograd graph lives on the outer wrapper while the inner ``_data`` tensor is
    detached. Reading ``_data`` directly would therefore silently break gradient flow.
    This ``Function`` returns the inner values as a plain tensor in forward and routes
    the incoming gradient back through the wrapper (re-wrapped with the field geometry)
    in backward, so the result is a genuine real/complex tensor that still participates
    in optimization.
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
    straight from a graph-carrying tensor therefore creates a wrapper with no edge back
    to it, so a module that goes on to work through ``__torch_dispatch__`` records its
    own gradients against that leaf and the gradient never reaches the tensor the field
    was built from.

    Routing the crossing through this ``Function`` supplies the missing edge, so the
    graph survives in both directions. This is the pattern PyTorch's own wrapper
    subclasses use (compare ``DTensor._FromTorchTensor``).
    """

    @staticmethod
    def forward(
        ctx: FunctionCtx,
        data: Tensor,
        wavelength: Tensor,
        pixel_size: Tensor,
    ) -> ComplexAmplitude:
        # The field is built in the canonical layout, which may add axes that ``data``
        # does not have, so the gradient is reshaped back to what came in.
        ctx.input_shape = tuple(data.shape)
        # The inner tensor is detached: the graph belongs on the wrapper, which
        # autograd links back to ``data`` through this Function.
        return ComplexAmplitude(data.detach(), wavelength, pixel_size)

    @staticmethod
    def backward(ctx: FunctionCtx, grad: ComplexAmplitude) -> tuple[Tensor, None, None]:
        inner = grad._data if isinstance(grad, ComplexAmplitude) else grad
        # wavelength / pixel_size are geometry metadata and never differentiable.
        return inner.reshape(ctx.input_shape), None, None


def pixel_area(pixel_size: Tensor) -> Tensor:
    """The area of one pixel per wavelength, from a ``(n_wavelengths, 2)`` pitch.
    Always float64, whatever the field's dtype.
    """
    return (pixel_size[:, 0] * pixel_size[:, 1]).to(torch.float64).reshape(-1)


def _power_factor(
    current_power: Tensor, power: float | Tensor, device: torch.device
) -> Tensor:
    """The amplitude scale taking a field of ``current_power`` to ``power``.

    ``current_power`` is one value per ``(*batch, wavelength)``, and the scale
    multiplies a field laid out as ``(*batch, component, wavelength, H, W)``, so the
    component and spatial axes are inserted.
    """
    target_power = torch.as_tensor(power, dtype=torch.float64, device=device)
    factor = torch.sqrt(target_power / current_power)
    return factor[..., None, :, None, None]


@dataclass(frozen=True)
class FieldGeometry:
    """The sampling of a field: its wavelengths, pixel pitch, resolution and pose.

    ``wavelength`` is stored as ``(n_wavelengths,)`` and ``pixel_size`` as
    ``(n_wavelengths, 2)`` in ``(height, width)`` order.

    ``number_of_components`` says how many field components the plane carries, one for
    a scalar field and three for a field vector. On a field it is read from the data,
    so the two cannot disagree. On a bare geometry it is a declaration, and a module
    probe and :meth:`ComplexAmplitude.from_geometry` build from it.
    """

    wavelength: Float[Tensor, ""] | Float[Tensor, " n_wavelengths"]
    pixel_size: (
        Float[Tensor, " 2"] | Float[Tensor, "1 2"] | Float[Tensor, "n_wavelengths 2"]
    )
    resolution: tuple[int, int]
    origin: Float[Tensor, " 3"] | None = None
    rotation: Float[Tensor, "3 3"] | None = None
    number_of_components: int = SCALAR

    def __post_init__(self) -> None:
        wavelength = self.wavelength.reshape(-1)
        pixel_size = self.pixel_size.reshape(-1, 2)
        if self.number_of_components not in (SCALAR, VECTOR):
            raise ValueError(
                f"number_of_components must be {SCALAR} for a scalar field or "
                f"{VECTOR} for a field vector, got {self.number_of_components}."
            )
        if pixel_size.shape[0] not in (1, wavelength.numel()):
            raise ValueError(
                "pixel_size must have shape (2,), (1, 2) or (n_wavelengths, 2), got "
                f"{tuple(self.pixel_size.shape)} for {wavelength.numel()} wavelengths."
            )
        if pixel_size.shape[0] == 1 and wavelength.numel() > 1:
            pixel_size = pixel_size.expand(wavelength.numel(), 2)
        object.__setattr__(self, "wavelength", wavelength)
        object.__setattr__(self, "pixel_size", pixel_size)
        object.__setattr__(
            self, "resolution", tuple(int(length) for length in self.resolution)
        )

    @property
    def number_of_wavelengths(self) -> int:
        return self.wavelength.numel()

    @property
    def wavenumber(self) -> Float[Tensor, " n_wavelengths"]:
        return 2 * torch.pi / self.wavelength

    @property
    def spatial_extent(self) -> Float[Tensor, "n_wavelengths 2"]:
        resolution = torch.as_tensor(
            self.resolution,
            device=self.wavelength.device,
            dtype=self.wavelength.dtype,
        )
        return self.pixel_size * resolution

    def get_spatial_grid(
        self, index: int = 0
    ) -> tuple[Float[Tensor, "H W"], Float[Tensor, "H W"]]:
        return get_spatial_grid(
            resolution=self.resolution,
            pixel_size=self.pixel_size[index],
            device=self.wavelength.device,
        )

    @property
    def is_transverse(self) -> bool:
        """True if the samples lie in a plane parallel to the optical axis."""
        if self.rotation is None:
            return True
        normal = self.rotation[:, 2]
        return bool(
            torch.allclose(normal, normal.new_tensor([0.0, 0.0, 1.0]), atol=1e-9)
        )

    def positions(self, index: int = 0) -> Float[Tensor, "H W 3"]:
        """Sample positions in metres, ``(H, W, 3)``, in world coordinates.

        Args:
            index: Wavelength index.

        Returns:
            Tensor: The ``(x, y, z)`` position of every sample.
        """
        grid_x, grid_y = self.get_spatial_grid(index)
        points = torch.stack((grid_x, grid_y, torch.zeros_like(grid_x)), dim=-1)
        if self.rotation is not None:
            rotation = self.rotation.to(device=points.device, dtype=points.dtype)
            points = points @ rotation.transpose(-2, -1)
        if self.origin is not None:
            points = points + self.origin.to(device=points.device, dtype=points.dtype)
        return points

    @classmethod
    def cross_section(
        cls,
        wavelength: float | Float[Tensor, ""] | Float[Tensor, " n_wavelengths"],
        distances: Float[Tensor, " n_z"],
        transverse_pitch: float,
        width: int,
        axis: str = "x",
        offset: float = 0.0,
    ) -> FieldGeometry:
        """A plane along the propagation axis, the x-z or y-z section.

        Args:
            wavelength: Wavelength(s) in metres.
            distances: Where along ``z`` to sample, in metres.
            transverse_pitch: Pitch along the transverse axis, in metres.
            width: Number of samples across the transverse axis.
            axis: Which transverse axis the section runs along, ``"x"`` or ``"y"``.
            offset: Where the section sits on the other transverse axis, in metres.

        Returns:
            FieldGeometry: The section, posed so that ``positions()[..., 2]`` reproduces
                ``distances`` down the rows.

        Raises:
            ValueError: ``distances`` is not evenly spaced, or ``axis`` is not ``"x"`` 
                or ``"y"``.
        """
        if axis not in ("x", "y"):
            raise ValueError(f"axis must be 'x' or 'y', not {axis!r}.")

        distances = torch.as_tensor(distances)
        if distances.ndim != 1 or distances.numel() < 1:
            raise ValueError("distances must be a non-empty 1D tensor.")
        if not distances.dtype.is_floating_point:
            distances = distances.to(torch.get_default_dtype())

        steps = torch.diff(distances)
        step = steps[0] if steps.numel() else torch.zeros_like(distances[0])

        tolerance = 8 * torch.finfo(distances.dtype).eps * float(
            distances.abs().max() + step.abs()
        )
        if steps.numel() and float((steps - step).abs().max()) > tolerance:
            raise ValueError(
                "cross_section needs evenly spaced distances: the section is an "
                "evenly spaced grid, and an uneven z spacing is not one. Use "
                "torch.linspace, "
                "or sample the uneven planes one propagation at a time."
            )

        device = distances.device
        dtype = distances.dtype
        if not isinstance(wavelength, Tensor):
            wavelength = torch.tensor([wavelength], device=device, dtype=dtype)
        elif wavelength.ndim == 0:
            wavelength = wavelength.unsqueeze(0)

        number_of_planes = distances.numel()
        pixel_size = torch.tensor(
            [[float(step), transverse_pitch]], device=device, dtype=dtype
        )

        transverse = [1.0, 0.0, 0.0] if axis == "x" else [0.0, 1.0, 0.0]
        along_z = [0.0, 0.0, 1.0]
        normal = [0.0, -1.0, 0.0] if axis == "x" else [1.0, 0.0, 0.0]
        rotation = torch.tensor(
            [transverse, along_z, normal], device=device, dtype=dtype
        ).transpose(0, 1)

        centre = [offset, 0.0] if axis == "y" else [0.0, offset]
        origin = torch.tensor(
            [*centre, float(distances[number_of_planes // 2])],
            device=device,
            dtype=dtype,
        )

        return cls(
            wavelength=wavelength,
            pixel_size=pixel_size,
            resolution=(number_of_planes, width),
            origin=origin,
            rotation=rotation,
        )


# Reductions that take a dim and can therefore remove a reserved axis.
_REDUCTIONS = (
    torch.ops.aten.sum.dim_IntList,
    torch.ops.aten.mean.dim,
    torch.ops.aten.amax.default,
    torch.ops.aten.amin.default,
    torch.ops.aten.prod.dim_int,
)


def _touches_reserved_axis(reduced: Any, ndim: int) -> bool:
    """Whether a reduction over ``reduced`` removes the component or wavelength axis."""
    if reduced is None:
        # Reducing every axis at once.
        return True
    axes = (reduced,) if isinstance(reduced, int) else tuple(reduced)
    if not axes:
        return True
    normalized = {axis if axis >= 0 else axis + ndim for axis in axes}
    return bool(normalized & {ndim + COMPONENT_DIM, ndim + WAVELENGTH_DIM})


def _same_pose(one: FieldGeometry, other: FieldGeometry) -> bool:
    """Whether two geometries sit in the same place, facing the same way."""
    for left, right in ((one.origin, other.origin), (one.rotation, other.rotation)):
        if (left is None) != (right is None):
            return False
        if left is not None and not torch.allclose(left, right.to(left)):
            return False
    return True


class ComplexAmplitude(Tensor):
    """An electric field: complex values carrying the geometry that gives them meaning.

    A tensor subclass, so a field can be multiplied, propagated and differentiated like
    any other tensor, while :attr:`wavelength` and :attr:`pixel_size` travel with it.
    An :class:`~hologradpy.optics.modules.abstract.OpticsModule` therefore reads
    the sampling off the field it is given.

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

        inner, wavelength, pixel_size = cls._sanitize_inputs(
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

        data, wavelength, pixel_size = self._sanitize_inputs(
            data, wavelength, pixel_size
        )

        # Optionally scale the field to an absolute power (watts) at construction.
        if power is not None:
            data = self._scale_to_power(data, pixel_size, power)

        self._data: Tensor = data
        self.geometry: FieldGeometry = FieldGeometry(
            wavelength,
            pixel_size,
            data.shape[-2:],
            number_of_components=data.shape[COMPONENT_DIM],
        )

    @staticmethod
    def _sanitize_inputs(
        data: Tensor,
        wavelength: float | Tensor,
        pixel_size: tuple[float, float] | Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """The data in the canonical layout, with the geometry metadata normalized."""
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

        return _to_canonical_layout(data, wavelength.numel()), wavelength, pixel_size

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
            data: Field values ``(..., H, W)``, promoted to the canonical layout. If
                ``None``, a uniform unit-amplitude field (ones) is created at the
                geometry's resolution, carrying its component and wavelength count.
            dtype: Dtype of the default field when ``data`` is ``None``.
            power: If given, scale the field to this absolute power (watts).

        Returns:
            ComplexAmplitude: A field carrying the geometry's wavelength and
            pixel size.
        """
        if data is None:
            data = torch.ones(
                (
                    geometry.number_of_components,
                    geometry.number_of_wavelengths,
                    *geometry.resolution,
                ),
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
    def number_of_components(self) -> int:
        """How many field components this carries, one for a scalar field and three for
        a field vector ``(E_x, E_y, E_z)``.
        """
        return self.geometry.number_of_components

    @property
    def batch_shape(self) -> tuple[int, ...]:
        """Leading batch dimensions, before ``(component, wavelength, H, W)``.

        An empty tuple for a single unbatched field.
        """
        return tuple(self.shape[:COMPONENT_DIM])

    @property
    def is_scalar(self) -> bool:
        """True while this carries a single field component."""
        return self.number_of_components == SCALAR

    @property
    def is_vector(self) -> bool:
        """True while this carries the three components of a field vector."""
        return self.number_of_components == VECTOR

    @property
    def spatial_extent(self) -> Tensor:
        return self.geometry.spatial_extent

    def get_spatial_grid(self) -> tuple[Tensor, Tensor]:
        return self.geometry.get_spatial_grid()

    def component(self, index: int) -> ComplexAmplitude:
        """One field component, still a field.

        Args:
            index: Which component, ``0`` for ``E_x``, ``1`` for ``E_y`` and ``2`` for
                ``E_z``, in the plane's own frame.

        Returns:
            ComplexAmplitude: A scalar field on the same plane. The component axis is
                kept at length one, so the layout survives.
        """
        if not 0 <= index < self.number_of_components:
            raise IndexError(
                f"Component {index} of a field carrying "
                f"{self.number_of_components}."
            )
        return self.narrow(COMPONENT_DIM, index, 1)

    def at_wavelength(self, index: int) -> ComplexAmplitude:
        """The field at one wavelength, carrying that wavelength's own pitch.

        Args:
            index: Which wavelength.

        Returns:
            ComplexAmplitude: A single-wavelength field. The wavelength axis is kept at
                length one, so the layout survives.
        """
        if not 0 <= index < self.number_of_wavelengths:
            raise IndexError(
                f"Wavelength {index} of a field carrying "
                f"{self.number_of_wavelengths}."
            )
        return self.narrow(WAVELENGTH_DIM, index, 1)

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
        """Real-valued intensity, differentiable and on-graph.

        The components are summed, so this is the irradiance ``sum_c |E_c|**2`` that a
        detector responds to, shaped ``(*batch, n_wavelengths, H, W)``. Use
        :attr:`amplitude` for the per-component magnitudes.

        Computed as ``real**2 + imag**2`` to avoid the gradient singularity of ``abs()``
        at zero field.
        """
        field = self.as_tensor()
        return (field.real**2 + field.imag**2).sum(dim=COMPONENT_DIM)

    @staticmethod
    def _integrate_power(intensity: Tensor, pixel_size: Tensor) -> Tensor:
        """Integrate intensity over area, giving optical power per
        ``(*batch, wavelength)``.

        ``intensity`` has already been summed over the components, so the power is that
        of the whole field. Reduces over ``(H, W)`` in float64 for precision.
        ``pixel_size`` is ``(n_wavelengths, 2)``.
        """
        area = pixel_area(pixel_size)
        return intensity.to(torch.float64).sum(dim=(-2, -1)) * area

    @classmethod
    def _scale_to_power(
        cls, data: Tensor, pixel_size: Tensor, power: float | Tensor
    ) -> Tensor:
        """Return ``data`` scaled so its integrated power equals ``power`` (W),
        preserving phase. The scale ratio is computed in float64.
        """
        intensity = (data.real**2 + data.imag**2).sum(dim=COMPONENT_DIM)
        current_power = cls._integrate_power(intensity, pixel_size)
        factor = _power_factor(current_power, power, data.device)
        return data * factor.to(corresponding_real_dtype(data.dtype))

    def power(self) -> Tensor:
        """Total optical power = integral of intensity over area
        (``sum(|E|^2) * pixel_area``), returned per ``(*batch, wavelength)``. The
        components are summed, so a field vector reports the power it carries in total.

        In SI this is watts when the field amplitude is in ``sqrt(W/m^2)`` and
        ``pixel_size`` in metres. The reduction is performed in float64.

        Raises:
            ValueError: The field is not sampled on a transverse plane. 
        """
        if not self.geometry.is_transverse:
            raise ValueError(
                "power() is the flux through a transverse plane, and this field is "
                "sampled on one that is tilted out of x-y"
            )
        return self._integrate_power(self.intensity, self.pixel_size)

    def with_power(self, power: float | Tensor) -> ComplexAmplitude:
        """Return this field scaled so ``power() == power``, preserving phase and the
        autograd graph.

        ``power`` is matched per ``(*batch, wavelength)``, summed over the components.
        """
        factor = _power_factor(self.power(), power, self.device)
        return self * factor.to(self.dtype_r)

    def with_polarization(
        self, jones: Sequence[complex] | Tensor
    ) -> ComplexAmplitude:
        """This scalar field given a polarization, as a field vector.

        The scalar amplitude is shared by the three components in the ratios ``jones``
        gives, so a field of unit Jones vector keeps the power it had.

        Args:
            jones: The Jones vector ``(J_x, J_y, J_z)`` in the plane's own frame. A
                real or complex sequence, or a tensor of three values.

        Returns:
            ComplexAmplitude: A field vector on the same plane.

        Raises:
            ValueError: This already carries a field vector, or ``jones`` is not three
                values.
        """
        if self.is_vector:
            raise ValueError(
                "This already carries a field vector. Build one from three scalar "
                "fields with from_components()."
            )
        weights = torch.as_tensor(jones, device=self.device).to(self.dtype_c)
        if weights.shape != (VECTOR,):
            raise ValueError(
                f"A Jones vector is {VECTOR} values (J_x, J_y, J_z), got shape "
                f"{tuple(weights.shape)}."
            )
        return self * weights.reshape(VECTOR, 1, 1, 1)

    @classmethod
    def from_components(
        cls,
        x: ComplexAmplitude,
        y: ComplexAmplitude,
        z: ComplexAmplitude,
    ) -> ComplexAmplitude:
        """A field vector from its three components.

        Args:
            x: The ``E_x`` component, a scalar field.
            y: The ``E_y`` component, on the same plane.
            z: The ``E_z`` component, on the same plane.

        Returns:
            ComplexAmplitude: A field vector carrying all three.

        Raises:
            ValueError: One of them is not a scalar field.
        """
        parts = (x, y, z)
        for axis, part in zip("xyz", parts):
            if not part.is_scalar:
                raise ValueError(
                    f"The {axis} component carries "
                    f"{part.number_of_components} components, and each one of a field "
                    "vector is a scalar field."
                )

        return torch.cat(parts, dim=COMPONENT_DIM)

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
            f"number_of_components={self.number_of_components}, "
            f"dtype={self._data.dtype}, "
            f"wavelength={self.wavelength}, "
            f"pixel_size={self.pixel_size})"
        )

    def with_geometry(
        self,
        wavelength: float | Tensor | None = None,
        pixel_size: tuple[float, float] | Tensor | None = None,
        origin: Tensor | None = None,
        rotation: Tensor | None = None,
    ) -> ComplexAmplitude:
        """Return this ComplexAmplitude with updated wavelength / pixel_size metadata,
        preserving the autograd graph.

        Only the ``geometry`` attribute is replaced, so the existing tensor wrapper, and
        the ``grad_fn``, stays intact.

        Args:
            wavelength: New wavelength(s). If *None*, the existing value is
                kept.
            pixel_size: New pixel size(s). If *None*, the existing value is
                kept.
            origin: Where the grid centre sits, ``(3,)`` in metres. If
                *None*, the existing value is kept.
            rotation: Grid axes as the columns of a ``(3, 3)``. If *None*,
                the existing value is kept.

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

        new_geometry = replace(
            self.geometry,
            wavelength=wavelength,
            pixel_size=pixel_size,
            origin=self.geometry.origin if origin is None else origin,
            rotation=self.geometry.rotation if rotation is None else rotation,
        )
        # Use object.__setattr__ to bypass any tensor attribute-setting
        # restrictions while keeping the autograd graph intact.
        object.__setattr__(self, "geometry", new_geometry)
        return self

    def flatten_batch(self, fold_components: bool = True) -> tuple[Tensor, BatchSpec]:
        """Collapse the leading axes into one, for a fixed-rank operation.

        This is the entry point for ND batch support in an
        :class:`~hologradpy.optics.modules.abstract.OpticsModule`: flatten, run the
        fixed-rank operation, then restore with :meth:`unflatten_batch`.

        By default, the component axis is folded into the leading axis along with the
        batch, giving ``(N, n_wavelengths, H, W)`` with
        ``N = prod(batch) * n_components``. A module that acts on each component
        independently, which is every propagator and every scalar phase or amplitude
        mask, therefore needs no knowledge of components at all.

        Args:
            fold_components: Fold the component axis into the leading axis. Pass
                False to keep it, giving ``(N, n_components, n_wavelengths, H, W)``.

        Returns:
            tuple[Tensor, BatchSpec]: The flattened tensor, sharing storage with the
            field wherever the layout allows, and the spec needed to restore the rank.
        """
        height, width = self.resolution
        n_wavelengths = self.number_of_wavelengths
        spec = BatchSpec(
            batch_shape=self.batch_shape,
            number_of_components=self.number_of_components,
            original_ndim=self.ndim,
        )
        data = self.as_tensor()

        if fold_components:
            return data.reshape(-1, n_wavelengths, height, width), spec
        return (
            data.reshape(
                -1, self.number_of_components, n_wavelengths, height, width
            ),
            spec,
        )

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
            # Sanitized for the metadata alone. The data itself is passed through as it
            # came.
            _, wavelength, pixel_size = cls._sanitize_inputs(
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
        number_of_components: int | None = None,
    ) -> ComplexAmplitude:
        """Restore a flattened tensor to the rank recorded in ``spec``.

        Inverse of :meth:`flatten_batch`, and it accepts either of that method's two
        layouts since both hold the same values in the same order. The output spatial
        resolution is taken from ``data``, so it may differ from the input after a
        resampling propagator, while the batch, component and wavelength axes are
        restored as they were.

        Args:
            data: The flattened tensor, ``(N, n_wavelengths, H_out, W_out)`` or
                ``(N, n_components, n_wavelengths, H_out, W_out)``.
            spec: Layout captured by :meth:`flatten_batch`.
            wavelength: Output wavelength(s).
            pixel_size: Output pixel size(s).
            number_of_components: How many components the result carries, for a
                module that changes their number. Defaults to what the spec
                recorded.

        Returns:
            ComplexAmplitude: Field with the same rank as the original input.
        """
        n_wavelengths = data.shape[WAVELENGTH_DIM]
        height_out, width_out = data.shape[-2:]
        if number_of_components is None:
            number_of_components = spec.number_of_components

        out = data.reshape(
            *spec.batch_shape,
            number_of_components,
            n_wavelengths,
            height_out,
            width_out,
        )

        # from_tensor, not the constructor: a resampling module reaches here with
        # an on-graph tensor, and the constructor would strand it on a leaf.
        return cls.from_tensor(out, wavelength, pixel_size)

    @classmethod
    def _wrap_like(
        cls,
        data: Tensor,
        geometry: FieldGeometry,
        wavelength: Tensor | None = None,
        pixel_size: Tensor | None = None,
    ) -> ComplexAmplitude:
        """Wrap ``data`` with ``geometry``'s pose, taking its wavelength and pitch
        unless a slicing operation supplied its own.
        """
        out = cls(
            data,
            geometry.wavelength if wavelength is None else wavelength,
            geometry.pixel_size if pixel_size is None else pixel_size,
        )
        if geometry.origin is None and geometry.rotation is None:
            return out
        moved = replace(
            out.geometry,
            origin=None if geometry.origin is None else geometry.origin.to(out.device),
            rotation=(
                None if geometry.rotation is None else geometry.rotation.to(out.device)
            ),
        )
        object.__setattr__(out, "geometry", moved)
        return out

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
                if not _same_pose(field.geometry, geometry):
                    raise ValueError(
                        "ComplexAmplitude arguments must be sampled on the same "
                        "plane: one carries a different origin or rotation, so a "
                        "cross section and a transverse field cannot be combined."
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

        # Slicing the wavelength axis carries the metadata with it.
        if func == torch.ops.aten.slice.Tensor:
            input_tensor, dim, start, end, step = _slice_arguments(args, kwargs)

            dim = dim if dim >= 0 else dim + input_tensor.ndim

            out = func(*tree_map(unwrap, args), **tree_map(unwrap, kwargs))

            if dim == input_tensor.ndim + COMPONENT_DIM:
                if out.shape[COMPONENT_DIM] not in (SCALAR, VECTOR):
                    return out
                new_wavelength = geometry.wavelength
                new_pixel_size = geometry.pixel_size
            elif dim == input_tensor.ndim + WAVELENGTH_DIM:
                new_wavelength = geometry.wavelength[start:end:step]
                new_pixel_size = geometry.pixel_size[start:end:step]
            else:
                new_wavelength = geometry.wavelength
                new_pixel_size = geometry.pixel_size

            return cls._wrap_like(out, geometry, new_wavelength, new_pixel_size)
        elif func == torch.ops.aten.select.int:
            input_tensor, dim, _ = args

            dim = dim if dim >= 0 else dim + input_tensor.ndim

            out = func(*tree_map(unwrap, args), **kwargs)

            # Selecting along a reserved axis drops it, which leaves the layout, so the
            # result is plain values. Use narrow (a slice) to keep the field.
            if dim in (
                input_tensor.ndim + COMPONENT_DIM,
                input_tensor.ndim + WAVELENGTH_DIM,
            ):
                return out

            return cls._wrap_like(out, geometry)
        elif func in _REDUCTIONS:
            input_tensor = args[0]
            reduced = args[1] if len(args) > 1 else kwargs.get("dim")
            keepdim = args[2] if len(args) > 2 else kwargs.get("keepdim", False)

            out = func(*tree_map(unwrap, args), **tree_map(unwrap, kwargs))

            # Reducing a reserved axis away leaves the layout behind, whatever the
            # resulting shape happens to look like.
            if not keepdim and _touches_reserved_axis(reduced, input_tensor.ndim):
                return out
            return cls._wrap_like(out, geometry) if isinstance(out, Tensor) else out
        elif func in (
            torch.ops.aten.to.device,
            torch.ops.aten._to_copy.default,
        ):
            out = func(*tree_map(unwrap, args), **kwargs)
            new_wavelength = geometry.wavelength.to(out.device)
            new_pixel_size = geometry.pixel_size.to(out.device)
            return cls._wrap_like(out, geometry, new_wavelength, new_pixel_size)
        else:
            pass

        out = func(*tree_map(unwrap, args), **tree_map(unwrap, kwargs))

        n_wavelength = geometry.wavelength.shape[0]

        def should_wrap_tensor(x: Tensor) -> bool:
            """Whether a result still has the layout of a field."""
            if isinstance(x, cls):
                return False
            if x.ndim < 4:
                return False
            return (
                x.shape[WAVELENGTH_DIM] == n_wavelength
                and x.shape[COMPONENT_DIM] in (SCALAR, VECTOR)
            )

        def wrap_output(x: Any) -> Any:
            if isinstance(x, Tensor):
                if should_wrap_tensor(x):
                    return cls._wrap_like(x, geometry)
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
