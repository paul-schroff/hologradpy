"""The fits of a zeroth-order calibration: the missed field on every pixel, then the
kernel of the field no hologram changes and the weight of the light at the wraps.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch
from torch import Tensor

from ...optimization import solve_linear_least_squares


def probe_phase(
    phase: Tensor, corrective: Tensor, amplitude: float, piston: float
) -> Tensor:
    """The phase of the sum of a hologram and a replica of the beam, as Eq. 9 of
    Cerioni et al. (arXiv:2601.20376) displays it.

    Args:
        phase: The hologram's phase in radians.
        corrective: The replica's phase in radians, on the SLM's grid.
        amplitude: The replica's amplitude, as a multiple of the hologram's.
        piston: The phase added to the replica, in radians.

    Returns:
        Tensor: The displayed phase in radians.
    """
    return torch.angle(
        torch.exp(1j * phase) + amplitude * torch.exp(1j * (corrective + piston))
    )


@dataclass
class MissedField:
    """The field the model misses on every pixel of a region, and the intensity that
    interferes with nothing there.

    Attributes:
        field: The complex field, zero where it was not resolved.
        floor: The intensity, zero where it was not resolved.
        resolved: True on the pixels the probes resolve.
    """

    field: Tensor
    floor: Tensor
    resolved: Tensor


def fit_missed_field(
    frames: Tensor,
    model_fields: Tensor,
    region: Tensor,
    valid: Tensor | None = None,
    contrast: float = 1e-3,
) -> MissedField:
    """The field the model misses and the floor, from frames of probes that change the
    model's field.

    On every pixel, ``I_k = |M_k + U|^2 + F`` is linear in ``|U|^2 + F``, ``Re U`` and
    ``Im U``, given the model's field ``M_k`` of probe ``k``. A least-squares solve over
    the probes gives ``U`` and ``F`` there. A pixel is left out when the model's fields
    of the probes vary there by less than ``contrast`` of their largest variation in
    the region, since the probes do not resolve it. A pixel with fewer valid
    measurements than the three unknowns is left out as well.

    Args:
        frames: The frames, in the model's units, ``(K, H, W)``.
        model_fields: The model's fields of the same probes, ``(K, H, W)``.
        region: True on the pixels to solve, ``(H, W)``.
        valid: True on the measurements to use, ``(K, H, W)``, or None for all. An
            overexposed pixel of a frame is left out of the solve of that pixel.
        contrast: The smallest variation that resolves a pixel, as a fraction of the
            largest in the region.

    Returns:
        MissedField: The field, the floor and the resolved pixels, on the frames' grid.
    """
    rows, columns = torch.nonzero(region, as_tuple=True)
    intensity = frames[:, rows, columns].double()
    model = model_fields[:, rows, columns].to(torch.complex128)
    weight = (
        torch.ones_like(intensity)
        if valid is None
        else valid[:, rows, columns].to(intensity.dtype)
    )
    design = torch.stack(
        [torch.ones_like(model.real), 2 * model.real, 2 * model.imag], dim=-1
    ) * weight[..., None]
    right_hand_side = (intensity - model.abs().square()) * weight
    solution = torch.linalg.lstsq(
        design.permute(1, 0, 2), right_hand_side.T[..., None]
    ).solution[..., 0]
    variation = (model - model.mean(dim=0)).abs().amax(dim=0)
    resolved = (variation > contrast * variation.max()) & (weight.sum(dim=0) >= 3)
    pixel_field = torch.complex(solution[:, 1], solution[:, 2])
    pixel_floor = solution[:, 0] - pixel_field.abs().square()

    shape = tuple(frames.shape[-2:])
    field = torch.zeros(shape, dtype=model_fields.dtype, device=frames.device)
    floor = torch.zeros(shape, dtype=frames.dtype, device=frames.device)
    full_resolved = torch.zeros(shape, dtype=torch.bool, device=frames.device)
    kept_rows, kept_columns = rows[resolved], columns[resolved]
    field[kept_rows, kept_columns] = pixel_field[resolved].to(field.dtype)
    floor[kept_rows, kept_columns] = pixel_floor[resolved].to(floor.dtype)
    full_resolved[kept_rows, kept_columns] = True
    return MissedField(field, floor, full_resolved)


@dataclass
class KernelFit:
    """The kernel and the wrap weight that describe the measured fields.

    Attributes:
        kernel: The image-plane kernel of the field no hologram changes.
        wrap_weight: The field scattered at one wrapped edge.
        relative_residual: The norm of what the two leave of the measured fields, over
            the norm of the measured fields.
    """

    kernel: Tensor
    wrap_weight: complex
    relative_residual: float


def fit_kernel_and_wrap_weight(
    measured: Tensor,
    wrap_fields: Tensor,
    masks: Tensor,
    kernel_focal_field: Callable[[Tensor], Tensor],
    kernel_focal_field_adjoint: Callable[[Tensor], Tensor],
) -> KernelFit:
    """Fit ``U_g = L(K) + w G_g`` to the fields measured with every grating ``g``.

    ``L(K)`` is the focal field of the kernel, the same for every grating, and ``G_g``
    the focal field of the beam on the wrapped edges of that grating's probes. The
    weight ``w`` is eliminated. The kernel is fitted to the measured fields and to the
    wrap fields by :func:`~hologradpy.optimization.solve_linear_least_squares`, and
    ``w`` is the ratio of the parts of the two that the kernel leaves. The kernel and
    one complex number have curvatures far apart, which this leaves out of the solve.

    Args:
        measured: The measured fields ``U_g``, ``(G, H, W)``.
        wrap_fields: The wrap fields ``G_g``, ``(G, H, W)``.
        masks: True on the pixels each field was resolved on, ``(G, H, W)``.
        kernel_focal_field: The focal field ``(H, W)`` of a kernel.
        kernel_focal_field_adjoint: Its adjoint, from ``(H, W)`` to the kernel.

    Returns:
        KernelFit: The kernel, the weight and the residual.
    """
    weights = masks.to(measured.dtype)

    def forward(kernel: Tensor) -> Tensor:
        return kernel_focal_field(kernel)[None] * weights

    def adjoint(stacked: Tensor) -> Tensor:
        return kernel_focal_field_adjoint((stacked * weights).sum(dim=0))

    data = measured * weights
    basis = wrap_fields * weights
    kernel_of_data, _ = solve_linear_least_squares(forward, adjoint, data)
    kernel_of_basis, _ = solve_linear_least_squares(forward, adjoint, basis)
    with torch.no_grad():
        left_of_data = (data - forward(kernel_of_data)).flatten()
        left_of_basis = (basis - forward(kernel_of_basis)).flatten()
        wrap_weight = torch.vdot(left_of_basis, left_of_data) / torch.vdot(
            left_of_basis, left_of_basis
        )
        kernel = kernel_of_data - wrap_weight * kernel_of_basis
        residual = data - forward(kernel) - wrap_weight * basis
        relative_residual = float(residual.norm() / data.norm())
    return KernelFit(kernel, complex(wrap_weight), relative_residual)
