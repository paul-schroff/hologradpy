from __future__ import annotations

import operator
from functools import reduce

from typing import TYPE_CHECKING

import torch
import torch.nn as nn

from .vector_fields import forward_difference

if TYPE_CHECKING:
    from .optics.modules.slm_fields import SLMField


INTENSITY_MSE_SCALE = 1e12


def smallest_divisor(tensor: torch.Tensor) -> float:
    """The smallest value it is safe to divide ``tensor`` by, for its own dtype."""
    return torch.finfo(tensor.dtype).smallest_normal


def normalize_to_unit_sum(images: torch.Tensor) -> torch.Tensor:
    """Scale every image in a batch to unit total intensity."""
    total = images.sum(dim=(-2, -1), keepdim=True)
    return images / total.clamp_min(smallest_divisor(total))


def normalize_single_to_unit_sum(image: torch.Tensor) -> torch.Tensor:
    """Scale one image to unit total intensity, summing to a scalar."""
    total = image.sum()
    return image / total.clamp_min(smallest_divisor(total))


def masked_intensity_mse(
    field: torch.Tensor,
    target_intensity: torch.Tensor,
    mask: torch.Tensor,
    region_pixel_count: float,
) -> torch.Tensor:
    """Squared error between a predicted and a measured intensity over a region.

    Both are masked and normalized to unit sum first, so the comparison is of intensity
    distribution rather than absolute counts.

    Args:
        field: Predicted complex field, ``(batch, height, width)``.
        target_intensity: Measured intensity over the same region.
        mask: Region of interest, broadcast against both.
        region_pixel_count: How many pixels the mask selects.

    Returns:
        torch.Tensor: The mismatch, averaged over the batch.
    """
    number_of_targets = target_intensity.shape[-3]
    intensity = field.abs() ** 2 * mask

    target_intensity = target_intensity * mask
    return (
        region_pixel_count
        * nn.functional.mse_loss(
            normalize_to_unit_sum(intensity),
            normalize_to_unit_sum(target_intensity),
            reduction="sum",
        )
        / number_of_targets
    )


# Cost functions


class LossFunction:
    """A term measuring a predicted field, addable to other terms.

    Terms compose with ``+``, which is how a data term and a prior are combined, or how
    several hologram-search costs are weighted against one another. Each term carries
    its own ``scale``, so a sum needs none of its own.
    """

    name: str = ""
    scale: float = 1.0

    def __call__(
        self, field: torch.Tensor | None = None, target: torch.Tensor | None = None
    ) -> torch.Tensor:
        return self.scale * self.evaluate(field, target)

    def evaluate(
        self, field: torch.Tensor | None = None, target: torch.Tensor | None = None
    ) -> torch.Tensor:
        """The unweighted cost, which ``__call__`` scales. Implemented by every term."""
        raise NotImplementedError(
            f"evaluate has not been implemented for {type(self).__name__}."
        )

    def components(
        self, field: torch.Tensor | None = None, target: torch.Tensor | None = None
    ) -> dict[str, torch.Tensor]:
        """The named parts this term is made of, whose sum is what ``__call__`` returns.

        Args:
            field: Predicted field, as :meth:`__call__` takes it.
            target: Measured target, for the terms that take one.

        Returns:
            dict[str, torch.Tensor]: One entry per term.
        """
        return {self.name or type(self).__name__: self(field, target)}

    def __add__(self, other: LossFunction) -> SumOfLosses:
        return SumOfLosses(self, other)


class SumOfLosses(LossFunction):
    """Every term evaluated on the same arguments and summed."""

    def __init__(self, *terms: LossFunction) -> None:
        self.terms: tuple[LossFunction, ...] = terms

    def _term_components(
        self, field: torch.Tensor | None, target: torch.Tensor | None
    ) -> dict[str, torch.Tensor]:
        """Each term's own parts, before this sum's scale."""
        merged: dict[str, torch.Tensor] = {}
        for term in self.terms:
            for label, value in term.components(field, target).items():
                merged[_unused_label(merged, label)] = value
        return merged

    def components(
        self, field: torch.Tensor | None = None, target: torch.Tensor | None = None
    ) -> dict[str, torch.Tensor]:
        parts = self._term_components(field, target)
        if self.scale == 1.0:
            return parts
        return {label: self.scale * value for label, value in parts.items()}

    def evaluate(
        self, field: torch.Tensor | None = None, target: torch.Tensor | None = None
    ) -> torch.Tensor:
        return reduce(operator.add, self._term_components(field, target).values())

    def __add__(self, other: LossFunction) -> SumOfLosses:
        return SumOfLosses(*self.terms, other)


def _unused_label(labels: dict[str, torch.Tensor], label: str) -> str:
    """``label``, suffixed if it is taken, so a merge never drops a term."""
    if label not in labels:
        return label

    suffix = 2
    while f"{label} ({suffix})" in labels:
        suffix += 1
    return f"{label} ({suffix})"


class MaskedIntensityMSE(LossFunction):
    """Error between the predicted intensity and the measured frame, over a region.

    :class:`LossIntensityMSE` measures the same quantity against a fixed target held
    construction rather than a measured frame per batch, and without the region scaling
    (see :func:`masked_intensity_mse`), so the two are not interchangeable weights.
    """

    name = "intensity mse"

    def __init__(self, mask: torch.Tensor, scale: float = 1.0) -> None:
        """
        Args:
            mask: Region of interest, already cropped to its bounding box and in the
                model's dtype and device.
            scale: Weight of this term.
        """
        self.mask: torch.Tensor = mask
        self.region_pixel_count: float = float(mask.sum())
        self.scale: float = scale

    def evaluate(
        self, field: torch.Tensor | None = None, target: torch.Tensor | None = None
    ) -> torch.Tensor:
        return masked_intensity_mse(
            field, target, self.mask, self.region_pixel_count
        )


class PhaseSmoothness(LossFunction):
    """Penalize structure in the SLM-plane phase.

    An SLM-plane field stored one value per pixel can fit the speckle with a rough,
    unphysical solution, so it needs a prior. A parameterization that is band limited by
    construction, such as a compact point spread function kernel, does not, and simply
    goes without one.

    Ignores ``field`` and ``target``: it reads the field module it was built on, which
    is what makes it addable to a data term.
    """

    name = "phase smoothness"

    def __init__(self, slm_field: SLMField, scale: float = 1e-3) -> None:
        """
        Args:
            slm_field: The field module being fitted, carrying ``phase``.
            scale: Weight of this penalty, relative to the data term.
        """
        self.slm_field = slm_field
        self.scale: float = scale

    def evaluate(
        self, field: torch.Tensor | None = None, target: torch.Tensor | None = None
    ) -> torch.Tensor:
        return gradient_loss(self.slm_field.phase)


class AmplitudeSmoothness(LossFunction):
    """Penalize structure in the SLM-plane amplitude.

    The amplitude counterpart of :class:`PhaseSmoothness`, and the other half of the
    prior a per-pixel field needs. Weighted separately, since the two quantities are
    independent.
    """

    name = "amplitude smoothness"

    def __init__(self, slm_field: SLMField, scale: float = 1e-3) -> None:
        """
        Args:
            slm_field: The field module being fitted, carrying ``amplitude``.
            scale: Weight of this penalty, relative to the data term.
        """
        self.slm_field = slm_field
        self.scale: float = scale

    def evaluate(
        self, field: torch.Tensor | None = None, target: torch.Tensor | None = None
    ) -> torch.Tensor:
        # Scale-free, so the weight means the same thing at any beam brightness.
        unit_amplitude = self.slm_field.amplitude.abs()
        mean_amplitude = unit_amplitude.mean()
        unit_amplitude = unit_amplitude / mean_amplitude.clamp_min(
            smallest_divisor(mean_amplitude)
        )
        return gradient_loss(unit_amplitude)


def gradient_loss(input: torch.Tensor, aperture_relative: bool = True) -> torch.Tensor:
    """Mean squared finite difference, measured across the aperture by default.

    Args:
        input: A ``(..., height, width)`` field to penalize.
        aperture_relative: Scale each difference by the pixel count along its
            axis. Pass False for the raw per-pixel differences.
    """
    gradient_x, gradient_y = forward_difference(input)
    if aperture_relative:
        gradient_x = gradient_x * input.shape[-1]
        gradient_y = gradient_y * input.shape[-2]
    return torch.mean(gradient_x**2) + torch.mean(gradient_y**2)


class LossIntensityMSE(LossFunction):
    """Squared error between the produced and target intensity, over the signal region.

    Compares intensity alone and says nothing about phase, so a retrieval driven by
    this term is free to choose whatever image-plane phase suits it.
    """

    def __init__(
        self,
        target_intensity: torch.Tensor,
        signal_mask: torch.Tensor,
        scale: float = INTENSITY_MSE_SCALE,
    ) -> None:
        """Amplitude-only cost function from https://doi.org/10.1364/OE.22.026548.

        Args:
            target_intensity: Target intensity pattern.
            signal_mask: Binary mask containing signal region.
            scale: Weight of this term, by default 1e12.
        """
        self.mse = nn.MSELoss(reduction="sum")
        self.signal_mask = signal_mask
        self.scale: float = scale

        self.target_intensity = normalize_single_to_unit_sum(
            target_intensity * signal_mask
        )

    def evaluate(
        self, field: torch.Tensor | None = None, target: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Calculate the loss based on the complex amplitude at the image plane.

        Args:
            field: Complex amplitude at the image plane.
            target: Ignored. The target was fixed at construction.

        Returns:
            torch.Tensor: Cost.
        """
        intensity_out = field.abs() ** 2 * self.signal_mask
        intensity_out = normalize_single_to_unit_sum(intensity_out)
        return self.mse(intensity_out, self.target_intensity)


class LossFidelity(LossFunction):
    """Overlap of the produced field with a target amplitude and phase.

    Measures amplitude and phase together, so unlike :class:`LossIntensityMSE` it
    constrains the image-plane phase as well.

    Equation (5) of https://doi.org/10.1364/OE.25.011692, where the intensity and the
    target are each normalized over the signal region before being overlapped.
    """

    def __init__(
        self,
        target_intensity: torch.Tensor,
        target_phase: torch.Tensor,
        signal_mask: torch.Tensor,
        scale: float = 1e12,
    ) -> None:
        """Phase and amplitude cost function from https://doi.org/10.1364/OE.25.011692.

        Args:
            target_intensity: Target intensity pattern.
            target_phase: Target phase pattern.
            signal_mask: Binary mask containing signal region.
            scale: Weight of this term, the paper's ``10**d``.
        """
        self.scale: float = scale
        self.signal_mask = signal_mask
        self.target_intensity = target_intensity * signal_mask
        self.target_amplitude = self.target_intensity.sqrt()
        self.target_phase = target_phase * signal_mask

    def evaluate(
        self, field: torch.Tensor | None = None, target: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Calculate the loss based on the electric field.

        Args:
            field: Electric field at the image plane.
            target: Ignored. The target was fixed at construction.

        Returns:
            torch.Tensor: Cost.
        """
        amplitude_out = field.abs()
        phase_out = field.angle()

        overlap = (
            self.signal_mask
            * amplitude_out
            * self.target_amplitude
            * (phase_out - self.target_phase).cos()
        ).sum()
        power_out = ((amplitude_out * self.signal_mask) ** 2).sum()
        overlap = overlap / (self.target_intensity.sum() * power_out).sqrt().clamp_min(
            torch.finfo(amplitude_out.dtype).tiny
        )

        return (1 - overlap) ** 2


class LossAbsoluteFidelity(LossFunction):
    """Squared error against a target field, in absolute intensity units."""

    def __init__(
        self,
        target_intensity: torch.Tensor,
        target_phase: torch.Tensor,
        signal_mask: torch.Tensor,
        scale: float | None = None,
    ) -> None:
        """
        Args:
            target_intensity: Target intensity in the same units the model produces.
            target_phase: Target phase in radians.
            signal_mask: Binary mask containing the signal region.
            scale: Weight of this term.
        """
        self.signal_mask = signal_mask
        self.target_intensity = target_intensity * signal_mask
        self.target_field = (
            self.target_intensity.sqrt() * torch.exp(1j * target_phase) * signal_mask
        )

        total = self.target_intensity.sum()
        lit_pixels = total**2 / (self.target_intensity**2).sum().clamp_min(
            smallest_divisor(total)
        )
        reference = (total * lit_pixels).clamp_min(smallest_divisor(total))
        self.scale: float = (
            float(INTENSITY_MSE_SCALE / reference) if scale is None else scale
        )

    def evaluate(
        self, field: torch.Tensor | None = None, target: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Calculate the loss based on the electric field.

        Args:
            field: Electric field at the image plane.
            target: Ignored. The target was fixed at construction.

        Returns:
            torch.Tensor: Cost.
        """
        difference = (field - self.target_field) * self.signal_mask
        return (difference.real**2 + difference.imag**2).sum()


class LossEfficiency(LossFunction):
    """The fraction of the total power that misses the signal region.

    Costs light thrown outside the target, so adding this term to a shape term trades
    accuracy inside the region against how much power reaches it.
    """

    def __init__(
        self,
        signal_mask: torch.Tensor,
        total_power: torch.Tensor,
        scale: float = 1e12,
    ) -> None:
        """Efficiency cost function.

        Args:
            signal_mask: Binary mask containing signal region.
            total_power: Total optical power.
            scale: Weight of this term, by default 1e12.
        """
        self.signal_mask = signal_mask
        self.total_power = total_power
        self.scale: float = scale

    def evaluate(
        self, field: torch.Tensor | None = None, target: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Calculate the loss based on the electric field.

        Args:
            field: Electric field at the image plane.
            target: Ignored. This term has no target.

        Returns:
            torch.Tensor: Cost.
        """
        intensity = torch.abs(field) ** 2
        efficiency = (intensity * self.signal_mask).sum() / self.total_power
        return (1 - efficiency)


class LossVorticity(LossFunction):
    def __init__(
        self,
        target_intensity: torch.Tensor,
        scale: float = 1e12,
    ) -> None:
        self.scale: float = scale
        self.target_intensity = target_intensity

    def evaluate(
        self, field: torch.Tensor | None = None, target: torch.Tensor | None = None
    ) -> torch.Tensor:
        intensity = field.abs() ** 2 + 1e-12
        _, grad_x = torch.gradient(field.conj())
        grad_y, _ = torch.gradient(field)
        vorticity = 1 / (2 * torch.pi) * (grad_x * grad_y).imag / intensity
        vorticity = vorticity * self.target_intensity
        return (vorticity**2).sum()




def field_intensity(field: torch.Tensor) -> torch.Tensor:
    """``|E|**2``, taken from the field's own intensity when it has one.

    A :class:`~hologradpy.optics.complex_amplitude.ComplexAmplitude` computes it as
    ``real**2 + imag**2``, which stays differentiable at a zero pixel where ``abs()``
    does not.

    Args:
        field: The image-plane field.

    Returns:
        torch.Tensor: The intensity, as a plain tensor.
    """
    intensity = getattr(field, "intensity", None)
    return field.abs() ** 2 if intensity is None else intensity


class LossAbsoluteIntensityMSE(LossFunction):
    """Squared error against a target read in absolute units.

    Neither the target nor the produced intensity is renormalized, so the cost keeps an
    absolute reference and light that leaves the simulated window is penalized.

    The target is used as given. Its absolute level is the statement of how much power
    should land in the shape, so scale it before passing it in.
    """

    def __init__(
        self,
        target_intensity: torch.Tensor,
        signal_mask: torch.Tensor,
        scale: float | None = None,
    ) -> None:
        """
        Args:
            target_intensity: Target intensity in the same units the model produces.
            signal_mask: Binary mask containing the signal region.
            scale: Weight of this term. Derived from the target when None, so the cost
                carries the same magnitude as :class:`LossIntensityMSE` at the point
                where the produced power matches the target's.
        """
        self.mse = nn.MSELoss(reduction="sum")
        self.signal_mask = signal_mask
        self.target_intensity = target_intensity * signal_mask

        total = self.target_intensity.sum()
        self.scale: float = (
            float(INTENSITY_MSE_SCALE / total.clamp_min(smallest_divisor(total)) ** 2)
            if scale is None
            else scale
        )

    def evaluate(
        self, field: torch.Tensor | None = None, target: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Calculate the loss based on the electric field.

        Args:
            field: Electric field at the image plane.
            target: Ignored. The target was fixed at construction.

        Returns:
            torch.Tensor: Cost.
        """
        intensity_out = field_intensity(field) * self.signal_mask
        return self.mse(intensity_out, self.target_intensity)
