from __future__ import annotations

import operator
from functools import reduce

import torch
import torch.nn as nn


# Floor for the divisors below
TINY = 1e-20


def normalize_to_unit_sum(images: torch.Tensor) -> torch.Tensor:
    """Scale every image in a batch to unit total intensity.
    """
    return images / images.sum(dim=(-2, -1), keepdim=True).clamp_min(TINY)


def normalize_single_to_unit_sum(image: torch.Tensor) -> torch.Tensor:
    """Scale one image to unit total intensity, summing to a scalar.
    """
    return image / image.sum().clamp_min(TINY)


def masked_intensity_mse(
    field: torch.Tensor,
    target_intensity: torch.Tensor,
    mask: torch.Tensor,
    region_pixel_count: float,
) -> torch.Tensor:
    """Squared error between a predicted and a measured intensity over a region.

    Both are masked and normalised to unit sum first, so the comparison is of
    intensity distribution rather than absolute counts.

    Args:
        field: Predicted complex field, ``(batch, height, width)``.
        target_intensity: Measured intensity over the same region.
        mask: Region of interest, broadcast against both.
        region_pixel_count: How many pixels the mask selects.

    Returns:
        The mismatch, averaged over the batch.
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
    """A term scoring a predicted field, addable to other terms.

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
    """How far the predicted intensity is from the measured frame, over a region.

    The data term every speckle parameterisation is fitted against. See
    :func:`masked_intensity_mse` for why the result is scaled by the region size.

    The same metric as :class:`LossIntensityMSE`, which differs only in where its target
    comes from: one fixed image held from construction rather than a measured frame per
    batch.

    Args:
        mask: Region of interest, already cropped to its bounding box and in the
            model's dtype and device.
        scale: Weight of this term.
    """

    name = "intensity mse"

    def __init__(self, mask: torch.Tensor, scale: float = 1.0) -> None:
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
    """Penalise structure in the SLM-plane phase.

    An SLM-plane field stored one value per pixel can fit the speckle with a rough,
    unphysical solution, so it needs a prior. A parameterisation that is band limited by
    construction, such as a compact point spread function kernel, does not, and simply
    goes without one.

    Ignores ``field`` and ``target``: it reads the field module it was built on, which
    is what makes it addable to a data term.

    Args:
        slm_field: The field module being fitted, carrying ``phase``.
        scale: Weight of this penalty, relative to the data term.
    """

    name = "phase smoothness"

    def __init__(self, slm_field, scale: float = 1e-3) -> None:
        self.slm_field = slm_field
        self.scale: float = scale

    def evaluate(
        self, field: torch.Tensor | None = None, target: torch.Tensor | None = None
    ) -> torch.Tensor:
        return gradient_loss(self.slm_field.phase)


class AmplitudeSmoothness(LossFunction):
    """Penalise structure in the SLM-plane amplitude.

    The amplitude counterpart of :class:`PhaseSmoothness`, and the other half of the
    prior a per-pixel field needs. Weighted separately, since the two quantities are
    independent.

    Args:
        slm_field: The field module being fitted, carrying ``amplitude``.
        scale: Weight of this penalty, relative to the data term.
    """

    name = "amplitude smoothness"

    def __init__(self, slm_field, scale: float = 1e-3) -> None:
        self.slm_field = slm_field
        self.scale: float = scale

    def evaluate(
        self, field: torch.Tensor | None = None, target: torch.Tensor | None = None
    ) -> torch.Tensor:
        # Scale-free, so the weight means the same thing at any beam brightness.
        unit_amplitude = self.slm_field.amplitude.abs()
        unit_amplitude = unit_amplitude / unit_amplitude.mean().clamp_min(TINY)
        return gradient_loss(unit_amplitude)


def forward_difference(input: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    grad_img_x = torch.diff(input, dim=-2)
    grad_img_y = torch.diff(input, dim=-1)
    return grad_img_x, grad_img_y


def gradient_loss(input: torch.Tensor, aperture_relative: bool = True) -> torch.Tensor:
    """Mean squared finite difference, measured across the aperture by default.

    Args:
        input: A ``(..., height, width)`` field to penalise.
        aperture_relative: Scale each difference by the pixel count along its
            axis. Pass False for the raw per-pixel differences.
    """
    gradient_x, gradient_y = forward_difference(input)
    if aperture_relative:
        gradient_x = gradient_x * input.shape[-2]
        gradient_y = gradient_y * input.shape[-1]
    return torch.mean(gradient_x**2) + torch.mean(gradient_y**2)


def mean_curvature(input: torch.Tensor, pixel_pitch: float = 1.0) -> torch.Tensor:
    """Calculate the mean curvature of a 2D image using finite differences."""
    gradient_x, gradient_y = torch.gradient(
        input, dim=[-2, -1], spacing=pixel_pitch, edge_order=2
    )
    curvature_xx, curvature_xy = torch.gradient(
        gradient_x, dim=[-2, -1], spacing=1, edge_order=2
    )

    # Note: curvature_yx is not needed in this calculation
    curvature_yx, curvature_yy = torch.gradient(
        gradient_y, dim=[-2, -1], spacing=1, edge_order=2
    )

    mean_curvature = (
        0.5
        * (
            (1 + gradient_x**2) * curvature_yy
            + (1 + gradient_y**2) * curvature_xx
            - 2 * gradient_x * gradient_y * curvature_xy
        )
        / ((1 + gradient_x**2 + gradient_y**2) ** (3 / 2))
    )
    return mean_curvature


class LossIntensityMSE(LossFunction):
    def __init__(
        self,
        target_intensity: torch.Tensor,
        signal_mask: torch.Tensor,
        scale: float = 1e12,
    ) -> None:
        """Amplitude-only cost function from https://doi.org/10.1364/OE.22.026548.

        Args:
            target_intensity : torch.Tensor
                Target intensity pattern.
            signal_mask : torch.Tensor
                Binary mask containing signal region.
            scale : float, optional
                Weight of this term, by default 1e12.
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
            field : torch.Tensor
                Complex amplitude at the image plane.
            target : torch.Tensor, optional
                Ignored. The target was fixed at construction.

        Returns:
            torch.Tensor
                Cost.
        """
        intensity_out = field.abs() ** 2 * self.signal_mask
        intensity_out = normalize_single_to_unit_sum(intensity_out)
        return self.mse(intensity_out, self.target_intensity)


class LossFidelity(LossFunction):
    def __init__(
        self,
        target_intensity: torch.Tensor,
        target_phase: torch.Tensor,
        signal_mask: torch.Tensor,
        scale: float = 1e12,
    ) -> None:
        """Phase and amplitude cost function from https://doi.org/10.1364/OE.25.011692.

        Args:
            target_intensity : torch.Tensor
                Target intensity pattern.
            target_phase : torch.Tensor
                Target phase pattern.
            signal_mask : torch.Tensor
                Binary mask containing signal region.
            scale : float, optional
                Weight of this term, by default 1e12.
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
            field : torch.Tensor
                Electric field at the image plane.
            target : torch.Tensor, optional
                Ignored. The target was fixed at construction.

        Returns:
            torch.Tensor
                Cost.
        """
        amplitude_out = field.abs()
        phase_out = field.angle()

        overlap = (
            self.signal_mask
            * amplitude_out
            * self.target_amplitude
            * (phase_out - self.target_phase).cos()
        ).sum()
        overlap /= (
            (self.target_intensity.sum() * (amplitude_out * self.signal_mask) ** 2)
            .sqrt()
            .sum()
        )

        return (1 - overlap) ** 2


class LossEfficiency(LossFunction):
    def __init__(
        self,
        signal_mask: torch.Tensor,
        total_power: torch.Tensor,
        scale: float = 1e12,
    ) -> None:
        """Efficiency cost function.

        Args:
            signal_mask : torch.Tensor
                Binary mask containing signal region.
            total_power : float
                Total optical power.
            scale : float, optional
                Weight of this term, by default 1e12.
        """
        self.signal_mask = signal_mask
        self.total_power = total_power
        self.scale: float = scale

    def evaluate(
        self, field: torch.Tensor | None = None, target: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Calculate the loss based on the electric field.

        Args:
            field : torch.Tensor
                Electric field at the image plane.
            target : torch.Tensor, optional
                Ignored. This term has no target.

        Returns:
            torch.Tensor
                Cost.
        """
        intensity = torch.abs(field) ** 2
        efficiency = (intensity * self.signal_mask).sum() / self.total_power
        return (1 - efficiency)


class LossVorticity(LossFunction):
    def __init__(
        self,
        target_intensity: torch.Tensor,
        scale: float = 1e12,
    ):
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


