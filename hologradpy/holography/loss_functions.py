import torch
import torch.nn as nn
from ..optics.complex_amplitude import ComplexAmplitude


class LossFunctionBase:
    def __init__(self) -> None:
        """Base class for loss functions."""
        pass

    def loss(self, electric_field: torch.Tensor) -> torch.Tensor:
        """Calculate the loss based on the electric field.

        Args:
            electric_field : torch.Tensor
                Electric field at the image plane.

        Returns:
            torch.Tensor
                Cost.
        """
        raise NotImplementedError("Loss function not implemented.")


class LossIntensityMSE(LossFunctionBase):
    def __init__(
        self,
        target_intensity: torch.Tensor,
        signal_mask: torch.Tensor,
        steepness: float = 1e12,
    ) -> None:
        """Amplitude-only cost function from
        https://doi.org/10.1364/OE.22.026548.

        Args:
            target_intensity : torch.Tensor
                Target intensity pattern.
            signal_mask : torch.Tensor
                Binary mask containing signal region.
            steepness : float, optional
                Steepness of the cost function, by default 1e12.
        """
        self.mse = nn.MSELoss(reduction="sum")
        self.signal_mask = signal_mask
        self.steepness = steepness

        target_intensity = target_intensity * signal_mask
        target_intensity /= target_intensity.sum()

        self.target_intensity = target_intensity

    def loss(self, complex_amplitude: torch.Tensor) -> torch.Tensor:
        """Calculate the loss based on the electric field.

        Args:
            complex_amplitude : torch.Tensor
                Complex amplitude at the image plane.

        Returns:
            torch.Tensor
                Cost.
        """
        intensity_out = complex_amplitude.abs() ** 2 * self.signal_mask
        intensity_out = intensity_out / intensity_out.sum()
        return self.steepness * self.mse(intensity_out, self.target_intensity)


class LossFidelity(LossFunctionBase):
    def __init__(
        self,
        target_intensity: torch.Tensor,
        target_phase: torch.Tensor,
        signal_mask: torch.Tensor,
        steepness: float = 1e12,
    ) -> None:
        """Phase and amplitude cost function from
        https://doi.org/10.1364/OE.25.011692.

        Args:
            target_intensity : torch.Tensor
                Target intensity pattern.
            target_phase : torch.Tensor
                Target phase pattern.
            signal_mask : torch.Tensor
                Binary mask containing signal region.
            steepness : float, optional
                Steepness of the cost function, by default 1e12.
        """
        self.steepness = steepness
        self.signal_mask = signal_mask
        self.target_intensity = target_intensity * signal_mask
        self.target_amplitude = self.target_intensity.sqrt()
        self.target_phase = target_phase * signal_mask

    def loss(self, electric_field: torch.Tensor) -> torch.Tensor:
        """Calculate the loss based on the electric field.

        Args:
            electric_field : torch.Tensor
                Electric field at the image plane.

        Returns:
            torch.Tensor
                Cost.
        """
        amplitude_out = electric_field.abs()
        phase_out = electric_field.angle()

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

        return self.steepness * (1 - overlap) ** 2


class LossEfficiency(LossFunctionBase):
    def __init__(
        self,
        signal_mask: torch.Tensor,
        total_power: torch.Tensor,
        steepness: float = 1e12,
    ) -> None:
        """Efficiency cost function.

        Args:
            signal_mask : torch.Tensor
                Binary mask containing signal region.
            total_power : float
                Total optical power.
            steepness : float, optional
                Steepness of the cost function, by default 1e12.
        """
        self.signal_mask = signal_mask
        self.total_power = total_power
        self.steepness = steepness

    def loss(self, electric_field: torch.Tensor) -> torch.Tensor:
        """Calculate the loss based on the electric field.

        Args:
            electric_field : torch.Tensor
                Electric field at the image plane.

        Returns:
            torch.Tensor
                Cost.
        """
        intensity = torch.abs(electric_field) ** 2
        efficiency = (intensity * self.signal_mask).sum() / self.total_power
        return self.steepness * (1 - efficiency)


class LossVorticity(LossFunctionBase):
    def __init__(
        self,
        target_intensity: torch.Tensor,
        steepness: float = 1e12,
    ):
        self.steepness = steepness
        self.target_intensity = target_intensity

    def loss(self, electric_field: torch.Tensor):
        intensity = electric_field.abs() ** 2 + 1e-12
        _, grad_x = torch.gradient(electric_field.conj())
        grad_y, _ = torch.gradient(electric_field)
        vorticity = 1 / (2 * torch.pi) * (grad_x * grad_y).imag / intensity
        vorticity = vorticity * self.target_intensity
        return self.steepness * (vorticity**2).sum()


# TODO: Tidy this up
def rms(
    signal: torch.Tensor, i_target: torch.Tensor, i_out: torch.Tensor, frac: float
) -> torch.Tensor:
    """
    Calculate normalised root-mean-squared error between two images inside a
    region of interest. Only pixels which are brighter than
    ``frac * np.max(i_target_norm)`` are taken into account, where
    ``i_target_norm`` is the normalised target intensity pattern.

    :param signal: Binary mask containing region of interest (signal region).
    :param i_target: Target intensity pattern.
    :param i_out: Intensity pattern of light potential.
    :param frac: Threshold as explained above.
    :return: Normalised rms error.
    """
    # Find non-zero indices of measure region.
    mr_idx = (i_target * signal) > ((1 - frac) * torch.max(i_target * signal))

    # Normalise intensity patterns
    i_target_w_norm = i_target[mr_idx] / torch.sum(i_target[mr_idx])
    i_out_w_norm = i_out[mr_idx] / torch.sum(i_out[mr_idx])

    # Calculate normalised root-mean-squared error
    n = ((i_out_w_norm - i_target_w_norm) / i_target_w_norm) ** 2
    n = torch.sqrt(torch.mean(n))
    return n
