import torch
import torch.nn as nn

class LossFunctionBase:
    def __init__(self) -> None:
        """Base class for loss functions."""
        pass

    def loss(self, electric_field: torch.Tensor) -> torch.Tensor:
        """ Calculate the loss based on the electric field.

        Parameters
        ----------
        electric_field : torch.Tensor
            Electric field at the image plane.
        
        returns
        -------
        torch.Tensor
            Cost.
        """
        raise NotImplementedError("Loss function not implemented.")


class LossFunctionIntensityMSE(LossFunctionBase):
    def __init__(
            self,
            target_intensity: torch.Tensor,
            signal_mask: torch.Tensor,
            steepness: float = 1e12
        ) -> None:
        """ Amplitude-only cost function from 
        https://doi.org/10.1364/OE.22.026548.

        Parameters
        ----------
        target_intensity : torch.Tensor
            Target intensity pattern.
        signal_mask : torch.Tensor
            Binary mask containing signal region.
        steepness : float, optional
            Steepness of the cost function, by default 1e12.
        """
        self.mse = nn.MSELoss(reduction='sum')
        self.signal_mask = signal_mask
        self.steepness = steepness
        
        target_intensity = target_intensity * signal_mask
        target_intensity /= target_intensity.sum()

        self.target_intensity = target_intensity
    
    def loss(self, electric_field: torch.Tensor) -> torch.Tensor:
        """ Calculate the loss based on the electric field.

        Parameters
        ----------
        electric_field : torch.Tensor
            Electric field at the image plane.
        
        returns
        -------
        torch.Tensor
            Cost.
        """
        intensity_out = torch.abs(electric_field) ** 2 * self.signal_mask
        intensity_out = intensity_out / intensity_out.sum()
        return self.steepness * self.mse(intensity_out, self.target_intensity)


class LossFunctionFidelity(LossFunctionBase):
    def __init__(
            self,
            target_intensity: torch.Tensor,
            target_phase: torch.Tensor,
            signal_mask: torch.Tensor,
            steepness: float = 1e12
        ) -> None:
        """ Phase and amplitude cost function from 
        https://doi.org/10.1364/OE.25.011692.

        Parameters
        ----------
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
        """ Calculate the loss based on the electric field.

        Parameters
        ----------
        electric_field : torch.Tensor
            Electric field at the image plane.
        
        returns
        -------
        torch.Tensor
            Cost.
        """
        amplitude_out = electric_field.abs()
        phase_out = electric_field.angle()

        overlap = (
            self.signal_mask * amplitude_out * self.target_amplitude *
            (phase_out - self.target_phase).cos()
        ).sum()
        overlap /= (
            self.target_intensity.sum() * 
            (amplitude_out * self.signal_mask) ** 2
        ).sqrt().sum()

        return self.steepness * (1 - overlap) ** 2

def rms(
    signal: torch.Tensor,
    i_target: torch.Tensor,
    i_out: torch.Tensor,
    frac: float
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

def eff(signal: torch.Tensor, i_out: torch.Tensor) -> torch.Tensor:
    """
    Calculates the predicted efficiency of a light potential by dividing the 
    pixel sum in the signal region by the pixel sum in the entire pattern.

    :param signal: Binary mask containing the signal region.
    :param i_out: Intensity pattern of the light potential.
    :return: Predicted efficiency.
    """
    return torch.sum(signal * i_out) / torch.sum(i_out)