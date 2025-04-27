import torch
import torch.nn as nn

def loss_fn_fid(
        e_out: torch.Tensor,
        i_tar: torch.Tensor,
        phi_tar: torch.Tensor,
        signal: torch.Tensor
    ) -> torch.Tensor:
    """
    Phase and amplitude cost function from
    https://doi.org/10.1364/OE.25.011692.

    :param e_out: Electric field at the image plane.
    :param i_tar: Target intensity pattern.
    :param phi_tar: Target phase pattern.
    :param signal: Binary mask containing signal region.
    :return: Cost.
    """
    a_out = e_out.abs()
    phi_out = e_out.angle()
    overlap = torch.sum(
        signal * a_out * torch.sqrt(i_tar) * torch.cos(phi_out - phi_tar)
    )
    overlap = (
        overlap / (torch.sqrt(torch.sum(i_tar) * torch.sum((a_out * signal) ** 2)))
    )
    return 1e12 * (1 - overlap) ** 2


def loss_fn_amp(
        e_out: torch.Tensor,
        i_tar: torch.Tensor,
        signal: torch.Tensor
    ) -> torch.Tensor:
    """
    Amplitude-only cost function from https://doi.org/10.1364/OE.22.026548.

    :param e_out: Electric field at the image plane.
    :param i_tar: Target intensity pattern.
    :param signal: Binary mask containing signal region.
    :return: Cost.
    """
    mse = nn.MSELoss(reduction='sum')
    i_out = torch.abs(e_out) ** 2
    return 5e11 * mse(i_out * signal / torch.sum(i_out * signal), i_tar)


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