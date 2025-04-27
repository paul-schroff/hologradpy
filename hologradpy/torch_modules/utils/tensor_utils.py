import torch
import numpy as np
from numpy.typing import NDArray

def check_device(verbose: bool = False) -> str:
    """
    Check if GPU is available.

    :param bool verbose: Verbose output?
    :return: 'cuda' if GPU available, otherwise 'cpu'.
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if verbose:
        print(f'Using {device} device')
    return device

def gpu_to_numpy(tensor: torch.Tensor) -> NDArray:
    return tensor.clone().cpu().detach().numpy()

def unsqueeze_to(
        input: torch.Tensor,
        max_dim: int,
        dim: int = 0
    ) -> torch.Tensor:
    while input.dim() < max_dim: input = input.unsqueeze(dim)
    return input

def pad_to_shape_2D(input: torch.Tensor,
                    target_shape: tuple[int, int]
                    ) -> torch.Tensor:
    input_shape = input.shape[-2:]

    # Zero-pad input if target_shape is larger than its resolution.
    if any(el[0] > el[1] for el in
            zip(input_shape, target_shape)):
        raise IndexError('Resolution of input is larger than specified in '
                            'target_shape.')
    elif input_shape == target_shape:
        pass
    else:
        pad_h = int((target_shape[0] - input_shape[0]) // 2)
        pad_w = int((target_shape[1] - input_shape[1]) // 2)
        pad = (pad_w, pad_w, pad_h, pad_h)
        return torch.nn.functional.pad(input, pad)

def crop_to_shape_2D(input: torch.Tensor,
                     target_shape: tuple[int, int]
                     ) -> torch.Tensor:
    input_shape = input.shape[-2:]
    crop_h = (input_shape[0] - target_shape[0]) // 2
    crop_w = (input_shape[1] - target_shape[1]) // 2
    return input[...,
                 crop_h:input_shape[0] - crop_h,
                 crop_w:input_shape[1] - crop_w]