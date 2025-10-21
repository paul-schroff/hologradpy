from typing import TypeVar
import torch
from numpy.typing import NDArray

from array_api_compat import array_namespace

ArrayLike = TypeVar("ArrayLike", torch.Tensor, NDArray)


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
    while input.dim() < max_dim:
        input = input.unsqueeze(dim)
    return input


def pad_to_shape_2D(
    input: torch.Tensor,
    target_shape: tuple[int, int]
    ) -> torch.Tensor:
    input_shape = input.shape[-2:]

    # Zero-pad input if target_shape is larger than its resolution.
    if any(input_shape[i] > target_shape[i] for i in range(2)):
        raise IndexError(
            'Resolution of input is larger than specified in target_shape.'
            )
    elif input_shape == target_shape:
        return input
    else:
        pad_y = int((target_shape[0] - input_shape[0]) // 2)
        pad_x = int((target_shape[1] - input_shape[1]) // 2)
        pad = (pad_x, pad_x, pad_y, pad_y)
        return torch.nn.functional.pad(input, pad)


def crop_to_shape_2D(
        input: ArrayLike,
        target_shape: tuple[int, int]
    ) -> ArrayLike:
    # TODO: This function cannot handle odd number of pixels in target_shape.
    input_shape = input.shape[-2:]
    n_crop_y = (input_shape[0] - target_shape[0]) // 2
    n_crop_x = (input_shape[1] - target_shape[1]) // 2
    return input[..., n_crop_y:-n_crop_y, n_crop_x:-n_crop_x]


def crop_to_roi(
    input: ArrayLike,
    roi: tuple[int, int, int, int],
) -> ArrayLike:
    """Crops the input image to the specified region of interest.

    Args:
        input (ArrayLike): The 2D data to be cropped.
        roi (tuple[int, int, int, int]): The region of interest to be cropped,
            as a tuple of (top, bottom, left, right) pixel indices.

    Returns:
        ArrayLike: The cropped image.
    """
    return input[..., roi[0]:roi[1], roi[2]:roi[3]]


def find_roi(
    input: ArrayLike, threshold: float = 0.5, pad: int = 10
) -> tuple[int, int, int, int]:
    """Finds the rectangular region of interest in an image including pixel 
    values larger than threshold * max(input). ROI can be padded symmetrically 
    along each axis using pad.

    Args:
        input (ArrayLike): The image data in which to find the region of 
            interest.
        threshold (float, optional): The fraction of the maximum pixel value
            at which pixels must be include in the region of interest.
            Defaults to 0.5.
        pad (int, optional): The number of pixels to pad on both axes around
            the threshold values. Defaults to 10.

    Returns:
        tuple[int, int, int, int]: The region of interest, as a tuple of (top,
            bottom, left, right) pixel indices.
    """
    xp = array_namespace(input)
    idx_y, idx_x = xp.nonzero(input > threshold * xp.max(input))

    max_idx_y = int(xp.clip(xp.max(idx_y) + pad, 0, input.shape[0]))
    min_idx_y = int(xp.clip(xp.min(idx_y) - pad, 0, input.shape[0]))
    max_idx_x = int(xp.clip(xp.max(idx_x) + pad, 0, input.shape[1]))
    min_idx_x = int(xp.clip(xp.min(idx_x) - pad, 0, input.shape[1]))

    return (min_idx_y, max_idx_y, min_idx_x, max_idx_x)