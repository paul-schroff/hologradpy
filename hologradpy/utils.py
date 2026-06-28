import time
from typing import TypeVar
import torch
from numpy.typing import NDArray

from array_api_compat import array_namespace, device as array_device

ArrayLike = TypeVar("ArrayLike", torch.Tensor, NDArray)


def get_device(verbose: bool = False) -> torch.device:
    """
    Select CUDA GPU if available, otherwise use CPU.

    Args:
        verbose (bool): If True, prints the selected device.

    Returns:
        torch.device: The selected device.
    """
    label = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(label)
    if verbose:
        print(f"Using device: {device}")
    return device


def gpu_to_numpy(tensor: torch.Tensor) -> NDArray:
    return tensor.clone().cpu().detach().numpy()


def unsqueeze_to(input: torch.Tensor, max_dim: int, dim: int = 0) -> torch.Tensor:
    while input.dim() < max_dim:
        input = input.unsqueeze(dim)
    return input


def pad_to_shape_2D(input: torch.Tensor, target_shape: tuple[int, int]) -> torch.Tensor:
    input_shape = input.shape[-2:]

    # Zero-pad input if target_shape is larger than its resolution.
    if any(input_shape[i] > target_shape[i] for i in range(2)):
        raise IndexError(
            "Resolution of input is larger than specified in target_shape."
        )
    elif input_shape == target_shape:
        return input
    else:
        pad_y = int((target_shape[0] - input_shape[0]) // 2)
        pad_x = int((target_shape[1] - input_shape[1]) // 2)
        pad = (pad_x, pad_x, pad_y, pad_y)
        return torch.nn.functional.pad(input, pad)


def crop_to_shape_2D(input: ArrayLike, target_shape: tuple[int, int]) -> ArrayLike:
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
    return input[..., roi[0] : roi[1], roi[2] : roi[3]]


def pad_from_roi(
    input: ArrayLike,
    roi: tuple[int, int, int, int],
    original_shape: tuple[int, int],
) -> ArrayLike:
    """Inverse of :func:`crop_to_roi`: pad a cropped image back to its original
    size, placing it at the region of interest and zero-filling the rest.

    Args:
        input (ArrayLike): The cropped 2D data, shaped (..., roi_height,
            roi_width).
        roi (tuple[int, int, int, int]): The region of interest used to crop, as
            a tuple of (top, bottom, left, right) pixel indices.
        original_shape (tuple[int, int]): The (height, width) of the original
            image before cropping.

    Returns:
        ArrayLike: The zero-padded image of shape (..., *original_shape), with
            input placed back at the region of interest.
    """
    xp = array_namespace(input)
    top, bottom, left, right = roi
    output = xp.zeros(
        (*input.shape[:-2], *original_shape),
        dtype=input.dtype,
        device=array_device(input),
    )
    output[..., top:bottom, left:right] = input
    return output


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


def roi_bounds(
    center: tuple[int, int],
    roi_size: tuple[int, int],
) -> tuple[int, int, int, int]:
    """Return the (x0, x1, y0, y1) pixel bounds of a region of interest centred
    on ``center`` (x, y) with ``roi_size`` (height, width).

    Note the (left, right, top, bottom) ordering here differs from the (top,
    bottom, left, right) convention used by :func:`crop_to_roi` and
    :func:`find_roi`; this order matches the camera window-of-interest layout
    (x, width, y, height) used by :func:`hologradpy.hardware.utils.set_camera_woi`.
    """
    x0 = center[0] - roi_size[1] // 2
    y0 = center[1] - roi_size[0] // 2
    return x0, x0 + roi_size[1], y0, y0 + roi_size[0]


class Timer:
    def __init__(self, use_cuda: bool = False, verbose: bool = False) -> None:
        """Timer class to measure elapsed time for CUDA and non-CUDA
        operations."""
        self.elapsed_time: float = None
        self.use_cuda: bool = use_cuda
        self.verbose: bool = verbose

        if use_cuda:
            self.start_event = None
            self.stop_event = None
        else:
            self.start_time: float = None
            self.stop_time: float = None

    def start(self):
        if self.use_cuda:
            self.start_event = torch.cuda.Event(enable_timing=True)
            self.stop_event = torch.cuda.Event(enable_timing=True)
            self.start_event.record()

        self.start_time = time.time()

        if self.verbose:
            date = time.strftime("%d-%m-%y__%H-%M-%S", time.localtime())
            print("Calculation start: %s\n" % date)

    def stop(self):
        if self.start_time is None:
            raise ValueError("Timer has not been started.")

        self.stop_time = time.time()

        if self.use_cuda:
            self.stop_event.record()
            torch.cuda.synchronize()
            self.elapsed_time = self.start_event.elapsed_time(self.stop_event) / 1e3
        else:
            self.elapsed_time = self.stop_time - self.start_time

        if self.verbose:
            print(
                f"Ran for {(self.elapsed_time // 60):.0f} minutes and "
                + f"{(self.elapsed_time % 60):.2f} seconds."
            )
