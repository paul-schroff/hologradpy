from __future__ import annotations

import time
from typing import Iterable, TypeVar
import torch
from numpy.typing import NDArray
from tqdm.auto import tqdm

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


def progress(
    iterable: Iterable,
    *,
    total: int | None = None,
    description: str = "",
    verbose: bool = True,
    **kwargs,
) -> Iterable:
    """Wrap ``iterable`` in a progress bar, or hand it back untouched. With 
    ``verbose=False`` the iterable is returned as it came.

    Args:
        iterable: What to iterate over.
        total: Number of steps, for an iterable without a length.
        description: Label shown to the left of the progress bar.
        verbose: Show the bar. False returns ``iterable`` unchanged.
        **kwargs: Passed to ``tqdm``.

    Returns:
        Iterable: The wrapped iterable, or the original one.
    """
    if not verbose:
        return iterable

    kwargs.setdefault("disable", None)
    return tqdm(iterable, total=total, desc=description or None, **kwargs)


class ProgressBar:
    """A bar for loops that are not a plain ``for``, advanced by :meth:`update`.

    The optimisers here run their loop inside ``torchmin.Minimizer.step`` using a
    callback, so there is no iterable to wrap. Use as a context manager so the bar
    closes even when the loop raises::

        with ProgressBar(total=iterations, description="Conjugate gradient") as bar:
            ...
            bar.update(loss=value)

    With ``verbose=False`` every method is a no-op.
    """

    def __init__(
        self,
        total: int | None = None,
        description: str = "",
        verbose: bool = True,
        **kwargs,
    ) -> None:
        self.total = total
        self.description = description
        self.verbose = verbose
        self._kwargs = kwargs
        self._bar = None

    def __enter__(self) -> ProgressBar:
        if self.verbose:
            # See progress() for why disable=None.
            self._kwargs.setdefault("disable", None)
            self._bar = tqdm(
                total=self.total, desc=self.description or None, **self._kwargs
            )
        return self

    def __exit__(self, *exception) -> None:
        self.close()

    def update(self, steps: int = 1, **postfix) -> None:
        """Advance by ``steps``, showing ``postfix`` after the bar. Postfix values are 
        formatted to four significant figures when they are numbers.
        """
        if self._bar is None:
            return
        if postfix:
            self._bar.set_postfix(
                {
                    key: f"{value:.4g}" if isinstance(value, (int, float)) else value
                    for key, value in postfix.items()
                },
                refresh=False,
            )
        self._bar.update(steps)

    def reset(self, total: int | None = None) -> None:
        """Reset the bar back to zero and change the number of steps."""
        if self._bar is None:
            return
        self._bar.set_postfix({}, refresh=False)
        self._bar.reset(total=total if total is not None else self.total)

    def close(self) -> None:
        if self._bar is not None:
            self._bar.close()
            self._bar = None


class Timer:
    def __init__(
        self,
        label: str = "Calculation",
        use_cuda: bool = False,
        verbose: bool = False,
    ) -> None:
        """
        Args:
            label: Name used in the printed messages.
            use_cuda: Time the CUDA stream with events rather than the wall clock. Only
                measures device work, so it under-reports anything that waits on
                hardware or on the CPU.
            verbose: Print when the timing starts and what it measured.
        """
        self.label: str = label
        self.use_cuda: bool = use_cuda
        self.verbose: bool = verbose

        self.elapsed_time: float | None = None
        self.start_time: float | None = None
        self.stop_time: float | None = None
        self.start_event = None
        self.stop_event = None

    def start(self) -> Timer:
        if self.use_cuda:
            self.start_event = torch.cuda.Event(enable_timing=True)
            self.stop_event = torch.cuda.Event(enable_timing=True)
            self.start_event.record()

        self.elapsed_time = None
        self.start_time = time.time()

        if self.verbose:
            date = time.strftime("%d-%m-%y__%H-%M-%S", time.localtime())
            print(f"{self.label} start: {date}\n")
        return self

    def stop(self) -> float:
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
            print(f"{self.label} took {self.formatted()}.")
        return self.elapsed_time

    def formatted(self) -> str:
        """The elapsed time, in whichever units read most naturally."""
        if self.elapsed_time is None:
            return "not measured"
        if self.elapsed_time < 1.0:
            return f"{self.elapsed_time * 1e3:.0f} ms"
        if self.elapsed_time < 60.0:
            return f"{self.elapsed_time:.1f} s"
        return f"{self.elapsed_time // 60:.0f} min {self.elapsed_time % 60:.1f} s"

    def __enter__(self) -> Timer:
        return self.start()

    def __exit__(self, *exception) -> None:
        # Stops on the way out of a failed block too, so a partial run still reports.
        self.stop()
