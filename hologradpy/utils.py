from __future__ import annotations

import importlib.util
import sys
import time
from typing import Iterable, TypeVar

import numpy as np
import torch
from array_api_compat import is_torch_array
from numpy.typing import NDArray

if importlib.util.find_spec("ipywidgets") is not None:
    # The widget bar, which is the nicer one and always renders under a kernel.
    from tqdm.auto import tqdm
else:
    # Without ipywidgets, ``tqdm.auto`` under a kernel warns that IProgress was not
    # found and then renders nothing. The text bar renders everywhere.
    from tqdm import tqdm

ArrayLike = TypeVar("ArrayLike", torch.Tensor, NDArray)


def _bar_would_be_seen() -> bool:
    """Whether a progress bar has somewhere worth rendering to."""
    if "ipykernel" in sys.modules:
        return True
    try:
        return bool(sys.stderr.isatty())
    except Exception:
        # A stream that will not answer is not one to write a bar to.
        return False


def to_canvas(field: ArrayLike, resolution: tuple[int, int]) -> ArrayLike:
    """Center ``field``'s last two axes on a canvas of ``resolution``, zero-padding or
    cropping each axis as needed.

    Args:
        field: The array to reframe, with the plane on its last two axes. Any leading
            batch and wavelength axes are left unchanged.
        resolution: The ``(height, width)`` of the wanted canvas.

    Returns:
        The field on a canvas of exactly ``resolution``.
    """
    (top, bottom), (left, right) = (
        _canvas_margins(field.shape[axis], target)
        for axis, target in ((-2, resolution[0]), (-1, resolution[1]))
    )

    if is_torch_array(field):
        # A negative width crops.
        return torch.nn.functional.pad(field, (left, right, top, bottom))

    # numpy pads outwards only, so the shrinking is done using slicing.
    height, width = field.shape[-2:]
    field = field[
        ...,
        max(0, -top) : height - max(0, -bottom),
        max(0, -left) : width - max(0, -right),
    ]
    margins = [(0, 0)] * (field.ndim - 2)
    margins += [(max(0, top), max(0, bottom)), (max(0, left), max(0, right))]
    return np.pad(field, margins)


def _canvas_margins(length: int, target: int) -> tuple[int, int]:
    """How much to add either side of an axis of ``length`` to make it ``target``.
    Negative means cropping.
    """
    before = target // 2 - length // 2
    return (before, target - length - before)


def get_device(verbose: bool = False) -> torch.device:
    """Select CUDA GPU if available, otherwise use CPU.

    Args:
        verbose: If True, prints the selected device.

    Returns:
        torch.device: The selected device.
    """
    label = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(label)
    if verbose:
        print(f"Using device: {device}")
    return device


def gpu_to_numpy(array: ArrayLike) -> NDArray:
    """A numpy copy of a tensor, off the device and off the graph."""
    if not is_torch_array(array):
        return np.asarray(array)
    return array.clone().cpu().detach().numpy()


def unsqueeze_to(input: torch.Tensor, max_dim: int, dim: int = 0) -> torch.Tensor:
    while input.dim() < max_dim:
        input = input.unsqueeze(dim)
    return input


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

    kwargs.setdefault("disable", not _bar_would_be_seen())
    return tqdm(iterable, total=total, desc=description or None, **kwargs)


class ProgressBar:
    """A bar for loops that are not a plain ``for``, advanced by :meth:`update`.

    The optimizers here run their loop inside ``torchmin.Minimizer.step`` using a
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
            self._kwargs.setdefault("disable", not _bar_would_be_seen())
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
    """Time a block of work, as a context manager.

    Reads the wall clock, or the CUDA stream through events when ``use_cuda`` is set.
    The measured span is left on :attr:`elapsed_time` in seconds.
    """

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
