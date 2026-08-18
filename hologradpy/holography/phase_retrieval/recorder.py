from __future__ import annotations

import os
from pathlib import Path

from ...datasets import RetrievalStepStore
from ...optics.systems import SLMFourierLensModel
from ...utils import ProgressBar, gpu_to_numpy

# The model the steps were taken from. Always stored with them, so the simulated
# output can be reconstructed.
MODEL_CHECKPOINT_NAME = "model_checkpoint.pt"

# The steps themselves, streamed into one file as the retrieval proceeds.
RETRIEVAL_STEPS_NAME = "retrieval_steps.asdf"


class RetrievalStepWriter:
    """Saves the retrieval's own parameter every ``stride`` iterations.

    Args:
        stride: Record every nth optimizer iteration.
        directory: Where to save the file and the checkpoint. Created if it does not 
            exist.
        model: The model whose parameter is recorded, checked here so a retrieval that
            could not be replayed fails before it starts rather than after.

    Raises:
        ValueError: The stride is below one, no directory was given, or the model draws
            randomness on every forward pass (output is not deterministic and cannot be
            reconstructed).
    """

    def __init__(
        self,
        stride: int,
        directory: str | os.PathLike | None,
        model: SLMFourierLensModel,
    ) -> None:
        if stride < 1:
            raise ValueError(f"step_stride must be at least 1, got {stride}.")
        if directory is None:
            raise ValueError(
                "step_stride was set but step_directory was not. The recorded "
                "steps and the model checkpoint need somewhere to go."
            )

        stochastic = model.stochastic_modules()
        if stochastic:
            raise ValueError(
                "A step records the parameter alone and rebuilds the predicted image "
                "from it, which a model that draws randomness per forward pass cannot "
                f"support. {type(model).__name__} carries {', '.join(stochastic)}. "
                "Those belong in a simulated bench rather than in the model a "
                "retrieval optimizes."
            )

        self.stride: int = int(stride)
        self.directory: Path = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)

        self.iterations: list[int] = []
        self.store: RetrievalStepStore = RetrievalStepStore.capture(
            self.directory / RETRIEVAL_STEPS_NAME,
            frame_shape=tuple(model.virtual_slm.levels.shape),
        )

    def record(self, iteration: int, model: SLMFourierLensModel) -> None:
        """Record the model's parameter if this iteration is one of the nth."""
        if iteration % self.stride:
            return
        self.store.append(gpu_to_numpy(model.virtual_slm.levels))
        self.iterations.append(iteration)

    def close(self) -> None:
        """Finish the file, which the retrieval does when it ends."""
        self.store.close()

    def save_checkpoint(
        self, model: SLMFourierLensModel, filename: str | None = None
    ) -> str:
        """Write the model beside the steps, and return the filename to record."""
        filename = filename or MODEL_CHECKPOINT_NAME
        model.save(str(self.directory / filename))
        return filename


class RetrievalRun:
    """The accumulated output of a phase retrieval run.

    Args:
        steps: Where the per-iteration parameter goes, or None to record none.
        progress_bar: A bar to advance, or None. Attached by whoever owns the bar, since
            it may be borrowed for several retrievals in a row.
    """

    def __init__(
        self,
        steps: RetrievalStepWriter | None = None,
        progress_bar: ProgressBar | None = None,
    ) -> None:
        self.steps: RetrievalStepWriter | None = steps
        self.progress_bar: ProgressBar | None = progress_bar
        self.loss_history: list[float] = []

    def record_loss(self, loss: float) -> None:
        """Note one objective evaluation."""
        self.loss_history.append(float(loss))

    def record_iteration(self, iteration: int, model: SLMFourierLensModel) -> None:
        """Note one completed optimizer iteration."""
        if self.steps is not None:
            self.steps.record(iteration, model)
        if self.progress_bar is not None:
            self.progress_bar.update(loss=self.latest_loss)

    @property
    def latest_loss(self) -> float:
        """The most recent objective value, or zero before the first evaluation."""
        return self.loss_history[-1] if self.loss_history else 0.0

    @property
    def step_iterations(self) -> list[int]:
        """The iterations a step was recorded at, empty when none were."""
        return list(self.steps.iterations) if self.steps else []

    def close(self) -> None:
        """Finish anything this retrieval opened."""
        if self.steps is not None:
            self.steps.close()
