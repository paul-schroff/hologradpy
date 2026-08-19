from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Sequence

import torch
import torchmin
from numpy.typing import NDArray

from ...analysis.error_metrics import (
    DEFAULT_INTENSITY_METRICS,
    IntensityMetric,
    evaluate_metrics,
)
from .recorder import RETRIEVAL_STEPS_NAME, RetrievalRun, RetrievalStepWriter

from ...datasets import RetrievalStepStore
from ...loss_functions import INTENSITY_MSE_SCALE, LossIntensityMSE
from ...optics.systems import SLMFourierLensModel, load_optical_system
from ...serialization import SaveableRecord, record_type
from ...utils import ProgressBar, Timer, gpu_to_numpy
from ...visualizer import VisualizationData


@record_type("phase_retrieval")
@dataclass
class PhaseRetrievalData(SaveableRecord):
    timestamp: datetime
    name: str
    phase: NDArray
    target: NDArray | None = None
    signal_region: NDArray | None = None
    loss_history: list[float] = field(default_factory=list)
    metrics: dict[str, list[float]] = field(default_factory=dict)
    model_checkpoint: str | None = None
    step_stride: int | None = None
    step_iterations: list[int] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
    visualization_data: VisualizationData | None = None

    def lean(self) -> PhaseRetrievalData:
        """A copy dropping the target and the signal region to avoid storing them
        multiple times in a feedback run. Mirrors
        :meth:`~hologradpy.calibration.camera_mapping.CameraMapping.lean`.
        """
        return replace(
            self, target=None, signal_region=None, visualization_data=None
        )

    def _directory(self, directory: str | os.PathLike | None) -> Path:
        """Where the files written beside this record live.

        ``directory`` wins when given. Otherwise the record's own location is used, so a
        record and its steps that were moved together still find each other.

        Raises:
            FileNotFoundError: The record was built in memory, so it has no location of
                its own and has to be told one.
        """
        if directory is not None:
            return Path(directory)
        if self.source_directory is None:
            raise FileNotFoundError(
                "This record was not loaded from a file, so there is nowhere to look "
                "for the files written beside it. Pass the directory they went to."
            )
        return self.source_directory

    def load_step(
        self, iteration: int, directory: str | os.PathLike | None = None
    ) -> NDArray:
        """Load the learnable parameter at ``iteration``.

        A fraction of full scale, which :meth:`replay` puts back on a model. Use
        ``virtual_slm.phase_response.phase_at`` to convert to phase.

        Args:
            iteration: One of :attr:`step_iterations`.
            directory: Where the recorded steps live. Defaults to the directory this
                record was loaded from, which is where they were written.

        Raises:
            KeyError: No step was recorded at that iteration.
            FileNotFoundError: There is nowhere to look, or nothing there to read.
        """
        self._check_step(iteration)
        index = self.step_iterations.index(iteration)
        with RetrievalStepStore.open(
            self._directory(directory) / RETRIEVAL_STEPS_NAME
        ) as store:
            return store.read(index)["slm_fraction"]

    def replay(
        self,
        iteration: int,
        directory: str | os.PathLike | None = None,
        model: SLMFourierLensModel | None = None,
    ) -> tuple[NDArray, NDArray]:
        """Recreate what the model predicted at ``iteration``.

        Args:
            iteration: One of :attr:`step_iterations`.
            directory: Where the recorded steps and the checkpoint live. Defaults to the
                directory this record was loaded from.
            model: A model to replay through. Defaults to rebuilding one from
                :attr:`model_checkpoint`. The phase it holds is restored afterwards, so
                replaying does not disturb a live search.

        Returns:
            The search parameter and the intensity the model predicts from it.

        Raises:
            KeyError: No step was recorded at that iteration.
            ValueError: No model was passed and the record names no checkpoint.
            FileNotFoundError: There is nowhere to look, or nothing there to read.
        """
        self._check_step(iteration)
        if model is None:
            if self.model_checkpoint is None:
                raise ValueError(
                    "This record names no model checkpoint, so the prediction cannot "
                    "be rebuilt. Pass a model to replay through."
                )
            model = load_optical_system(
                self._directory(directory) / self.model_checkpoint
            )

        fraction = self.load_step(iteration, directory)

        # The parameter goes back exactly as it was, so a step holds it and not the
        # phase.
        restore = model.virtual_slm.levels.detach().clone()
        try:
            model.virtual_slm.levels.data = torch.as_tensor(
                fraction, dtype=restore.dtype, device=model.device
            )
            with torch.no_grad():
                return fraction, gpu_to_numpy(model().intensity)
        finally:
            model.virtual_slm.levels.data = restore

    def _check_step(self, iteration: int) -> None:
        if iteration not in self.step_iterations:
            raise KeyError(
                f"No step recorded for iteration {iteration}. Recorded iterations are "
                f"{self.step_iterations}."
            )


class PhaseRetrieverBase:
    """A search for the SLM phase that produces a target intensity."""

    optimizer: torchmin.Minimizer
    loss_function: Callable

    def __init__(
        self,
        slm_camera_model: SLMFourierLensModel,
        target: torch.Tensor | None = None,
        signal_region: torch.Tensor | None = None,
        loss_scale: float = INTENSITY_MSE_SCALE,
    ) -> None:
        self.slm_camera_model: SLMFourierLensModel = slm_camera_model
        # The device is determined by the optical model rather than passed in.
        self.device: torch.device = slm_camera_model.device

        self.target: torch.Tensor | None = None
        self.signal_region: torch.Tensor | None = None

        self.run: RetrievalRun = RetrievalRun()
        self.timer: Timer = Timer(
            use_cuda=self.device.type == "cuda", verbose=False
        )
        self.iteration: int = 0

        self.loss_scale: float = loss_scale
        if target is not None:
            self.set_target(target, signal_region)

    @property
    def loss_history(self) -> list[float]:
        """The objective values of the most recent search."""
        return self.run.loss_history

    def set_optimizer(
        self, number_of_iterations: int, method: str, display: int = 0
    ) -> None:
        """Build the optimizer this search steps.

        Args:
            number_of_iterations: Maximum optimizer iterations.
            method: Which torchmin method to use.
            display: torchmin's own verbosity.
        """
        self.optimizer = torchmin.Minimizer(
            self.slm_camera_model.parameters(),
            method=method,
            max_iter=number_of_iterations,
            disp=display,
            callback=self.callback,
            **self.optimizer_options(),
        )

    def optimizer_options(self) -> dict:
        """Extra keyword arguments for the optimizer, for a search that needs them."""
        return {}

    def callback(self, _: torch.Tensor) -> None:
        """One optimizer iteration done, which is where the bar advances and a step is
        due.

        Not :meth:`closure`, which the line search calls several times per iteration.
        """
        self.iteration += 1
        self.run.record_iteration(self.iteration, self.slm_camera_model)

    def closure(self) -> torch.Tensor:
        """The objective, evaluated once per optimizer call."""
        self.optimizer.zero_grad()
        loss = self.loss_function(self.slm_camera_model())
        self.run.record_loss(loss.item())
        return loss

    def set_gradient_requirements(
        self, parameter_name: str = "virtual_slm.levels"
    ) -> None:
        named_parameters = dict(self.slm_camera_model.named_parameters())

        if parameter_name not in named_parameters:
            _ = self.slm_camera_model()
            named_parameters = dict(self.slm_camera_model.named_parameters())

        if parameter_name not in named_parameters:
            available_parameters = ", ".join(named_parameters.keys())
            raise ValueError(
                f"Parameter '{parameter_name}' not found in "
                f"slm_camera_model.named_parameters(). "
                f"Available parameters: {available_parameters}"
            )

        for name, parameter in named_parameters.items():
            parameter.requires_grad = name == parameter_name

    def set_target(
        self,
        target: torch.Tensor,
        signal_region: torch.Tensor | None = None,
    ) -> None:
        """Update the intensity target.

        Args:
            target: Target intensity, on the model's output grid.
            signal_region: Region the target is scored over. Defaults to the one already
                held, so a retarget need only pass the target.
        """
        self.target = target.detach()
        if signal_region is not None:
            self.signal_region = signal_region.detach()

        if self.signal_region is None:
            raise ValueError(
                f"{type(self).__name__} has no signal region to score the target over. "
                "Pass one to set_target, or to the constructor."
            )

        self.loss_function = LossIntensityMSE(
            target_intensity=self.target,
            signal_mask=self.signal_region,
            scale=self.loss_scale,
        )

    def set_loss_function(self, loss_function: Callable) -> None:
        pass

    # What this search optimizes, and with which method. Named per subclass, so
    # retrieve_phase below is the one implementation.
    PARAMETER_NAME: str = "virtual_slm.levels"
    METHOD: str = "cg"

    def retrieve_phase(
        self,
        number_of_iterations: int = 10,
        parameter_name: str | None = None,
        method: str | None = None,
        verbose: bool = True,
        progress_bar: ProgressBar | None = None,
        run: RetrievalRun | None = None,
    ) -> torch.Tensor:
        """Run the retrieval and return the SLM phase it arrived at.

        Args:
            number_of_iterations: Maximum optimizer iterations.
            parameter_name: What to optimize. Defaults to this retriever's
                :attr:`PARAMETER_NAME`.
            method: Optimizer method. Defaults to this retriever's :attr:`METHOD`.
            verbose: Show a progress bar when one is not supplied.
            progress_bar: A bar to borrow. Reset here and handed back untouched, which
                is what lets a feedback loop reuse a single bar across many retrievals.
            run: The run to record into. A fresh one is made when none is given.

        Returns:
            torch.Tensor: The retrieved SLM phase.
        """
        self.timer.start()
        self.run = run if run is not None else RetrievalRun()
        self.iteration = 0
        self.set_gradient_requirements(parameter_name or self.PARAMETER_NAME)
        self.set_optimizer(
            number_of_iterations, method=method or self.METHOD, display=0
        )

        borrowed = progress_bar is not None
        if borrowed:
            progress_bar.reset(total=number_of_iterations)
        else:
            progress_bar = ProgressBar(
                total=number_of_iterations,
                description="Phase retrieval",
                verbose=verbose,
            ).__enter__()

        self.run.progress_bar = progress_bar
        try:
            self.optimizer.step(self.closure)
        finally:
            self.run.progress_bar = None
            if not borrowed:
                progress_bar.close()
        self.timer.stop()

        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        return self.slm_camera_model.virtual_slm.get_phase().detach()

    def predicted_intensity(self) -> NDArray:
        with torch.no_grad():
            return gpu_to_numpy(self.slm_camera_model().intensity)

    def retrieve(
        self,
        number_of_iterations: int = 10,
        *,
        name: str = "",
        metrics: Sequence[IntensityMetric] = DEFAULT_INTENSITY_METRICS,
        step_stride: int | None = None,
        step_directory: str | os.PathLike | None = None,
        model_checkpoint: str | None = None,
        metadata: dict | None = None,
        **options: Any,
    ) -> PhaseRetrievalData:
        """Run the search and return it as a saveable record.

        Args:
            number_of_iterations: Passed to :meth:`retrieve_phase`.
            name: Label stored on the record.
            metrics: How the predicted potential is scored against the target, once at
                the end. Defaults to rmse and psnr.
            step_stride: Record the retrieval's parameter every nth optimizer
                iteration. Defaults to None, which records nothing.
            step_directory: Where to save the recorded steps and model checkpoint. 
                Required when a stride is given.
            model_checkpoint: Filename for that checkpoint, written into
                ``step_directory``. Defaults to ``model_checkpoint.pt``.
            metadata: Anything else worth keeping with the result.
            **options: Passed to :meth:`retrieve_phase`, such as ``method``.

        Returns:
            PhaseRetrievalData: The hologram, the convergence, and metrics.
        """
        steps = (
            None
            if step_stride is None
            else RetrievalStepWriter(
                step_stride, step_directory, self.slm_camera_model
            )
        )
        run = RetrievalRun(steps=steps)
        try:
            phase = self.retrieve_phase(
                number_of_iterations, run=run, **options
            )
        finally:
            run.close()

        checkpoint_name = (
            None
            if steps is None
            else steps.save_checkpoint(self.slm_camera_model, model_checkpoint)
        )

        target = gpu_to_numpy(self.target)
        signal_region = gpu_to_numpy(self.signal_region)
        scores = evaluate_metrics(
            metrics, signal_region, target, self.predicted_intensity()
        )

        return PhaseRetrievalData(
            timestamp=datetime.now(),
            name=name or type(self).__name__,
            phase=gpu_to_numpy(phase),
            target=target,
            signal_region=signal_region,
            loss_history=list(run.loss_history),
            metrics=scores,
            model_checkpoint=checkpoint_name,
            step_stride=step_stride,
            step_iterations=run.step_iterations,
            metadata=metadata or {},
        )

    def save_results(self) -> None:
        pass
