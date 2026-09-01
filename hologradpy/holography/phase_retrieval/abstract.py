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
from ...loss_functions import (
    INTENSITY_MSE_SCALE,
    LossFunction,
    LossIntensityMSE,
)
from ...optics.systems import SLMFourierLensModel, load_optical_system
from ...serialization import SaveableRecord, record_type
from ...utils import ProgressBar, Timer, gpu_to_numpy
from ...visualizer import VisualizationData
from .visualizer import (
    PhaseRetrievalVisualizationData,
    PhaseRetrievalVisualizer,
)


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

    def visualizer(self) -> PhaseRetrievalVisualizer:
        """The visualizer that draws this retrieval."""
        return PhaseRetrievalVisualizer(self)

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
    """Something that finds an SLM phase and can report what it produces.

    Holds the optical model, a run to record into, a timer, and :meth:`retrieve`, which
    wraps a search up as a saveable record.
    """
    target: torch.Tensor | None = None
    signal_region: torch.Tensor | None = None
    loss_function: Callable | None = None

    def __init__(self, slm_camera_model: SLMFourierLensModel) -> None:
        self.slm_camera_model: SLMFourierLensModel = slm_camera_model
        # The device is determined by the optical model rather than passed in.
        self.device: torch.device = slm_camera_model.device

        self.run: RetrievalRun = RetrievalRun()
        self.timer: Timer = Timer(use_cuda=self.device.type == "cuda", verbose=False)
        self.iteration: int = 0

    @property
    def loss_history(self) -> list[float]:
        """The objective values of the most recent search."""
        return self.run.loss_history


    def retrieve_phase(
        self,
        number_of_iterations: int = 10,
        *,
        run: RetrievalRun | None = None,
        verbose: bool = True,
        progress_bar: ProgressBar | None = None,
        **options: Any,
    ) -> torch.Tensor:
        """Work out the phase, put it on the model, and return it.

        Args:
            number_of_iterations: What an iteration means depends on the subclass.
            run: The run to record into. A fresh one is made when none is given.
            verbose: Show a progress bar when one is not supplied.
            progress_bar: A bar to borrow, reset and handed back untouched.
            **options: Whatever else the subclass takes.

        Returns:
            torch.Tensor: The phase the SLM is now showing.
        """
        raise NotImplementedError

    def predicted_intensity(self) -> NDArray:
        with torch.no_grad():
            return gpu_to_numpy(self.slm_camera_model().intensity)

    def predicted_field(self) -> tuple[NDArray, NDArray]:
        """The intensity and the phase the current SLM phase produces.

        Returns:
            tuple[NDArray, NDArray]: The intensity, and the phase in radians.
        """
        with torch.no_grad():
            field = self.slm_camera_model()
            return gpu_to_numpy(field.intensity), gpu_to_numpy(field.phase)

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
            metrics: How the predicted potential is compared against the target, once at
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
        initial_intensity = self.predicted_intensity()

        run = RetrievalRun(
            steps=steps,
            metrics=metrics,
            signal_region=self.signal_region,
            target=self.target,
        )
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

        target = None if self.target is None else gpu_to_numpy(self.target)
        signal_region = (
            None if self.signal_region is None else gpu_to_numpy(self.signal_region)
        )
        retrieved_intensity, retrieved_phase = self.predicted_field()
        constrained_phase = getattr(self.loss_function, "target_phase", None)
        metric_values = (
            {}
            if target is None or signal_region is None
            else evaluate_metrics(metrics, signal_region, target, retrieved_intensity)
        )

        return PhaseRetrievalData(
            timestamp=datetime.now(),
            name=name or type(self).__name__,
            phase=gpu_to_numpy(phase),
            target=target,
            signal_region=signal_region,
            loss_history=list(run.loss_history),
            metrics=metric_values,
            model_checkpoint=checkpoint_name,
            step_stride=step_stride,
            step_iterations=run.step_iterations,
            metadata=metadata or {},
            visualization_data=PhaseRetrievalVisualizationData(
                retrieved_intensity=retrieved_intensity,
                retrieved_phase=retrieved_phase,
                target_phase=(
                    None
                    if constrained_phase is None
                    else gpu_to_numpy(constrained_phase)
                ),
                metric_history={
                    name: list(values)
                    for name, values in run.metric_history.items()
                },
                initial_intensity=initial_intensity,
            ),
        )

    def save_results(self) -> None:
        pass


FIRST_ORDER_OPTIMIZERS: dict[str, type[torch.optim.Optimizer]] = {
    "adam": torch.optim.Adam,
    "adamw": torch.optim.AdamW,
    "rmsprop": torch.optim.RMSprop,
    "sgd": torch.optim.SGD,
}

DEFAULT_LEARNING_RATE = 0.03


class GradientPhaseRetriever(PhaseRetrieverBase):
    """A retriever that reaches a target by minimizing a cost with an optimizer.

    Holds an intensity target, the signal region, and an optimizer.
    """

    optimizer: torchmin.Minimizer

    PARAMETER_NAME: str
    METHOD: str = "cg"

    def __init__(
        self,
        slm_camera_model: SLMFourierLensModel,
        target: torch.Tensor | None = None,
        signal_region: torch.Tensor | None = None,
        loss_scale: float = INTENSITY_MSE_SCALE,
    ) -> None:
        super().__init__(slm_camera_model)

        self.target: torch.Tensor | None = None
        self.signal_region: torch.Tensor | None = None
        self.loss_scale: float = loss_scale

        self.loss_factory: (
            Callable[[torch.Tensor, torch.Tensor], LossFunction] | None
        ) = None
        if target is not None:
            self.set_target(target, signal_region)

    def set_target(
        self,
        target: torch.Tensor,
        signal_region: torch.Tensor | None = None,
    ) -> None:
        """Update the intensity target.

        Args:
            target: Target intensity, on the model's output grid.
            signal_region: Region the cost evaluated in.

        Raises:
            ValueError: Neither this call nor an earlier one supplied a region.
        """
        self.target = target.detach()
        if signal_region is not None:
            self.signal_region = signal_region.detach()

        if self.signal_region is None:
            raise ValueError(
                f"{type(self).__name__} minimizes a cost masked by a signal region, "
                "so it needs one. Pass it to set_target, or to the constructor."
            )

        self.loss_function = self.default_loss_function()

    def default_loss_function(self) -> LossFunction:
        """The cost for a newly set target.

        Returns:
            LossFunction: The cost for the current target and signal region.
        """
        if self.loss_factory is not None:
            return self.loss_factory(self.target, self.signal_region)
        return LossIntensityMSE(
            target_intensity=self.target,
            signal_mask=self.signal_region,
            scale=self.loss_scale,
        )

    def set_loss_factory(
        self,
        loss_factory: Callable[[torch.Tensor, torch.Tensor], LossFunction] | None,
    ) -> None:
        """Evaluates the target with a custom cost function.

        Args:
            loss_factory: Takes ``(target, signal_region)`` and returns the cost, or
                None to go back to the default intensity cost.
        """
        self.loss_factory = loss_factory
        if self.target is not None and self.signal_region is not None:
            self.loss_function = self.default_loss_function()

    def set_loss_function(self, loss_function: LossFunction) -> None:
        """Update the cost function.

        Args:
            loss_function: The cost, taking ``(predicted_field, target)``. For example
                a :class:`~hologradpy.loss_functions.LossFidelity` to constrain the
                image-plane phase as well as the intensity.
        """
        self.loss_function = loss_function
    def _parameter_name(self) -> str:
        """What this optimization varies, or a clear error message if no subclass ever
        specified it.
        """
        name = getattr(self, "PARAMETER_NAME", None)
        if name is None:
            raise NotImplementedError(
                f"{type(self).__name__} does not say what it optimizes. "
                "GradientPhaseRetriever is a base class: set PARAMETER_NAME on a "
                "subclass, or use PixelwisePhaseRetriever to vary the SLM phase pixel "
                "by pixel."
            )
        return name

    def set_optimizer(
        self,
        number_of_iterations: int,
        method: str,
        display: int = 0,
        learning_rate: float | None = None,
    ) -> None:
        """Build the optimizer this search steps.

        Args:
            number_of_iterations: Maximum optimizer iterations.
            method: A torchmin method, or a key of :data:`FIRST_ORDER_OPTIMIZERS`.
            display: torchmin's own verbosity.
            learning_rate: Step size for a first-order method. Defaults to
                :data:`DEFAULT_LEARNING_RATE`.
        """
        first_order = FIRST_ORDER_OPTIMIZERS.get(method)
        if first_order is not None:
            self.optimizer = first_order(
                [
                    parameter
                    for parameter in self.slm_camera_model.parameters()
                    if parameter.requires_grad
                ],
                lr=DEFAULT_LEARNING_RATE if learning_rate is None else learning_rate,
            )
            return

        self.optimizer = torchmin.Minimizer(
            [
                parameter
                for parameter in self.slm_camera_model.parameters()
                if parameter.requires_grad
            ],
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

    def step_first_order(self, number_of_iterations: int) -> None:
        """Drive a torch.optim optimizer, which takes one step per call.

        Args:
            number_of_iterations: How many steps to take.
        """
        for _ in range(number_of_iterations):
            loss = self.closure()
            loss.backward()
            self.optimizer.step()
            self.callback(loss)

    def closure(self) -> torch.Tensor:
        """The objective, evaluated once per optimizer call."""
        self.optimizer.zero_grad()
        field = self.slm_camera_model()
        loss = self.loss_function(field)
        self.run.record_loss(loss.item(), field)
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

    def retrieve_phase(
        self,
        number_of_iterations: int = 10,
        parameter_name: str | None = None,
        method: str | None = None,
        learning_rate: float | None = None,
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
            learning_rate: Step size, for a first-order method.
            verbose: Show a progress bar when one is not supplied.
            progress_bar: A bar to borrow. Reset here and handed back untouched.
            run: The run to record into. A new one is made when none is given.

        Returns:
            torch.Tensor: The retrieved SLM phase.
        """
        self.timer.start()
        self.run = run if run is not None else RetrievalRun()
        self.iteration = 0
        self.set_gradient_requirements(parameter_name or self._parameter_name())
        chosen = method or self.METHOD
        self.set_optimizer(
            number_of_iterations,
            method=chosen,
            display=0,
            learning_rate=learning_rate,
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
            if chosen in FIRST_ORDER_OPTIMIZERS:
                self.step_first_order(number_of_iterations)
            else:
                self.optimizer.step(self.closure)
        finally:
            self.run.progress_bar = None
            if not borrowed:
                progress_bar.close()
        self.timer.stop()

        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        return self.slm_camera_model.virtual_slm.get_phase().detach()

