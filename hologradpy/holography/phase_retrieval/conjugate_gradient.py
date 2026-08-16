from __future__ import annotations

import torch
import torchmin

from .abstract import PhaseRetrieverBase
from .recorder import RetrievalRun

from ...loss_functions import INTENSITY_MSE_SCALE

from ...utils import ProgressBar, Timer

from ...optics.systems import SLMFourierLensModel


# TODO: Add convergence error metrics
class CGPhaseRetriever(PhaseRetrieverBase):
    def __init__(
        self,
        slm_camera_model: SLMFourierLensModel,
        target: torch.Tensor | None = None,
        signal_region: torch.Tensor | None = None,
        init_slm_phase: torch.Tensor | None = None,
        loss_scale: float = INTENSITY_MSE_SCALE,
    ) -> None:
        super().__init__(slm_camera_model, target, signal_region, loss_scale)

        if init_slm_phase is not None:
            self.slm_camera_model.virtual_slm.set_phase(init_slm_phase.detach())

        self.timer = Timer(use_cuda=self.device.type == "cuda", verbose=False)

        self.iteration: int = 0

    def set_optimizer(
        self, number_of_iterations: int, method: str = "cg", display: int = 0
    ) -> None:
        self.optimizer = torchmin.Minimizer(
            self.slm_camera_model.parameters(),
            method=method,
            max_iter=number_of_iterations,
            disp=display,
            callback=self.callback,
        )

    def callback(self, _):
        self.iteration += 1
        self.run.record_iteration(self.iteration, self.slm_camera_model)

    def closure(self):
        self.optimizer.zero_grad()
        complex_amplitude = self.slm_camera_model()
        loss = self.loss_function(complex_amplitude)
        self.run.record_loss(loss.item())
        return loss

    def retrieve_phase(
        self: CGPhaseRetriever,
        number_of_iterations: int = 10,
        parameter_name: str = "virtual_slm.levels",
        method: str = "cg",
        verbose: bool = True,
        progress_bar: ProgressBar | None = None,
        run: RetrievalRun | None = None,
    ) -> torch.Tensor:
        """Retrieve the phase that produces the current target.

        Args:
            number_of_iterations: Maximum optimiser iterations.
            parameter_name: Model parameter to optimise.
            method: torchmin method, such as "cg" or "l-bfgs".
            verbose: Show a progress bar.
            progress_bar: An existing bar to drive instead of opening one. Camera
                feedback passes the same bar to every search so it stays on one line,
                and owns closing it.
            run: Where this retrieval's convergence and steps are collected.
                Defaults to a fresh one, which is what a caller that only wants the
                hologram gets.

        Returns:
            torch.Tensor: The retrieved SLM phase.
        """
        self.timer.start()
        self.run = run if run is not None else RetrievalRun()
        self.iteration = 0
        self.set_gradient_requirements(parameter_name)
        self.set_optimizer(number_of_iterations, method=method, display=0)

        resuse_previous_bar = progress_bar is not None
        if resuse_previous_bar:
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
            if not resuse_previous_bar:
                progress_bar.close()
        self.timer.stop()

        if self.slm_camera_model.device.type == "cuda":
            torch.cuda.empty_cache()
        return self.slm_camera_model.virtual_slm.get_phase().detach()
