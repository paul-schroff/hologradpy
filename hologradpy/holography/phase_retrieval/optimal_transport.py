from __future__ import annotations

from typing import Callable

import torch

from .abstract import PhaseRetrieverBase
from .recorder import RetrievalRun

from ...optics.systems import SLMFourierLensModel
from ...roi import ROI
from ...utils import ProgressBar
from ...vector_fields import integrate_along_path

# Fraction of the target's peak used to find the box the target is located in.
SUPPORT_THRESHOLD = 0.01


class OptimalTransportPhaseRetriever(PhaseRetrieverBase):
    """A phase guess from the optimal transport of light onto the target.

    This is the scalable optimal transport method:

    A. Torchylo, H. Swan, L. Tellez and J. M. Hogan, "A fast, large-scale optimal
    transport algorithm for holographic beam shaping", arXiv:2512.19072 (2025),
    https://arxiv.org/abs/2512.19072.
    """

    def __init__(
        self,
        slm_camera_model: SLMFourierLensModel,
        target: torch.Tensor | None = None,
        source_intensity: torch.Tensor | None = None,
        regularization: float = 1e-3,
    ) -> None:
        """
        Args:
            slm_camera_model: The model of the optical system.
            target: Target intensity on the model's output grid.
            source_intensity: The beam at the SLM. Defaults to the wavefront the
                model's ``slm_field`` carries, which is the calibrated beam.
            regularization: Entropic weight. Too large shrinks the shaped beam, while
                too small causes the kernel to underflow.
        """
        super().__init__(slm_camera_model)

        self.source_intensity: torch.Tensor | None = source_intensity
        self.regularization: float = regularization
        if target is not None:
            self.set_target(target)

    def set_target(self, target: torch.Tensor) -> None:
        """Update the intensity target. Cropped internally to the smallest box
        containing the nonzero target.

        Args:
            target: Target intensity, on the model's output grid.
        """
        self.target = target.detach()

    def _beam(self) -> torch.Tensor:
        """The beam at the SLM, on the SLM's own grid."""
        if self.source_intensity is not None:
            return self.source_intensity
        return self.slm_camera_model.slm_field.get_wavefront().abs() ** 2

    def retrieve_phase(
        self,
        number_of_iterations: int = 1000,
        *,
        run: RetrievalRun | None = None,
        verbose: bool = True,
        progress_bar: ProgressBar | None = None,
        **_: object,
    ) -> torch.Tensor:
        """Solve the transport and put the phase it implies onto the model.

        Args:
            number_of_iterations: Cap on Sinkhorn iterations. Not gradient steps: there
                is no optimizer here, so the base class's optimizer arguments do not
                apply.
            run: The run to record into. A new one is made when none is given.
            verbose: Show a progress bar over the Sinkhorn sweeps when one is not
                supplied.
            progress_bar: A bar to borrow, reset here and handed back untouched, so a
                feedback loop can drive this retriever like any other.

        Returns:
            torch.Tensor: The phase the SLM is now showing.
        """
        self.timer.start()
        self.run = run if run is not None else RetrievalRun()

        borrowed = progress_bar is not None
        if borrowed:
            progress_bar.reset(total=number_of_iterations)
        else:
            progress_bar = ProgressBar(
                total=number_of_iterations,
                description="Optimal transport",
                verbose=verbose,
            ).__enter__()

        virtual_slm = self.slm_camera_model.virtual_slm
        try:
            phase = self._transport_phase(number_of_iterations, progress_bar.update)
        finally:
            if not borrowed:
                progress_bar.close()

        virtual_slm.set_phase(phase.to(torch.float32))
        self.timer.stop()
        return virtual_slm.get_phase().detach()

    def _transport_phase(
        self, number_of_iterations: int, on_sweep: Callable[[], None] | None = None
    ) -> torch.Tensor:
        """The phase from the transport map at the SLM's resolution."""
        model = self.slm_camera_model
        virtual_slm = model.virtual_slm

        slm_x, slm_y = virtual_slm.get_slm_grid()

        # The grids are tensor products, so one row and one column between them carry
        # every coordinate the separable kernel needs.
        slm_axis_y, slm_axis_x = slm_y[:, 0].double(), slm_x[0, :].double()
        beam = self._beam().double()

        # Crop the target to the box it occupies. A box is still a tensor product, so
        # the kernel stays separable and empty cells only cost a row of it. It also
        # keeps the regularization meaningful: a target adrift on a wide canvas would
        # otherwise take its scale from the canvas corner rather than from itself.
        camera_x, camera_y = model[-1].get_spatial_grid_output()
        region = ROI.detect(self.target, threshold=SUPPORT_THRESHOLD, pad=1)
        target = region.crop(self.target).double()
        target_axis_y = region.crop(camera_y)[:, 0].double()
        target_axis_x = region.crop(camera_x)[0, :].double()

        # One scalar per plane, never one per axis.
        slm_scale = max(slm_axis_y.abs().max(), slm_axis_x.abs().max())
        target_scale = max(target_axis_y.abs().max(), target_axis_x.abs().max())

        epsilon = self.regularization
        kernel_y = _kernel(
            slm_axis_y / slm_scale, target_axis_y / target_scale, epsilon
        )
        kernel_x = _kernel(
            slm_axis_x / slm_scale, target_axis_x / target_scale, epsilon
        )

        # Weighted by the target's own metres, so the map is returned in metres.
        map_y, map_x = _separable_map(
            beam,
            target,
            kernel_y,
            kernel_x,
            target_axis_y,
            target_axis_x,
            number_of_iterations,
            on_sweep,
        )

        pitch = (
            float(slm_axis_y[1] - slm_axis_y[0]),
            float(slm_axis_x[1] - slm_axis_x[0]),
        )
        wavenumber = float(model.input_geometry.wavenumber.reshape(()))
        return (wavenumber / model.focal_length) * integrate_along_path(
            map_x, map_y, pitch
        )


def _kernel(
    source_axis: torch.Tensor, target_axis: torch.Tensor, epsilon: float
) -> torch.Tensor:
    """The exponentiated cost for one axis, as a ``(source, target)`` matrix.

    ``Lambda`` in the paper, where the square case is written ``Lambda V Lambda``.
    One rectangular matrix per axis lets the planes differ in shape and in sample
    count.
    """
    separation = source_axis[:, None] - target_axis[None, :]
    return torch.exp(-(separation**2) / (2 * epsilon))


def _safe_divide(numerator: torch.Tensor, denominator: torch.Tensor) -> torch.Tensor:
    """Divide, giving zero wherever the denominator has underflowed."""
    floor = denominator.max() * 1e-250
    return torch.where(
        denominator > floor, numerator / denominator.clamp(min=floor), 0.0
    )


def _separable_map(
    source: torch.Tensor,
    target: torch.Tensor,
    kernel_y: torch.Tensor,
    kernel_x: torch.Tensor,
    target_axis_y: torch.Tensor,
    target_axis_x: torch.Tensor,
    number_of_iterations: int,
    on_sweep: Callable[[], None] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Where each source pixel sends its light.

    Follows Algorithm 1 of Torchylo et al., arXiv:2512.19072, generalized to rectangular
    kernels.

    Args:
        source: Beam intensity, ``(a, b)``, any positive scaling.
        target: Target intensity, ``(c, d)``, independent of the source's shape.
        kernel_y: ``(a, c)`` exponentiated cost along the first axis.
        kernel_x: ``(b, d)`` exponentiated cost along the second.
        target_axis_y: The target's ``(c,)`` coordinates along the first axis, in the
            units the map should come back in.
        target_axis_x: The target's ``(d,)`` coordinates along the second.
        number_of_iterations: Sinkhorn sweeps.
        on_sweep: Called once per sweep, to drive a progress bar.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: The ``(y, x)`` components of the map, on the
        source's grid.
    """
    # Float64 needed throughout.
    beam = (source / source.sum()).double()
    wanted = (target / target.sum()).double()
    kernel_y, kernel_x = kernel_y.double(), kernel_x.double()

    scaling = torch.full_like(beam, 1 / beam.numel())
    other = torch.full_like(wanted, 1 / wanted.numel())
    for _ in range(number_of_iterations):
        scaling = _safe_divide(beam, kernel_y @ other @ kernel_x.T)
        scaling = scaling / scaling.max()

        other = _safe_divide(wanted, kernel_y.T @ scaling @ kernel_x)
        if on_sweep is not None:
            on_sweep()

    return (
        scaling
        * _safe_divide(kernel_y @ (target_axis_y[:, None] * other) @ kernel_x.T, beam),
        scaling
        * _safe_divide(kernel_y @ (other * target_axis_x[None, :]) @ kernel_x.T, beam),
    )
