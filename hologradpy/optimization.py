"""Optimization routines shared across the package."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence

import torch
from torch import Tensor

from .utils import as_real_vector, from_real_vector


def solve_linear_least_squares(
    forward: Callable[[Tensor], Tensor],
    adjoint: Callable[[Tensor], Tensor],
    data: Tensor,
    tolerance: float = 1e-6,
    max_iterations: int = 500,
) -> tuple[Tensor, float]:
    """The ``x`` that minimizes ``||forward(x) - data||``.

    Conjugate gradients run on the normal equations,
    ``adjoint(forward(x)) = adjoint(data)``. The iteration starts from zero in the
    shape, dtype and device of ``adjoint(data)``, so ``adjoint`` sets the shape of
    ``x``. The operator is only applied and never stored, so a transform over millions
    of samples costs one forward and one adjoint call per step.

    Args:
        forward: The linear operator, mapping ``x`` onto a tensor of the shape of
            ``data``.
        adjoint: The adjoint of ``forward``, mapping a tensor of the shape of ``data``
            onto one of the shape of ``x``.
        data: The values ``forward(x)`` is fitted to.
        tolerance: The iteration stops once the norm of the gradient of the squared
            error falls to this fraction of its initial value.
        max_iterations: The largest number of steps the iteration takes.

    Returns:
        tuple[Tensor, float]: The solution, in the shape, dtype and device of
        ``adjoint(data)``, and the relative RMS error
        ``||forward(x) - data|| / ||data||``.
    """
    with torch.no_grad():
        solution = _conjugate_gradient(
            lambda vector: adjoint(forward(vector)),
            adjoint(data),
            max_iterations,
            tolerance**2,
        )
        residual = data - forward(solution)
        relative_error = float(
            ((residual.abs() ** 2).sum() / (data.abs() ** 2).sum()).sqrt()
        )
    return solution, relative_error


class LevenbergMarquardt(torch.optim.Optimizer):
    """Levenberg-Marquardt steps of the parameters that minimize half the sum of squares
    of a residual.

    The residual is given in chunks, so a large fit holds the graph of one chunk at a
    time. Each step solves the damped Gauss-Newton equations
    ``(J^T J + damping) step = -J^T r`` by at most ``cg_steps`` conjugate gradient
    steps. The conjugate gradients stop once the squared norm of their residual falls
    to 1e-4 of its first value. The Jacobian ``J`` is only applied. A product ``J v``
    is a one-sided difference from the residuals at the current parameters, with one
    forward pass per chunk. A product ``J^T w`` is one backward pass per chunk.

    Each parameter tensor is a block, scaled to unit curvature, the mean diagonal of
    ``J^T J`` over the block. This is Marquardt's scaling of the normal equations by
    their diagonal, taken per block, so the damping and the conjugate gradients treat
    blocks of different units alike. The curvature is measured at the first step from a
    backward pass of random weights ``w``, since ``E |J_b^T w|^2 = trace(J_b^T J_b)``.
    The difference step moves the scaled parameters by 1e-3 of the residual's norm, so
    it changes the residual by a similar amount along every block.

    The damping starts at 1e-3 times the curvature along the gradient. A step that
    lowers the cost is taken and divides the damping by 3. Any other step is dropped
    and multiplies the damping by 4 (Madsen, Nielsen and Tingleff 2004, Sec. 3.2).

    The closure of :meth:`step` is ``closure(chunk, values)``. It returns the real
    residual of one chunk of the data with the parameters replaced by ``values``, one
    tensor per parameter in their order, differentiable in ``values``. For None, it
    takes the parameters themselves. The parameters change only when a step lowers the
    cost. A step dropped leaves the residual and its graph at the current point for the
    next step with the same closure.
    """

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        number_of_chunks: int = 1,
        cg_steps: int = 10,
    ) -> None:
        """
        Args:
            params: The parameters to fit, real or complex, of one real dtype.
            number_of_chunks: The chunks the residual is split into. With a single
                chunk, the graph is held through each step.
            cg_steps: The conjugate gradient steps of each Levenberg-Marquardt step.
        """
        super().__init__(params, {})
        self.number_of_chunks: int = number_of_chunks
        self.cg_steps: int = cg_steps
        self._scale: Tensor | None = None
        self._damping: float | None = None
        self._residual: _ChunkedResidual | None = None
        self._closure: Callable[[int, list[Tensor] | None], Tensor] | None = None

    def step(self, closure: Callable[[int, list[Tensor] | None], Tensor]) -> float:
        """Take one Levenberg-Marquardt step, and return the cost before it.

        Args:
            closure: ``closure(chunk, values)``, the residual of one chunk with the
                parameters replaced by ``values``.

        Returns:
            float: Half the sum of squares of the residual before the step. The
            parameters stay where they are when the gradient there vanishes.
        """
        parameters = [p for group in self.param_groups for p in group["params"]]
        point, blocks = as_real_vector(parameters)
        residual = self._residual
        if (
            residual is None
            or closure is not self._closure
            or not torch.equal(residual.point, point)
        ):
            residual = _ChunkedResidual(
                lambda values, chunk: closure(
                    chunk, from_real_vector(values, parameters, blocks)
                ),
                self.number_of_chunks,
                point,
            )
            self._residual, self._closure = residual, closure
        cost = residual.cost
        if self._scale is None:
            self._scale = self._block_scale(residual, blocks)
        scale = self._scale

        def scaled_jacobian_product(scaled_direction: Tensor) -> list[Tensor]:
            """``J`` applied to ``scale * scaled_direction``, by a one-sided
            difference.
            """
            size = (
                1e-3
                * (2.0 * residual.cost) ** 0.5
                / float(scaled_direction.double().norm())
            )
            return residual.jacobian_product(scale * scaled_direction, size)

        scaled_gradient = scale * residual.transpose_product(residual.values)
        gradient_norm = float((scaled_gradient.double() ** 2).sum())
        if gradient_norm == 0.0:
            return cost
        if self._damping is None:
            curvature = sum(
                float((product.double() ** 2).sum())
                for product in scaled_jacobian_product(scaled_gradient)
            )
            self._damping = 1e-3 * curvature / gradient_norm
        damping = self._damping

        # The damped normal equations (S J^T J S + damping) z = -S J^T r, with S the
        # scale and z the scaled step.
        scaled_step = _conjugate_gradient(
            lambda direction: scale
            * residual.transpose_product(scaled_jacobian_product(direction))
            + damping * direction,
            -scaled_gradient,
            self.cg_steps,
            1e-4,
        )

        trial = residual.point + scale * scaled_step
        trial_values = residual.values_at(trial)
        if _cost_of(trial_values) < cost:
            residual.move_to(trial, trial_values)
            with torch.no_grad():
                for parameter, value in zip(
                    parameters, from_real_vector(trial, parameters, blocks)
                ):
                    parameter.copy_(value)
            self._damping = damping / 3.0
        else:
            self._damping = damping * 4.0
        return cost

    @staticmethod
    def _block_scale(residual: _ChunkedResidual, blocks: Sequence[slice]) -> Tensor:
        """The scale of every block, the inverse square root of its curvature, or zero
        for a block that leaves the residual unchanged.
        """
        generator = torch.Generator(device=residual.values[0].device).manual_seed(0)
        random_weights = [
            torch.randn(
                values.shape,
                generator=generator,
                device=values.device,
                dtype=values.dtype,
            )
            for values in residual.values
        ]
        sensitivity = residual.transpose_product(random_weights)
        scale = torch.zeros_like(residual.point)
        for block in blocks:
            curvature = float((sensitivity[block].double() ** 2).mean())
            if curvature > 0.0:
                scale[block] = curvature**-0.5
        return scale


def _conjugate_gradient(
    apply: Callable[[Tensor], Tensor],
    right_hand_side: Tensor,
    steps: int,
    tolerance: float,
) -> Tensor:
    """The ``x`` that solves ``apply(x) = right_hand_side``, for a Hermitian positive
    definite ``apply``, by conjugate gradients from ``x = 0``.

    The steps end after ``steps``, once the squared norm of the residual falls to
    ``tolerance`` times its first value, or at a direction of no positive curvature.
    """
    solution = torch.zeros_like(right_hand_side)
    residual = right_hand_side.clone()
    direction = residual.clone()
    residual_norm = _squared_norm(residual)
    first_norm = residual_norm
    for _ in range(steps):
        if residual_norm <= tolerance * first_norm:
            break
        product = apply(direction)
        curvature = float((direction.conj() * product).real.double().sum())
        if curvature <= 0.0:
            break
        step_length = residual_norm / curvature
        solution = solution + step_length * direction
        residual = residual - step_length * product
        next_norm = _squared_norm(residual)
        direction = residual + (next_norm / residual_norm) * direction
        residual_norm = next_norm
    return solution


def _squared_norm(values: Tensor) -> float:
    """The sum of the squared magnitudes of ``values``, in double precision."""
    return float((values.abs().double() ** 2).sum())


def _cost_of(values: list[Tensor]) -> float:
    """Half the sum of squares of the residual of every chunk."""
    return 0.5 * sum(float((chunk.double() ** 2).sum()) for chunk in values)


class _ChunkedResidual:
    """A residual given in chunks, at a current point, with its Jacobian products
    there. With a single chunk, the graph at the point is held for every transpose
    product.
    """

    def __init__(
        self,
        chunk_residual: Callable[[Tensor, int], Tensor],
        number_of_chunks: int,
        point: Tensor,
    ) -> None:
        self.chunk_residual = chunk_residual
        self.chunks = range(number_of_chunks)
        # The leaf and the residual of a single chunk at the current point.
        self._held_graph: tuple[Tensor, Tensor] | None = None
        self.move_to(point)

    def values_at(self, point: Tensor) -> list[Tensor]:
        """The residual of every chunk at ``point``, without a graph."""
        with torch.no_grad():
            return [self.chunk_residual(point, chunk) for chunk in self.chunks]

    def move_to(self, point: Tensor, values: list[Tensor] | None = None) -> None:
        """Make ``point`` the current point, with its residuals ``values`` when they
        are known. With a single chunk, the residual is evaluated again to hold the
        graph.
        """
        self.point = point
        if len(self.chunks) == 1:
            # The graph at the previous point is let go before the next is built.
            self._held_graph = None
            leaf = point.clone().requires_grad_(True)
            values_with_graph = self.chunk_residual(leaf, 0)
            self._held_graph = (leaf, values_with_graph)
            self.values = [values_with_graph.detach()]
        else:
            self.values = self.values_at(point) if values is None else values
        self.cost = _cost_of(self.values)

    def jacobian_product(self, direction: Tensor, size: float) -> list[Tensor]:
        """``J`` applied to ``direction``, per chunk, as the one-sided difference from
        the current point over ``size`` times ``direction``.
        """
        moved = self.values_at(self.point + size * direction)
        return [(new - old) / size for new, old in zip(moved, self.values)]

    def transpose_product(self, weights: list[Tensor]) -> Tensor:
        """``J^T`` applied to one weight per chunk, one backward pass per chunk."""
        if self._held_graph is not None:
            leaf, values_with_graph = self._held_graph
            return torch.autograd.grad(
                values_with_graph, leaf, weights[0], retain_graph=True
            )[0]
        total = torch.zeros_like(self.point)
        for chunk, weight in zip(self.chunks, weights):
            leaf = self.point.clone().requires_grad_(True)
            values = self.chunk_residual(leaf, chunk)
            total = total + torch.autograd.grad(values, leaf, weight)[0]
        return total
