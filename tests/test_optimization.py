"""The conjugate gradient solver for linear least squares, and the Levenberg-Marquardt
optimizer built on the same conjugate gradients.
"""

from __future__ import annotations

import torch
from scipy.optimize import least_squares

from hologradpy.optimization import (
    LevenbergMarquardt,
    _ChunkedResidual,
    solve_linear_least_squares,
)


def _random_operator(rows: int, columns: int, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.complex(
        torch.randn(rows, columns, generator=generator, dtype=torch.float64),
        torch.randn(rows, columns, generator=generator, dtype=torch.float64),
    )


def _solve(matrix: torch.Tensor, data: torch.Tensor) -> tuple[torch.Tensor, float]:
    return solve_linear_least_squares(
        lambda x: matrix @ x,
        lambda y: matrix.conj().T @ y,
        data,
    )


def test_matches_the_dense_least_squares_solution() -> None:
    """An overdetermined system has a residual, and the solution and its error agree
    with a direct solve.
    """
    matrix = _random_operator(40, 12, seed=0)
    data = _random_operator(40, 1, seed=1)[:, 0]

    solution, relative_error = _solve(matrix, data)

    expected = torch.linalg.lstsq(matrix, data.unsqueeze(-1)).solution[:, 0]
    expected_error = float((matrix @ expected - data).norm() / data.norm())
    torch.testing.assert_close(solution, expected, rtol=1e-6, atol=1e-6)
    assert abs(relative_error - expected_error) < 1e-6
    assert relative_error > 0.1


def test_recovers_the_solution_of_a_consistent_system() -> None:
    matrix = _random_operator(40, 12, seed=2)
    true_solution = _random_operator(12, 1, seed=3)[:, 0]

    solution, relative_error = _solve(matrix, matrix @ true_solution)

    torch.testing.assert_close(solution, true_solution, rtol=1e-6, atol=1e-6)
    assert relative_error < 1e-6


# --- Levenberg-Marquardt ------------------------------------------------------------

POSITIONS = torch.linspace(-5.0, 5.0, 400, dtype=torch.float64)
# Amplitude, centre and width of the peak, and a start far from them.
TRUE_PEAK = torch.tensor([2.0, 0.7, 1.3], dtype=torch.float64)
ROUGH_START = torch.tensor([1.0, -0.5, 2.0], dtype=torch.float64)


def _peak(parameters: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    amplitude, centre, width = parameters
    return amplitude * torch.exp(-((positions - centre) ** 2) / (2 * width**2))


def _noisy_peak() -> torch.Tensor:
    generator = torch.Generator().manual_seed(0)
    noise = torch.randn(POSITIONS.shape, generator=generator, dtype=torch.float64)
    return _peak(TRUE_PEAK, POSITIONS) + 0.05 * noise


def _peak_residual(data: torch.Tensor, number_of_chunks: int):
    """The residual of the peak against ``data``, in chunks of equal length."""
    length = len(POSITIONS) // number_of_chunks

    def chunk_residual(parameters: torch.Tensor, chunk: int) -> torch.Tensor:
        part = slice(chunk * length, (chunk + 1) * length)
        return _peak(parameters, POSITIONS[part]) - data[part]

    return chunk_residual


def _levenberg_marquardt(
    chunk_residual,
    starts: list[torch.Tensor],
    number_of_chunks: int,
    steps: int,
    cg_steps: int = 10,
) -> tuple[torch.Tensor, list[float]]:
    """The fitted parameters, joined into one vector, and the cost before every step and
    after the last. ``chunk_residual(vector, chunk)`` takes the parameters joined into
    one vector. Each tensor of ``starts`` is one parameter, and so one block.
    """
    parameters = [torch.nn.Parameter(start.clone()) for start in starts]
    optimizer = LevenbergMarquardt(
        parameters, number_of_chunks=number_of_chunks, cg_steps=cg_steps
    )

    def closure(chunk, values):
        return chunk_residual(
            torch.cat(list(parameters if values is None else values)), chunk
        )

    costs = [optimizer.step(closure) for _ in range(steps)]
    with torch.no_grad():
        costs.append(
            0.5
            * sum(
                float((closure(chunk, None) ** 2).sum())
                for chunk in range(number_of_chunks)
            )
        )
    return torch.cat([parameter.detach() for parameter in parameters]), costs


def test_a_peak_fit_reaches_the_least_squares_optimum() -> None:
    """From a start far off in every parameter, the fit lands on the optimum that
    scipy's trust-region solver finds, and the cost never rises.
    """
    data = _noisy_peak()

    fitted, costs = _levenberg_marquardt(
        _peak_residual(data, 1), [ROUGH_START], 1, steps=20
    )

    reference = least_squares(
        lambda values: (_peak(torch.as_tensor(values), POSITIONS) - data).numpy(),
        ROUGH_START.numpy(),
        xtol=1e-14,
        ftol=1e-14,
    )
    torch.testing.assert_close(
        fitted, torch.as_tensor(reference.x), rtol=1e-6, atol=1e-8
    )
    assert len(costs) == 21
    assert all(later <= earlier for earlier, later in zip(costs, costs[1:]))
    assert costs[-1] < 0.01 * costs[0]


def test_one_chunk_and_several_give_the_same_parameters() -> None:
    data = _noisy_peak()

    whole, _ = _levenberg_marquardt(_peak_residual(data, 1), [ROUGH_START], 1, steps=20)
    chunked, _ = _levenberg_marquardt(
        _peak_residual(data, 4), [ROUGH_START], 4, steps=20
    )

    torch.testing.assert_close(chunked, whole, rtol=1e-6, atol=1e-8)


def _two_block_residual(steepness: float):
    """A residual whose second block of two parameters is ``steepness`` times as steep
    as its first. The curvatures of the two blocks differ by the square of
    ``steepness``.
    """
    generator = torch.Generator().manual_seed(1)
    design = torch.randn(300, 4, generator=generator, dtype=torch.float64)
    target = torch.randn(300, generator=generator, dtype=torch.float64)

    def chunk_residual(parameters: torch.Tensor, chunk: int) -> torch.Tensor:
        scaled = torch.cat([parameters[:2], steepness * parameters[2:]])
        return torch.tanh(design @ scaled) - target

    return chunk_residual


def test_blocks_of_different_curvature_converge_as_alike_ones() -> None:
    """With the curvatures of two blocks 1e4 apart, the fit takes the same steps as with
    them alike, since each block is scaled by its own curvature. In six steps of two
    conjugate gradient steps each, it reaches the optimum. One block over both leaves
    more than 1 % of the cost above the optimum.
    """
    start = torch.zeros(4, dtype=torch.float64)
    blocks = [start[:2], start[2:]]

    _, alike = _levenberg_marquardt(
        _two_block_residual(1.0), blocks, 1, steps=6, cg_steps=2
    )
    _, apart = _levenberg_marquardt(
        _two_block_residual(100.0), blocks, 1, steps=6, cg_steps=2
    )
    _, one_block = _levenberg_marquardt(
        _two_block_residual(100.0), [start], 1, steps=6, cg_steps=2
    )
    _, converged = _levenberg_marquardt(
        _two_block_residual(100.0), blocks, 1, steps=30
    )

    torch.testing.assert_close(
        torch.tensor(apart), torch.tensor(alike), rtol=1e-9, atol=0.0
    )
    optimum = converged[-1]
    start_excess = apart[0] - optimum
    assert apart[-1] - optimum < 1e-5 * start_excess
    assert one_block[-1] - optimum > 1e-2 * start_excess


def test_the_jacobian_products_are_transposes() -> None:
    """The one-sided difference and the backward pass apply the same Jacobian,
    ``<J v, w> = <v, J^T w>``, chunk by chunk.
    """
    generator = torch.Generator().manual_seed(2)
    residual = _ChunkedResidual(_peak_residual(_noisy_peak(), 4), 4, ROUGH_START)
    direction = torch.randn(3, generator=generator, dtype=torch.float64)
    weights = [
        torch.randn(100, generator=generator, dtype=torch.float64) for _ in range(4)
    ]

    products = residual.jacobian_product(direction, size=1e-7)
    forward = sum(
        float((product * weight).sum()) for product, weight in zip(products, weights)
    )
    backward = float((direction * residual.transpose_product(weights)).sum())

    assert abs(forward - backward) < 1e-5 * abs(backward)


def test_a_block_that_leaves_the_residual_alone_keeps_its_values() -> None:
    """A block without curvature cannot be fitted, so it keeps its start. The other
    blocks are fitted as without it.
    """
    data = _noisy_peak()

    def residual_with_an_idle_block(parameters: torch.Tensor, chunk: int):
        return _peak(parameters[:3], POSITIONS) - data

    fitted, _ = _levenberg_marquardt(
        residual_with_an_idle_block,
        [ROUGH_START, torch.tensor([5.0], dtype=torch.float64)],
        1,
        steps=20,
    )
    alone, _ = _levenberg_marquardt(_peak_residual(data, 1), [ROUGH_START], 1, steps=20)

    assert float(fitted[3]) == 5.0
    torch.testing.assert_close(fitted[:3], alone, rtol=1e-6, atol=1e-8)
