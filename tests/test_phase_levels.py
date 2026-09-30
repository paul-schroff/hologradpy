"""The phase response of an SLM, as a straight line or as a table of one phase per
level.

A table read off a straight line displays the levels the line does, for every phase
and at every bit depth. The table's checks and its sign convention are pinned here as
well.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest
import torch

from hologradpy.phase_levels import (
    LinearResponse,
    LookupResponse,
    PhaseResponseModule,
)

CYCLES = np.array([0.0, -0.25, 0.25, 0.5, -0.5, 0.75, -0.75, -1.5, -1.0])


def _line_and_table(
    bitdepth: int, full_scale_cycles: float
) -> tuple[LinearResponse, LookupResponse]:
    """A straight line and the table read off it, one entry per level."""
    line = LinearResponse(bitdepth=bitdepth, full_scale_cycles=full_scale_cycles)
    levels = np.arange(line.number_of_levels)
    return line, LookupResponse(bitdepth=bitdepth, phases=line.to_phase(levels))


def _curved_delays(bitdepth: int, span: float) -> np.ndarray:
    """The delays of a monotone S-shaped curve reaching ``span`` radians."""
    levels = np.arange(2**bitdepth)
    return span * (0.5 - 0.5 * np.cos(np.pi * levels / levels[-1]))


# --- A table of a line behaves as the line --------------------------------------------


@pytest.mark.parametrize("bitdepth", [8, 12])
@pytest.mark.parametrize("full_scale_cycles", [0.6, 0.95, 1.0, 1.2, 1.5, 2.0])
def test_a_table_of_a_line_displays_the_levels_the_line_does(
    bitdepth: int, full_scale_cycles: float
) -> None:
    line, table = _line_and_table(bitdepth, full_scale_cycles)
    phases = np.concatenate(
        [
            np.random.default_rng(0).uniform(-30.0, 30.0, 100_000),
            table.phases,
            CYCLES * 2 * np.pi,
        ]
    )

    assert np.array_equal(table.display_levels(phases), line.display_levels(phases))
    np.testing.assert_allclose(
        table.to_levels(phases), line.to_levels(phases), atol=1e-9, rtol=0
    )
    assert table.full_scale_cycles == pytest.approx(full_scale_cycles, rel=1e-12)


def test_positive_phases_wrap_onto_their_levels() -> None:
    """A phase and the same phase a cycle further on show the same level."""
    _, table = _line_and_table(8, 1.0)

    assert table.display_levels(CYCLES[:7] * 2 * np.pi).tolist() == [
        0,
        64,
        192,
        128,
        128,
        64,
        192,
    ]
    assert table.display_levels(CYCLES[7:] * 2 * np.pi).tolist() == [128, 0]


@pytest.mark.parametrize("bitdepth", [8, 12])
@pytest.mark.parametrize("full_scale_cycles", [0.6, 1.0, 1.5])
def test_a_table_of_a_line_reads_back_the_line(
    bitdepth: int, full_scale_cycles: float
) -> None:
    """The table continues past full scale as the line does, in NumPy and in torch."""
    line, table = _line_and_table(bitdepth, full_scale_cycles)
    fractions = np.concatenate(
        [
            np.random.default_rng(1).uniform(-3.0, 3.0, 10_000),
            [0.0, 1.0 - 1.0 / 2**bitdepth, 1.0],
        ]
    )
    as_tensor = torch.as_tensor(fractions, dtype=torch.float64)
    edges = np.array([0.9985, 1.2, -0.1])

    np.testing.assert_allclose(
        table.phase_at(fractions), line.phase_at(fractions), atol=1e-12, rtol=0
    )
    torch.testing.assert_close(
        table.phase_at(as_tensor), line.phase_at(as_tensor), atol=1e-12, rtol=0
    )
    np.testing.assert_allclose(
        table.wrap_fraction(edges), line.wrap_fraction(edges), atol=1e-12, rtol=0
    )


def test_a_table_at_one_cycle_wraps_past_its_end() -> None:
    _, table = _line_and_table(8, 1.0)

    assert table.wrap_fraction(np.array([0.9985, 1.2, -0.1])) == pytest.approx(
        [0.9985, 0.2, 0.9], abs=1e-12
    )


@pytest.mark.parametrize("full_scale_cycles", [0.6, 1.0, 1.5])
def test_the_gradient_crosses_the_end_of_the_table(full_scale_cycles: float) -> None:
    """Past the last entry the table keeps the slope of the line, so a fit that
    optimizes levels directly never stalls there.
    """
    _, table = _line_and_table(8, full_scale_cycles)
    fraction = torch.tensor(
        [0.9985, 1.2, -0.1], dtype=torch.float64, requires_grad=True
    )

    table.phase_at(fraction).sum().backward()

    torch.testing.assert_close(
        fraction.grad,
        torch.full((3,), -2 * np.pi * full_scale_cycles, dtype=torch.float64),
    )


def test_a_curved_table_inverts_to_the_phase_it_was_asked_for() -> None:
    """Through a curved table past a full cycle, the phase comes back wrapped and with
    unit gradient.
    """
    table = LookupResponse.from_phase_delays(8, _curved_delays(8, 2.4 * np.pi))
    phase = torch.as_tensor(
        np.random.default_rng(2).uniform(-np.pi, np.pi, 2000), dtype=torch.float64
    ).requires_grad_()

    round_trip = table.phase_at(table.fraction_at(phase))
    round_trip.sum().backward()

    torch.testing.assert_close(phase.grad, torch.ones_like(phase))
    wrapped_error = torch.angle(torch.exp(1j * (round_trip - phase).detach()))
    assert float(wrapped_error.abs().max()) < 1e-12


@pytest.mark.parametrize("bitdepth", [8, 12])
@pytest.mark.parametrize("full_scale_cycles", [0.6, 1.0, 1.5])
def test_torch_and_numpy_give_the_same_levels(
    bitdepth: int, full_scale_cycles: float
) -> None:
    """Exactly in float64. In float32 a table and its line can round a phase on a
    half-level boundary one level apart.
    """
    line, table = _line_and_table(bitdepth, full_scale_cycles)
    phases = np.random.default_rng(3).uniform(-30.0, 30.0, 20_000)
    in_float32 = torch.as_tensor(phases, dtype=torch.float32)

    in_float64 = table.display_levels(torch.as_tensor(phases, dtype=torch.float64))
    assert np.array_equal(in_float64.numpy(), table.display_levels(phases))

    difference = np.abs(
        table.display_levels(in_float32).numpy().astype(np.int64)
        - line.display_levels(in_float32).numpy().astype(np.int64)
    )
    around_the_circle = np.minimum(difference, 2**bitdepth - difference)
    assert around_the_circle.max() <= 1


@pytest.mark.parametrize("two_pi_level", [256, 255, 213])
def test_a_vendor_delay_table_reads_as_the_line(two_pi_level: int) -> None:
    """A vendor table gives the delay of every level, reaching two pi at one level."""
    table = LookupResponse.from_phase_delays(
        8, 2 * np.pi * np.arange(256) / two_pi_level
    )
    line = LinearResponse(bitdepth=8, full_scale_cycles=256 / two_pi_level)
    phases = np.random.default_rng(4).uniform(-30.0, 30.0, 100_000)

    assert table.full_scale_cycles == pytest.approx(256 / two_pi_level)
    assert np.array_equal(table.display_levels(phases), line.display_levels(phases))


def test_an_unreachable_phase_is_clipped_to_the_last_level() -> None:
    """At 0.6 of a cycle, the delays between the last level and a full cycle are out of
    reach, and each is clipped to the delay of the last level.
    """
    line, table = _line_and_table(8, 0.6)
    phases = np.array([-1.4 * np.pi, -1.9 * np.pi, 0.3 * np.pi, 0.01])

    for response in (line, table):
        np.testing.assert_array_equal(response.to_levels(phases), [255, 255, 255, 255])


# --- The table's checks ---------------------------------------------------------------


@pytest.mark.parametrize(
    ("phases", "message"),
    [
        (2 * np.pi * np.arange(256) / 256, "from_phase_delays"),
        (np.zeros(256), "same phase"),
        (-np.sin(np.linspace(0.0, 3 * np.pi, 256)), "numpy.unwrap"),
        (np.where(np.arange(256) == 10, np.nan, -np.arange(256.0)), "NaN"),
        (-np.arange(255.0), "needs 256 phases"),
    ],
    ids=["rising", "flat", "rising and falling", "nan", "short"],
)
def test_a_table_the_response_cannot_use_is_refused(
    phases: np.ndarray, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        LookupResponse(bitdepth=8, phases=phases)


def test_a_table_is_taken_relative_to_level_zero() -> None:
    """A constant offset changes no hologram, so the table is shifted to start at
    zero.
    """
    table = LookupResponse.from_phase_delays(8, _curved_delays(8, 1.9 * np.pi))
    shifted = LookupResponse(bitdepth=8, phases=table.phases + 1.3)
    phases = np.random.default_rng(5).uniform(-30.0, 30.0, 10_000)

    assert shifted.phases[0] == 0.0
    np.testing.assert_allclose(shifted.phases, table.phases, atol=1e-12)
    assert np.array_equal(shifted.display_levels(phases), table.display_levels(phases))


def test_a_tensor_table_is_read_in() -> None:
    _, table = _line_and_table(8, 1.0)

    from_tensor = LookupResponse(
        bitdepth=8, phases=torch.tensor(table.phases, requires_grad=True)
    )

    assert np.array_equal(from_tensor.phases, table.phases)


def test_the_stored_table_is_a_read_only_copy() -> None:
    phases = LinearResponse(bitdepth=8).to_phase(np.arange(256))
    response = LookupResponse(bitdepth=8, phases=phases)

    phases[5] = 1.0

    assert response.phases[5] != 1.0
    with pytest.raises(ValueError):
        response.phases[0] = 1.0


def test_a_module_on_a_table_warns_nothing() -> None:
    _, table = _line_and_table(8, 1.0)

    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        module = PhaseResponseModule(table)
        module.fraction_at(torch.zeros(3, dtype=torch.float64))
