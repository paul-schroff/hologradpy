"""Tests for the unified loss-function contract.

The hologram-search costs had no test at all before the two base classes were merged, so
renaming ``loss(field)`` to ``__call__(field, target=None)`` could have changed what
they compute without anything noticing. The values below were captured through the old
API immediately before the change, so they pin the arithmetic across it.
"""

from __future__ import annotations

import pytest
import torch

from hologradpy.loss_functions import (
    AmplitudeSmoothness,
    normalize_single_to_unit_sum,
    normalize_to_unit_sum,
    smallest_divisor,
    LossEfficiency,
    LossAbsoluteIntensityMSE,
    LossFidelity,
    LossFunction,
    LossIntensityMSE,
    LossVorticity,
    MaskedIntensityMSE,
    PhaseSmoothness,
    SumOfLosses,
    gradient_loss,
)

RESOLUTION = (16, 24)


def _fixture():
    """The exact inputs the baseline values were captured on."""
    torch.manual_seed(0)

    target_intensity = torch.rand(RESOLUTION, dtype=torch.float64) + 0.1
    target_phase = torch.rand(RESOLUTION, dtype=torch.float64) * 2 - 1
    signal_mask = torch.zeros(RESOLUTION, dtype=torch.float64)
    signal_mask[4:12, 6:18] = 1.0

    field = (
        torch.rand(RESOLUTION, dtype=torch.float64)
        * torch.exp(1j * torch.rand(RESOLUTION, dtype=torch.float64))
    )
    return target_intensity, target_phase, signal_mask, field


# Captured through the pre-merge ``.loss(field)`` API. LossFidelity is the exception:
# its baseline was recaptured when its normalisation was corrected, since the pinned
# value encoded a denominator that took the square root before summing and so could
# never reach zero. See test_a_matching_field_costs_nothing.
BASELINES = {
    "LossIntensityMSE": 9827295079.770136,
    "LossFidelity": 157866822238.04988,
    "LossEfficiency": -10065783943216.678,
    "LossVorticity": 9.900637990590128e16,
}


def _hologram_losses():
    target_intensity, target_phase, signal_mask, field = _fixture()
    return {
        "LossIntensityMSE": LossIntensityMSE(target_intensity, signal_mask),
        "LossFidelity": LossFidelity(target_intensity, target_phase, signal_mask),
        "LossEfficiency": LossEfficiency(signal_mask, total_power=torch.tensor(3.0)),
        "LossVorticity": LossVorticity(target_intensity),
    }, field


@pytest.mark.parametrize("name", sorted(BASELINES))
def test_hologram_losses_are_unchanged_by_the_merge(name) -> None:
    """The arithmetic each cost performs must survive the rename untouched."""
    losses, field = _hologram_losses()

    assert float(losses[name](field)) == pytest.approx(BASELINES[name], rel=1e-12)


def test_a_matching_field_costs_nothing() -> None:
    """The fidelity cost must bottom out at zero when the field is the target.

    The overlap is a Cauchy-Schwarz ratio, so it reaches one only when the square root
    is taken over the summed powers. Taking it pixel by pixel instead caps the overlap
    somewhere far below one and leaves a floor the optimiser cannot get under.
    """
    target_intensity, target_phase, signal_mask, _ = _fixture()
    loss = LossFidelity(target_intensity, target_phase, signal_mask)

    matching = target_intensity.sqrt() * torch.exp(1j * target_phase) * signal_mask

    assert float(loss(matching)) == pytest.approx(0.0, abs=1e-12)


def test_the_fidelity_cost_ignores_brightness() -> None:
    """Scaling the field must not change the cost.

    The normalisation is over the power inside the signal region, which is what lets
    the cost be used with a model that images only a window of the plane and never
    computes the light outside it.
    """
    target_intensity, target_phase, signal_mask, field = _fixture()
    loss = LossFidelity(target_intensity, target_phase, signal_mask)

    assert float(loss(field * 7.0)) == pytest.approx(float(loss(field)), rel=1e-12)


@pytest.mark.parametrize("name", sorted(BASELINES))
def test_a_stored_target_ignores_a_passed_one(name) -> None:
    """These costs fix their target at construction, so the optional second argument
    exists only to satisfy the shared contract and must not change the answer.
    """
    losses, field = _hologram_losses()
    nonsense = torch.full(RESOLUTION, 12345.0, dtype=torch.float64)

    assert float(losses[name](field, nonsense)) == pytest.approx(
        float(losses[name](field)), rel=1e-12
    )


def test_every_cost_shares_one_base() -> None:
    """The point of the merge: one contract, so anything that takes a cost takes all of
    them.
    """
    losses, _ = _hologram_losses()
    _, _, signal_mask, _ = _fixture()

    for loss in losses.values():
        assert isinstance(loss, LossFunction)
    assert isinstance(MaskedIntensityMSE(signal_mask), LossFunction)


def test_costs_compose_across_the_two_families() -> None:
    """A hologram-search cost can now be weighted against another one, which is what the
    efficiency and vorticity terms are for and what the old split prevented.
    """
    losses, field = _hologram_losses()
    first, second = losses["LossIntensityMSE"], losses["LossEfficiency"]

    combined = first + second
    assert isinstance(combined, SumOfLosses)
    assert float(combined(field)) == pytest.approx(
        float(first(field)) + float(second(field)), rel=1e-12
    )


def test_sums_stay_flat() -> None:
    """``a + b + c`` holds three terms rather than nesting a sum inside a sum, so the
    forwarding of the optional target does not deepen with each term.
    """
    losses, field = _hologram_losses()
    terms = [losses[name] for name in ("LossIntensityMSE", "LossEfficiency")]
    terms.append(losses["LossVorticity"])

    combined = terms[0] + terms[1] + terms[2]
    assert len(combined.terms) == 3
    assert float(combined(field)) == pytest.approx(
        sum(float(term(field)) for term in terms), rel=1e-12
    )


def test_the_calibration_costs_are_unchanged_by_the_merge() -> None:
    """Same pinning for the side that already used ``__call__``, since its signature
    widened.
    """
    target_intensity, _, signal_mask, field = _fixture()
    batched_field = field.unsqueeze(0).unsqueeze(0)
    batched_target = target_intensity.unsqueeze(0).unsqueeze(0)

    mismatch = MaskedIntensityMSE(signal_mask)

    assert float(mismatch(batched_field, batched_target)) == pytest.approx(
        0.9434203276579332, rel=1e-12
    )


class _SLMField:
    """The two attributes a prior reads off the field module being fitted."""

    def __init__(self, structured: bool = False) -> None:
        ramp = torch.linspace(0.0, 1.0, RESOLUTION[1], dtype=torch.float64)
        self.phase = (ramp * 3.0).expand(RESOLUTION).clone() if structured else (
            torch.zeros(RESOLUTION, dtype=torch.float64)
        )
        self.amplitude = (
            (2.0 + ramp).expand(RESOLUTION).clone()
            if structured
            else torch.full(RESOLUTION, 2.0, dtype=torch.float64)
        )


@pytest.mark.parametrize("prior", [PhaseSmoothness, AmplitudeSmoothness])
def test_a_prior_ignores_both_arguments(prior) -> None:
    """A prior reads the field module it was built on, which is what lets it be added to
    a data term. Both arguments are optional for that reason.
    """
    smoothness = prior(_SLMField())

    # A flat field has no gradient, so the penalty is zero however it is called.
    assert float(smoothness()) == pytest.approx(0.0, abs=1e-15)
    assert float(smoothness(torch.zeros(RESOLUTION))) == pytest.approx(0.0, abs=1e-15)


def test_the_two_priors_are_the_old_combined_penalty() -> None:
    """The split has to be arithmetically invisible: the same weights on the same field
    must give the same number the single bundled term gave, or the prior itself moved
    and every calibration fitted with it changes.
    """
    slm_field = _SLMField(structured=True)
    phase_scale, amplitude_scale = 2e-3, 5e-4

    unit_amplitude = slm_field.amplitude.abs()
    unit_amplitude = unit_amplitude / unit_amplitude.mean()
    bundled = phase_scale * gradient_loss(slm_field.phase) + amplitude_scale * (
        gradient_loss(unit_amplitude)
    )

    split = PhaseSmoothness(slm_field, phase_scale) + AmplitudeSmoothness(
        slm_field, amplitude_scale
    )

    assert float(split()) == pytest.approx(float(bundled), rel=1e-12)
    assert float(bundled) > 0.0


def test_a_sum_reports_each_term_and_they_add_up_to_it() -> None:
    """What the convergence graph draws. The parts must be the same additions the total
    is, so a curve can never say something the total does not.
    """
    _, _, signal_mask, _ = _fixture()
    slm_field = _SLMField(structured=True)

    loss = (
        MaskedIntensityMSE(signal_mask)
        + PhaseSmoothness(slm_field)
        + AmplitudeSmoothness(slm_field)
    )
    target_intensity, _, _, field = _fixture()
    batched_field = field.unsqueeze(0).unsqueeze(0)
    batched_target = target_intensity.unsqueeze(0).unsqueeze(0)

    components = loss.components(batched_field, batched_target)

    assert list(components) == [
        "intensity mse",
        "phase smoothness",
        "amplitude smoothness",
    ]
    assert float(sum(components.values())) == pytest.approx(
        float(loss(batched_field, batched_target)), rel=1e-12
    )


def test_a_single_term_reports_itself() -> None:
    """No override needed for a term that is not a sum, so every cost can be plotted the
    same way.
    """
    target_intensity, _, signal_mask, field = _fixture()
    losses, hologram_field = _hologram_losses()

    batched_field = field.unsqueeze(0).unsqueeze(0)
    batched_target = target_intensity.unsqueeze(0).unsqueeze(0)

    assert list(
        MaskedIntensityMSE(signal_mask).components(batched_field, batched_target)
    ) == ["intensity mse"]
    # Unnamed costs fall back to the class name rather than going unlabeled.
    assert list(losses["LossVorticity"].components(hologram_field)) == ["LossVorticity"]


def test_a_repeated_term_keeps_both_curves() -> None:
    """Two terms of the same class in one sum must not collapse into one entry, or a
    curve would silently vanish from the graph and the parts would stop adding up.
    """
    slm_field = _SLMField(structured=True)

    loss = PhaseSmoothness(slm_field, 1e-3) + PhaseSmoothness(slm_field, 4e-3)
    components = loss.components(None)

    assert list(components) == ["phase smoothness", "phase smoothness (2)"]
    assert float(sum(components.values())) == pytest.approx(
        float(loss(None)), rel=1e-12
    )


@pytest.mark.parametrize("name", sorted(BASELINES))
def test_every_cost_is_retuned_through_the_same_attribute(name) -> None:
    """One weight, named the same on every term, settable after construction. Tuning a
    fit should not mean rebuilding the cost or remembering four different keywords.
    """
    losses, field = _hologram_losses()
    loss = losses[name]
    before = float(loss(field))

    loss.scale = loss.scale * 3.0

    assert float(loss(field)) == pytest.approx(3.0 * before, rel=1e-12)


def test_the_weight_is_applied_once() -> None:
    """The base multiplies by the scale, so a term's own evaluate must return the
    unweighted cost. Applying it in both places would square the weight and silently
    change every fit.
    """
    slm_field = _SLMField(structured=True)
    prior = PhaseSmoothness(slm_field, scale=1e-2)

    assert float(prior()) == pytest.approx(
        1e-2 * float(prior.evaluate()), rel=1e-12
    )


def test_retuning_a_term_moves_the_sum_and_its_parts_together() -> None:
    """What tuning a calibration actually looks like: reach into the sum, change one
    weight, and both the total and that term's curve follow.
    """
    slm_field = _SLMField(structured=True)
    loss = PhaseSmoothness(slm_field) + AmplitudeSmoothness(slm_field)
    before = loss.components()["phase smoothness"]

    loss.terms[0].scale = 10.0 * loss.terms[0].scale
    after = loss.components()

    assert float(after["phase smoothness"]) == pytest.approx(
        10.0 * float(before), rel=1e-12
    )
    assert float(sum(after.values())) == pytest.approx(float(loss()), rel=1e-12)


def test_an_unimplemented_cost_says_which_one() -> None:
    """The base is not usable on its own, and the error names the subclass."""

    class _Incomplete(LossFunction):
        pass

    with pytest.raises(NotImplementedError, match="_Incomplete"):
        _Incomplete()(torch.zeros(RESOLUTION))


# --- The divisor floor --------------------------------------------------------------


def test_smallest_divisor_follows_the_dtype() -> None:
    """The safe floor differs by hundreds of orders of magnitude between dtypes."""
    assert smallest_divisor(torch.zeros(1, dtype=torch.float32)) == pytest.approx(
        torch.finfo(torch.float32).smallest_normal
    )
    assert smallest_divisor(torch.zeros(1, dtype=torch.float64)) == pytest.approx(
        torch.finfo(torch.float64).smallest_normal
    )
    assert smallest_divisor(torch.zeros(1, dtype=torch.float64)) < smallest_divisor(
        torch.zeros(1, dtype=torch.float32)
    )


@pytest.mark.parametrize(
    "normalize, shape",
    [
        (normalize_single_to_unit_sum, (4, 4)),
        (normalize_to_unit_sum, (2, 4, 4)),
    ],
)
def test_normalizing_an_empty_image_does_not_produce_nan(normalize, shape) -> None:
    """The floor exists for this case: a frame that summed to zero must not poison the
    loss with a nan, which would take the whole fit with it.
    """
    result = normalize(torch.zeros(shape))
    assert not torch.isnan(result).any()
    assert not torch.isinf(result).any()


def test_a_genuinely_small_sum_is_not_clamped() -> None:
    """A sum the dtype represents perfectly normalizes exactly, without a floor."""
    image = torch.full((4,), 1e-30, dtype=torch.float64)
    assert float(normalize_single_to_unit_sum(image).sum()) == pytest.approx(1.0)


def test_the_floor_does_not_block_gradients() -> None:
    """These are loss functions, so anything in the path has to stay differentiable."""
    image = torch.rand(4, 4, requires_grad=True)
    normalize_single_to_unit_sum(image).sum().backward()
    assert image.grad is not None
    assert not torch.isnan(image.grad).any()


def _shaped(target, mask):
    """A field that produces exactly ``target``, with an arbitrary phase."""
    generator = torch.Generator().manual_seed(3)
    phase = torch.rand(target.shape, generator=generator, dtype=torch.float64)
    return (target * mask).sqrt() * torch.exp(1j * phase)


def test_the_absolute_cost_matches_the_normalised_one_at_the_matching_level() -> None:
    """Where the default scale comes from.

    A field carrying exactly the target's total must give the same value from the two
    costs. That is what lets the weight tuned into LossIntensityMSE carry over instead
    of being found again by hand, which matters because the optimizer stops on an
    absolute gradient tolerance.
    """
    target_intensity, _, signal_mask, field = _fixture()

    produced = (field.abs() ** 2 * signal_mask).sum()
    wanted = target_intensity * signal_mask
    levelled = wanted * (produced / wanted.sum())

    absolute = float(LossAbsoluteIntensityMSE(levelled, signal_mask)(field))
    normalised = float(LossIntensityMSE(target_intensity, signal_mask)(field))

    assert absolute == pytest.approx(normalised, rel=1e-9)


def test_losing_light_is_free_under_the_normalised_cost_but_not_this_one() -> None:
    """The whole reason this cost exists.

    A chirp-z model computes only a window of the focal plane. Dimming stands in for
    the search pushing power out of that window: the normalised cost cannot see it,
    because it scales whatever it is given back to unit sum.
    """
    target_intensity, _, signal_mask, _ = _fixture()
    target = target_intensity * signal_mask
    field = _shaped(target_intensity, signal_mask)

    absolute = LossAbsoluteIntensityMSE(target, signal_mask)
    normalised = LossIntensityMSE(target_intensity, signal_mask)

    assert float(absolute(field)) == pytest.approx(0.0, abs=1e-12)
    assert float(absolute(field * 0.5)) > 1.0
    # Half the light thrown away, and the normalised cost does not move.
    assert float(normalised(field * 0.5)) == pytest.approx(
        float(normalised(field)), abs=1e-12
    )


def test_overshooting_the_requested_level_costs_something_too() -> None:
    """The penalty is symmetric, so a target set below what is reachable is not free.

    Worth pinning: it is the difference between this cost and one that only ever asks
    for more light.
    """
    target_intensity, _, signal_mask, _ = _fixture()
    target = target_intensity * signal_mask
    absolute = LossAbsoluteIntensityMSE(target, signal_mask)
    field = _shaped(target_intensity, signal_mask)

    assert float(absolute(field * 1.5)) > 1.0


def test_the_default_scale_follows_the_target_total() -> None:
    """The scale goes as one over the total squared, which is what keeps the cost's
    magnitude fixed while the target's absolute units are free to change.
    """
    target_intensity, _, signal_mask, _ = _fixture()
    target = target_intensity * signal_mask

    one = LossAbsoluteIntensityMSE(target, signal_mask).scale
    three = LossAbsoluteIntensityMSE(target * 3, signal_mask).scale

    assert one / three == pytest.approx(9.0, rel=1e-9)


def test_an_explicit_scale_wins_over_the_derived_one() -> None:
    """The derived default is a convenience, not a lock."""
    target_intensity, _, signal_mask, _ = _fixture()

    loss = LossAbsoluteIntensityMSE(
        target_intensity * signal_mask, signal_mask, scale=7.0
    )

    assert loss.scale == 7.0
