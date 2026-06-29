import math

import pytest
import torch

from differential_sim.device_parity import build_parity_context, max_result_difference
from differential_sim.small_model_training import (
    DETAIL_INTERVAL,
    HORIZONS,
    TrainingConfig,
    aggregate_horizons,
    build_small_model,
    classify_h2,
    clone_state_dict,
    evaluate_held_out,
    evaluate_split,
    flatten_gradients,
    gradient_field_diagnostics,
    model_initialization_record,
    prepare_split,
    run_training,
    state_dict_hash,
)


def small_context():
    return build_parity_context(
        train_limit=2,
        held_out_limit=1,
        init_limit=1,
    )


def test_small_model_architecture_bounds_and_parameter_count():
    context = small_context()
    model = build_small_model(context.normalization, seed=1000)
    inputs = torch.tensor(
        [[0.0, 0.0, 0.0], [100.0, -100.0, 200.0], [-100.0, 100.0, -200.0]],
        dtype=torch.float64,
    )

    headway = model(torch.empty(0, dtype=torch.float64), inputs)

    assert model.hidden.in_features == 3
    assert model.hidden.out_features == 16
    assert model.output.in_features == 16
    assert sum(parameter.numel() for parameter in model.parameters()) == 81
    assert headway.shape == (3,)
    assert torch.all(headway >= 0.5)
    assert torch.all(headway <= 3.0)


def test_initialization_is_deterministic_distinct_and_hashable():
    context = small_context()
    first = build_small_model(context.normalization, seed=1000)
    second = build_small_model(context.normalization, seed=1000)
    other = build_small_model(context.normalization, seed=1001)

    assert state_dict_hash(first.state_dict()) == state_dict_hash(second.state_dict())
    assert state_dict_hash(first.state_dict()) != state_dict_hash(other.state_dict())
    record = model_initialization_record(first, seed=1000)
    assert record["parameter_count"] == 81
    assert record["state_hash"] == state_dict_hash(first.state_dict())


def test_forward_values_identical_across_all_horizons_and_gradients_differ():
    context = small_context()
    split = prepare_split(context.train)
    initial = build_small_model(context.normalization, seed=1000).state_dict()
    results = {}
    gradients = {}
    for horizon in HORIZONS:
        model = build_small_model(context.normalization, seed=1000)
        model.load_state_dict(initial)
        evaluation = evaluate_split(
            context,
            split,
            model=model,
            horizon=horizon,
            mode="scenario-batched",
        )
        evaluation.aggregate.total.backward()
        results[horizon] = evaluation
        gradients[horizon] = flatten_gradients(model)

    for horizon in HORIZONS[:-1]:
        assert max_result_difference(results[horizon].rollouts, results[80].rollouts) == 0.0
        assert torch.equal(
            results[horizon].aggregate.total.detach(),
            results[80].aggregate.total.detach(),
        )
    assert torch.linalg.vector_norm(gradients[6]) > 0.0
    assert not torch.allclose(gradients[6], gradients[80], atol=1e-10, rtol=1e-8)


def test_full_gradient_matches_finite_difference_direction():
    context = build_parity_context(train_limit=1, held_out_limit=1, init_limit=1)
    split = prepare_split(context.train)
    model = build_small_model(context.normalization, seed=1000)
    parameters = list(model.parameters())
    direction = [torch.ones_like(parameter) for parameter in parameters]
    norm = torch.sqrt(sum(torch.sum(torch.square(value)) for value in direction))
    direction = [value / norm for value in direction]

    evaluation = evaluate_split(
        context,
        split,
        model=model,
        horizon=80,
        mode="scenario-batched",
    )
    evaluation.aggregate.total.backward()
    autograd_dd = sum(
        torch.sum(parameter.grad * value)
        for parameter, value in zip(parameters, direction, strict=True)
    )
    base_state = clone_state_dict(model.state_dict())
    errors = []
    for epsilon in (1e-4, 3e-5, 1e-5):
        values = []
        for sign in (1.0, -1.0):
            candidate = build_small_model(context.normalization, seed=1000)
            candidate.load_state_dict(base_state)
            with torch.no_grad():
                for parameter, value in zip(candidate.parameters(), direction, strict=True):
                    parameter.add_(sign * epsilon * value)
            objective = evaluate_split(
                context,
                split,
                model=candidate,
                horizon=80,
                mode="scenario-batched",
            ).aggregate.total
            values.append(objective)
        finite_difference = (values[0] - values[1]) / (2.0 * epsilon)
        absolute = torch.abs(finite_difference - autograd_dd)
        relative = absolute / torch.clamp(torch.abs(autograd_dd), min=1e-12)
        errors.append((float(absolute), float(relative)))
    assert any(relative < 1e-3 or absolute < 1e-6 for absolute, relative in errors)


def test_training_pairing_update_order_determinism_and_heldout_isolation():
    context = small_context()
    train = prepare_split(context.train)
    held_out = prepare_split(context.held_out)
    initial = clone_state_dict(build_small_model(context.normalization, seed=1000).state_dict())
    kwargs = {
        "context": context,
        "train": train,
        "held_out": held_out,
        "initial_state": initial,
        "seed": 1000,
        "horizon": 6,
        "config": TrainingConfig(learning_rate=1e-3, updates=2),
        "stage": "test",
    }

    first = run_training(**kwargs)
    second = run_training(**kwargs)

    assert first.failed is False
    assert first.updates_completed == 2
    assert [row["update"] for row in first.rows] == [0, 1, 2]
    assert first.rows[0]["parameter_update_norm"] == 0.0
    assert first.rows[1]["parameter_update_norm"] > 0.0
    assert first.rows[0]["initial_state_hash"] == state_dict_hash(initial)
    assert first.rows[-1]["train_components"] == second.rows[-1]["train_components"]
    assert state_dict_hash(first.final_state) == state_dict_hash(second.final_state)
    assert all(
        row["heldout_grad_enabled"] is False
        for row in first.rows
        if "heldout_grad_enabled" in row
    )


def test_policy_validation_and_nonfinite_failure():
    context = small_context()
    train = prepare_split(context.train)
    initial = clone_state_dict(build_small_model(context.normalization, seed=1000).state_dict())
    common = {
        "context": context,
        "train": train,
        "held_out": None,
        "initial_state": initial,
        "seed": 1000,
        "horizon": 6,
        "stage": "test",
    }
    with pytest.raises(ValueError, match="detail_interval"):
        run_training(
            **common,
            config=TrainingConfig(learning_rate=1e-3, updates=1, detail_interval=10),
        )
    with pytest.raises(ValueError, match="CPU"):
        run_training(
            **common,
            config=TrainingConfig(learning_rate=1e-3, updates=1, device="cuda"),
        )

    bad = clone_state_dict(initial)
    bad["hidden.weight"][0, 0] = math.nan
    failed = run_training(
        **{**common, "initial_state": bad},
        config=TrainingConfig(learning_rate=1e-3, updates=5),
    )
    assert failed.failed is True
    assert failed.failure_reason == "nonfinite_loss"
    assert failed.updates_completed == 0
    assert len(failed.rows) == 1


def test_batched_unbatched_objective_and_gradient_equivalence():
    context = small_context()
    split = prepare_split(context.train)
    state = clone_state_dict(build_small_model(context.normalization, seed=1000).state_dict())
    evaluations = {}
    gradients = {}
    for mode in ("unbatched", "scenario-batched"):
        model = build_small_model(context.normalization, seed=1000)
        model.load_state_dict(state)
        evaluation = evaluate_split(context, split, model=model, horizon=35, mode=mode)
        evaluation.aggregate.total.backward()
        evaluations[mode] = evaluation
        gradients[mode] = flatten_gradients(model)

    assert max_result_difference(
        evaluations["unbatched"].rollouts,
        evaluations["scenario-batched"].rollouts,
    ) <= 1e-10
    assert torch.allclose(
        evaluations["unbatched"].aggregate.total,
        evaluations["scenario-batched"].aggregate.total,
        atol=1e-10,
        rtol=1e-9,
    )
    assert torch.allclose(
        gradients["unbatched"],
        gradients["scenario-batched"],
        atol=1e-10,
        rtol=1e-9,
    )


def test_heldout_helper_is_no_grad():
    context = small_context()
    held_out = prepare_split(context.held_out)
    model = build_small_model(context.normalization, seed=1000)

    evaluation, grad_enabled = evaluate_held_out(
        context,
        held_out,
        model=model,
        mode="scenario-batched",
    )

    assert grad_enabled is False
    assert evaluation.aggregate.total.requires_grad is False


def test_gradient_field_diagnostic_is_shared_state_and_nonmutating():
    context = build_parity_context(train_limit=1, held_out_limit=1, init_limit=1)
    train = prepare_split(context.train)
    state = clone_state_dict(build_small_model(context.normalization, seed=1000).state_dict())
    before = state_dict_hash(state)

    rows = gradient_field_diagnostics(
        context,
        train,
        seed=1000,
        reference_update=0,
        reference_state=state,
    )

    assert len(rows) == len(HORIZONS) * len(HORIZONS)
    assert state_dict_hash(state) == before
    assert {row["reference_state_hash"] for row in rows} == {before}
    assert all(row["cosine"] is not None for row in rows)


def test_h2_classification_uses_best_truncated_horizon():
    run_summaries = []
    values_by_horizon = {
        6: [-0.10, -0.11, -0.12, -0.10, -0.09, -0.11],
        10: [-0.20, -0.21, -0.22, -0.20, -0.19, -0.21],
        20: [-0.25, -0.26, -0.27, -0.25, -0.24, -0.26],
        35: [-0.30, -0.31, -0.32, -0.30, -0.29, -0.31],
        50: [-0.28, -0.29, -0.30, -0.28, -0.27, -0.29],
        80: [-0.45, -0.46, -0.47, -0.45, -0.44, -0.46],
    }
    for horizon, values in values_by_horizon.items():
        for index, value in enumerate(values):
            run_summaries.append(
                {
                    "K": horizon,
                    "model_seed": 1000 + index,
                    "failed": False,
                    "final_heldout_relative_change": value,
                    "final_train_relative_change": value,
                    "normalized_training_auc": 1.0,
                    "first_update_50pct": 1,
                    "first_update_90pct": 2,
                    "max_positive_rebound_after_90pct": 0.0,
                    "tail_std_final_10pct": 0.0,
                    "tail_mean_abs_relative_change_100": 0.0,
                    "runtime_s": 1.0,
                }
            )
    aggregates = aggregate_horizons(run_summaries)
    result = classify_h2(run_summaries, aggregates)

    assert result["best_truncated_horizon"] == 35
    assert result["classification"] == "supported"
    assert result["full_win_count"] == 6
