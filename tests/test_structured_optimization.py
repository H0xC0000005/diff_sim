import math

import pytest
import torch

from differential_sim.device_parity import build_parity_context
from differential_sim.structured_optimization import (
    OptimizationConfig,
    aggregate_horizons,
    convergence_metrics,
    evaluate_held_out,
    operational_ties,
    prepare_split,
    rank_horizons,
    run_optimization,
    select_shared_learning_rate,
    summarize_run,
)


def small_context():
    return build_parity_context(
        train_limit=2,
        held_out_limit=1,
        horizon_limit=1,
        init_limit=1,
    )


def test_run_optimization_update_order_and_schema():
    context = small_context()
    train = prepare_split(context.train)
    held_out = prepare_split(context.held_out)
    run = run_optimization(
        context,
        train,
        held_out,
        beta_initial=context.init_betas_cpu[0],
        horizon=1,
        initialization_id=0,
        t_init=0.9,
        seed=0,
        config=OptimizationConfig(learning_rate=0.01, updates=2, detail_interval=1),
        stage="test",
    )

    assert run.failed is False
    assert run.updates_completed == 2
    assert [row["update"] for row in run.rows] == [0, 1, 2]
    assert run.rows[0]["beta"] == run.rows[0]["initial_beta"]
    assert run.rows[1]["beta"] != run.rows[0]["beta"]
    assert all(row["optimizer"] == "Adam" for row in run.rows)
    assert all(row["execution_mode"] == "scenario-batched" for row in run.rows)
    assert all(row["heldout_grad_enabled"] is False for row in run.rows)
    assert len(run.rows[-1]["train_per_scenario"]) == 2
    assert len(run.rows[-1]["heldout_per_scenario"]) == 1


def test_optimizer_state_is_independent_and_deterministic():
    context = small_context()
    train = prepare_split(context.train)
    config = OptimizationConfig(learning_rate=0.01, updates=3)
    kwargs = {
        "context": context,
        "train": train,
        "held_out": None,
        "beta_initial": context.init_betas_cpu[0],
        "horizon": 1,
        "initialization_id": 0,
        "t_init": 0.9,
        "seed": 0,
        "config": config,
        "stage": "test",
    }
    first = run_optimization(**kwargs)
    second = run_optimization(**kwargs)

    assert first.rows[-1]["beta"] == second.rows[-1]["beta"]
    assert first.rows[-1]["train_components"] == second.rows[-1]["train_components"]


def test_shared_policy_enforces_cpu_float64():
    context = small_context()
    train = prepare_split(context.train)
    common = {
        "context": context,
        "train": train,
        "held_out": None,
        "beta_initial": context.init_betas_cpu[0],
        "horizon": 1,
        "initialization_id": 0,
        "t_init": 0.9,
        "seed": 0,
        "stage": "test",
    }
    with pytest.raises(ValueError, match="CPU"):
        run_optimization(**common, config=OptimizationConfig(learning_rate=0.01, device="cuda"))
    with pytest.raises(ValueError, match="float64"):
        run_optimization(
            **common,
            config=OptimizationConfig(learning_rate=0.01, dtype=torch.float32),
        )
    with pytest.raises(ValueError, match="betas"):
        run_optimization(
            **common,
            config=OptimizationConfig(learning_rate=0.01, betas=(0.8, 0.999)),
        )
    with pytest.raises(ValueError, match="weight_decay"):
        run_optimization(
            **common,
            config=OptimizationConfig(learning_rate=0.01, weight_decay=0.01),
        )


def test_heldout_evaluation_is_no_grad():
    context = small_context()
    held_out = prepare_split(context.held_out)
    evaluation, grad_enabled = evaluate_held_out(
        context,
        held_out,
        beta=context.init_betas_cpu[0].requires_grad_(True),
        mode="scenario-batched",
    )

    assert grad_enabled is False
    assert evaluation.aggregate.total.requires_grad is False


def test_nonfinite_initialization_is_reported_without_retry():
    context = small_context()
    train = prepare_split(context.train)
    beta = context.init_betas_cpu[0].clone()
    beta[0] = math.nan
    run = run_optimization(
        context,
        train,
        None,
        beta_initial=beta,
        horizon=1,
        initialization_id=0,
        t_init=0.9,
        seed=0,
        config=OptimizationConfig(learning_rate=0.01, updates=5),
        stage="test",
    )

    assert run.failed is True
    assert run.failure_reason == "nonfinite_loss"
    assert run.updates_completed == 0
    assert len(run.rows) == 1


def test_lr_selection_uses_eligibility_and_adjacent_near_tie_rule():
    rows = [
        {
            "learning_rate": 0.003,
            "run_count": 30,
            "finite_run_count": 30,
            "median_relative_training_change": -0.20,
        },
        {
            "learning_rate": 0.01,
            "run_count": 30,
            "finite_run_count": 30,
            "median_relative_training_change": -0.30,
        },
        {
            "learning_rate": 0.03,
            "run_count": 30,
            "finite_run_count": 30,
            "median_relative_training_change": -0.301,
        },
        {
            "learning_rate": 0.1,
            "run_count": 30,
            "finite_run_count": 29,
            "median_relative_training_change": -0.40,
        },
    ]

    selected, comparisons = select_shared_learning_rate(rows)

    assert selected == 0.01
    assert comparisons[0]["near_tie"] is True


def test_convergence_metrics_follow_fixed_definitions():
    rows = [
        {"update": update, "train_components": {"total": value}}
        for update, value in ((0, 10.0), (1, 7.0), (2, 5.0), (3, 5.5), (4, 4.0))
    ]
    metrics = convergence_metrics(rows, failed=False)

    assert metrics["normalized_training_auc"] == pytest.approx(0.6125)
    assert metrics["first_update_50pct"] == 1
    assert metrics["first_update_90pct"] == 4
    assert metrics["max_positive_rebound_after_90pct"] == 0.0
    assert metrics["tail_std_updates_451_500"] is None


def test_horizon_aggregation_ranking_and_operational_tie():
    run_summaries = []
    for horizon, values in ((1, (-0.19, -0.202)), (3, (-0.205, -0.195)), (80, (-0.4, -0.5))):
        for init_id, value in enumerate(values):
            run_summaries.append(
                {
                    "K": horizon,
                    "failed": False,
                    "final_heldout_relative_change": value,
                    "final_train_relative_change": value,
                    "normalized_training_auc": 1.0,
                    "first_update_50pct": 1,
                    "first_update_90pct": 2,
                    "max_positive_rebound_after_90pct": 0.0,
                    "tail_std_updates_451_500": 0.0,
                    "runtime_s": 1.0,
                }
            )
    aggregates = aggregate_horizons(run_summaries)

    assert rank_horizons(aggregates) == [80, 3, 1]
    assert [1, 3] in operational_ties(aggregates)


def test_summarize_run_keeps_failed_run_in_schema():
    context = small_context()
    train = prepare_split(context.train)
    run = run_optimization(
        context,
        train,
        None,
        beta_initial=context.init_betas_cpu[0],
        horizon=1,
        initialization_id=0,
        t_init=0.9,
        seed=0,
        config=OptimizationConfig(learning_rate=0.01, updates=1),
        stage="test",
    )
    summary = summarize_run(run)

    assert summary["run_id"] == "K1_init0"
    assert summary["updates_completed"] == 1
    assert summary["failed"] is False
