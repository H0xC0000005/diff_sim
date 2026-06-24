import argparse
import importlib.util
from pathlib import Path
import sys

import torch

from differential_sim.batched_temporal_gradients import (
    build_batched_scenarios,
    objective_for_batched_scenarios,
    split_batched_rollout,
)
from differential_sim.device_parity import build_parity_context, max_result_difference
from differential_sim.objectives import rollout_objective
from differential_sim.temporal_gradients import objective_for_scenarios


def test_batched_scenario_shapes_and_order():
    context = build_parity_context(train_limit=2, held_out_limit=1, horizon_limit=1, init_limit=1)

    scenarios = build_batched_scenarios(context.train)

    assert scenarios.scenario_names == tuple(name for name, _ in context.train)
    assert scenarios.leader_x.shape == (2, 81)
    assert scenarios.leader_v.shape == (2, 81)


def test_batched_and_unbatched_forward_objective_and_gradient_match_all_horizons():
    context = build_parity_context(train_limit=2, held_out_limit=1, init_limit=1)
    beta_initial = context.init_betas_cpu[0]

    for horizon in context.horizons:
        beta_unbatched = beta_initial.detach().clone().requires_grad_(True)
        unbatched_components, unbatched_rollouts = objective_for_scenarios(
            context.train,
            beta=beta_unbatched,
            controller=context.controller,
            base_params=context.base_params,
            rollout_config=context.rollout_config,
            horizon_k=horizon,
            objective_config=context.objective_config,
        )
        grad_unbatched = torch.autograd.grad(unbatched_components.total, beta_unbatched)[0]

        beta_batched = beta_initial.detach().clone().requires_grad_(True)
        batched = objective_for_batched_scenarios(
            context.train,
            beta=beta_batched,
            controller=context.controller,
            base_params=context.base_params,
            rollout_config=context.rollout_config,
            horizon_k=horizon,
            objective_config=context.objective_config,
        )
        grad_batched = torch.autograd.grad(batched.aggregate.total, beta_batched)[0]

        assert max_result_difference(
            [item.result for item in unbatched_rollouts],
            split_batched_rollout(batched.rollout),
        ) <= 1e-10
        assert torch.allclose(batched.aggregate.total, unbatched_components.total, atol=1e-10, rtol=1e-9)
        assert torch.allclose(grad_batched, grad_unbatched, atol=1e-10, rtol=1e-9)


def test_batched_per_scenario_components_preserve_equal_weight_mean():
    context = build_parity_context(train_limit=2, held_out_limit=1, horizon_limit=1, init_limit=1)
    batched = objective_for_batched_scenarios(
        context.train,
        beta=context.init_betas_cpu[0],
        controller=context.controller,
        base_params=context.base_params,
        rollout_config=context.rollout_config,
        horizon_k=context.horizons[0],
        objective_config=context.objective_config,
    )
    split_results = split_batched_rollout(batched.rollout)
    individual = [rollout_objective(result, context.objective_config) for result in split_results]

    for field in ("total", "progress", "safety", "jerk"):
        expected = torch.stack([getattr(item, field) for item in individual])
        actual = getattr(batched.per_scenario, field)
        assert torch.allclose(actual, expected, atol=1e-12, rtol=1e-11)
        assert torch.allclose(getattr(batched.aggregate, field), expected.mean(), atol=1e-12, rtol=1e-11)


def test_batched_heldout_evaluation_runs_under_no_grad():
    context = build_parity_context(train_limit=1, held_out_limit=1, horizon_limit=1, init_limit=1)

    with torch.no_grad():
        result = objective_for_batched_scenarios(
            context.held_out,
            beta=context.init_betas_cpu[0],
            controller=context.controller,
            base_params=context.base_params,
            rollout_config=context.rollout_config,
            horizon_k=80,
            objective_config=context.objective_config,
        )
        grad_enabled = torch.is_grad_enabled()

    assert grad_enabled is False
    assert result.aggregate.total.requires_grad is False
    assert torch.isfinite(result.aggregate.total)


def test_d1b_smoke_runner_writes_required_schema(tmp_path):
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "check_milestone2_gpu_batching.py"
    spec = importlib.util.spec_from_file_location("check_milestone2_gpu_batching", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    args = argparse.Namespace(
        execution_mode="compare",
        device="cpu",
        probe_lr=0.03,
        probe_updates=1,
        warmup_repeats=0,
        timed_repeats=1,
        output_dir=tmp_path,
        smoke=True,
    )

    summary = module.run(args)

    assert summary["passed"] is True
    assert summary["parity_row_count"] == 2
    assert summary["timing_sample_count"] == 4
    assert summary["d2_policy_selected"] is False
    assert (tmp_path / "parity.jsonl").is_file()
    assert (tmp_path / "timing.jsonl").is_file()
    assert (tmp_path / "summary.json").is_file()
    assert (tmp_path / "summary.md").is_file()
