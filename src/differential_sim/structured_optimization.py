"""Milestone 2 D.2 iterative structured-controller optimization."""

from __future__ import annotations

from dataclasses import dataclass
import math
import statistics
import time
from typing import Any, Literal, Sequence

import torch

from differential_sim.batched_temporal_gradients import (
    BatchedScenarioData,
    batched_rollout_objective,
    build_batched_scenarios,
    rollout_batched_scenarios_with_controller,
)
from differential_sim.controllers import StructuredHeadwayController, tensor_to_list
from differential_sim.device_parity import DTYPE, ParityContext, device_metadata
from differential_sim.milestone1_diagnostics import min_gap_and_speed
from differential_sim.objectives import (
    ObjectiveComponents,
    component_floats,
    mean_scenario_objective,
    rollout_objective,
    weighted_component_floats,
)
from differential_sim.rollout import RolloutResult
from differential_sim.scenarios import LeaderProfile, leader_profile
from differential_sim.temporal_gradients import rollout_with_controller


ExecutionMode = Literal["scenario-batched", "unbatched"]
COMPONENT_FIELDS = (
    "total",
    "progress",
    "safety",
    "jerk",
    "weighted_progress",
    "weighted_safety",
    "weighted_jerk",
)
LR_CANDIDATES = (0.003, 0.01, 0.03, 0.1)


@dataclass(frozen=True)
class OptimizationConfig:
    learning_rate: float
    updates: int = 500
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    weight_decay: float = 0.0
    amsgrad: bool = False
    detail_interval: int = 10
    device: str = "cpu"
    dtype: torch.dtype = DTYPE
    execution_mode: ExecutionMode = "scenario-batched"


@dataclass(frozen=True)
class PreparedSplit:
    names: tuple[str, ...]
    leaders: tuple[LeaderProfile, ...]
    batched: BatchedScenarioData


@dataclass(frozen=True)
class Evaluation:
    aggregate: ObjectiveComponents
    per_scenario: ObjectiveComponents
    rollouts: tuple[RolloutResult, ...]
    scenario_names: tuple[str, ...]


@dataclass(frozen=True)
class OptimizationRun:
    rows: list[dict[str, Any]]
    failed: bool
    failure_reason: str
    updates_completed: int
    runtime_s: float


def prepare_split(
    scenarios,
    *,
    dtype: torch.dtype = DTYPE,
    device: torch.device | str = "cpu",
) -> PreparedSplit:
    return PreparedSplit(
        names=tuple(name for name, _ in scenarios),
        leaders=tuple(leader_profile(config, dtype=dtype, device=device) for _, config in scenarios),
        batched=build_batched_scenarios(scenarios, dtype=dtype, device=device),
    )


def evaluate_split(
    context: ParityContext,
    split: PreparedSplit,
    *,
    beta: torch.Tensor,
    horizon: int,
    mode: ExecutionMode,
    dtype: torch.dtype = DTYPE,
    device: torch.device | str = "cpu",
) -> Evaluation:
    if mode == "scenario-batched":
        rollout = rollout_batched_scenarios_with_controller(
            split.batched,
            beta=beta,
            controller=context.controller,
            base_params=context.base_params,
            rollout_config=context.rollout_config,
            horizon_k=horizon,
            dtype=dtype,
            device=device,
        )
        aggregate, per_scenario = batched_rollout_objective(rollout, context.objective_config)
        rollouts = tuple(_split_rollout(rollout))
    else:
        rollouts = tuple(
            rollout_with_controller(
                leader=leader,
                beta=beta,
                controller=context.controller,
                base_params=context.base_params,
                rollout_config=context.rollout_config,
                horizon_k=horizon,
                dtype=dtype,
                device=device,
            )
            for leader in split.leaders
        )
        individual = [rollout_objective(result, context.objective_config) for result in rollouts]
        aggregate = mean_scenario_objective(rollouts, context.objective_config)
        per_scenario = ObjectiveComponents(
            **{
                field: torch.stack([getattr(component, field) for component in individual])
                for field in COMPONENT_FIELDS
            }
        )
    return Evaluation(
        aggregate=aggregate,
        per_scenario=per_scenario,
        rollouts=rollouts,
        scenario_names=split.names,
    )


def evaluate_held_out(
    context: ParityContext,
    split: PreparedSplit,
    *,
    beta: torch.Tensor,
    mode: ExecutionMode,
    dtype: torch.dtype = DTYPE,
    device: torch.device | str = "cpu",
) -> tuple[Evaluation, bool]:
    with torch.no_grad():
        evaluation = evaluate_split(
            context,
            split,
            beta=beta.detach(),
            horizon=80,
            mode=mode,
            dtype=dtype,
            device=device,
        )
        grad_enabled = torch.is_grad_enabled()
    return evaluation, grad_enabled


def run_optimization(
    context: ParityContext,
    train: PreparedSplit,
    held_out: PreparedSplit | None,
    *,
    beta_initial: torch.Tensor,
    horizon: int,
    initialization_id: int,
    t_init: float,
    seed: int,
    config: OptimizationConfig,
    stage: str,
    include_metadata: bool = True,
) -> OptimizationRun:
    if config.learning_rate <= 0.0:
        raise ValueError("learning_rate must be positive")
    if config.updates < 0:
        raise ValueError("updates must be nonnegative")
    if config.detail_interval < 1:
        raise ValueError("detail_interval must be positive")
    if config.device != "cpu":
        raise ValueError("Milestone 2 D.2 is approved for CPU execution only")
    if config.dtype != torch.float64:
        raise ValueError("Milestone 2 D.2 requires torch.float64")
    if config.betas != (0.9, 0.999):
        raise ValueError("Milestone 2 D.2 requires Adam betas=(0.9, 0.999)")
    if config.eps != 1e-8:
        raise ValueError("Milestone 2 D.2 requires Adam eps=1e-8")
    if config.weight_decay != 0.0:
        raise ValueError("Milestone 2 D.2 requires Adam weight_decay=0.0")
    if config.amsgrad:
        raise ValueError("Milestone 2 D.2 requires Adam amsgrad=False")

    device = torch.device(config.device)
    beta = beta_initial.detach().clone().to(device=device, dtype=config.dtype).requires_grad_(True)
    optimizer = torch.optim.Adam(
        [beta],
        lr=config.learning_rate,
        betas=config.betas,
        eps=config.eps,
        weight_decay=config.weight_decay,
        amsgrad=config.amsgrad,
    )
    metadata = device_metadata(device, dtype=config.dtype) if include_metadata else None
    rows: list[dict[str, Any]] = []
    failed = False
    failure_reason = ""
    updates_completed = 0
    run_start = time.perf_counter()

    for update in range(config.updates + 1):
        optimizer.zero_grad(set_to_none=True)
        step_start = time.perf_counter()
        training = evaluate_split(
            context,
            train,
            beta=beta,
            horizon=horizon,
            mode=config.execution_mode,
            dtype=config.dtype,
            device=device,
        )
        finite_loss = bool(torch.isfinite(training.aggregate.total).detach().cpu())
        grad_norm = math.nan
        finite_grad = False
        if finite_loss:
            training.aggregate.total.backward()
            finite_grad = beta.grad is not None and bool(torch.all(torch.isfinite(beta.grad)).detach().cpu())
            if beta.grad is not None:
                grad_norm = float(torch.linalg.vector_norm(beta.grad).detach().cpu())
        step_runtime_s = time.perf_counter() - step_start

        scheduled_detail = update % config.detail_interval == 0 or update == config.updates
        heldout_evaluation = None
        heldout_grad_enabled = None
        if scheduled_detail and held_out is not None:
            heldout_evaluation, heldout_grad_enabled = evaluate_held_out(
                context,
                held_out,
                beta=beta,
                mode=config.execution_mode,
                dtype=config.dtype,
                device=device,
            )

        if not finite_loss:
            failed = True
            failure_reason = "nonfinite_loss"
        elif not finite_grad:
            failed = True
            failure_reason = "nonfinite_grad"

        rows.append(
            optimization_row(
                context,
                training,
                heldout_evaluation,
                heldout_grad_enabled=heldout_grad_enabled,
                beta=beta,
                beta_initial=beta_initial,
                horizon=horizon,
                initialization_id=initialization_id,
                t_init=t_init,
                seed=seed,
                update=update,
                grad_norm=grad_norm,
                finite_loss=finite_loss,
                finite_grad=finite_grad,
                failed=failed,
                failure_reason=failure_reason,
                step_runtime_s=step_runtime_s,
                config=config,
                stage=stage,
                metadata=metadata,
                include_detail=scheduled_detail,
            )
        )
        if failed or update == config.updates:
            break
        optimizer.step()
        updates_completed += 1

    return OptimizationRun(
        rows=rows,
        failed=failed,
        failure_reason=failure_reason,
        updates_completed=updates_completed,
        runtime_s=time.perf_counter() - run_start,
    )


def optimization_row(
    context: ParityContext,
    training: Evaluation,
    held_out: Evaluation | None,
    *,
    heldout_grad_enabled: bool | None,
    beta: torch.Tensor,
    beta_initial: torch.Tensor,
    horizon: int,
    initialization_id: int,
    t_init: float,
    seed: int,
    update: int,
    grad_norm: float,
    finite_loss: bool,
    finite_grad: bool,
    failed: bool,
    failure_reason: str,
    step_runtime_s: float,
    config: OptimizationConfig,
    stage: str,
    metadata: Any,
    include_detail: bool,
) -> dict[str, Any]:
    train_min_gap, train_min_speed = min_gap_and_speed(training.rollouts)
    train_headway_min, train_headway_max = headway_range(
        context.controller,
        beta,
        training.rollouts,
    )
    row: dict[str, Any] = {
        "milestone": "2_d2_iterative_structured_optimization",
        "stage": stage,
        "run_id": f"K{horizon}_init{initialization_id}",
        "seed": seed,
        "initialization_id": initialization_id,
        "T_init": t_init,
        "K": horizon,
        "horizon_label": "T" if horizon == 80 else str(horizon),
        "update": update,
        "initial_beta": tensor_to_list(beta_initial),
        "beta": tensor_to_list(beta),
        "optimizer": "Adam",
        "learning_rate": config.learning_rate,
        "adam_betas": list(config.betas),
        "adam_eps": config.eps,
        "weight_decay": config.weight_decay,
        "amsgrad": config.amsgrad,
        "updates_requested": config.updates,
        "train_components": component_floats(training.aggregate),
        "train_weighted_components": weighted_component_floats(training.aggregate),
        "gradient_norm": grad_norm,
        "finite_loss": finite_loss,
        "finite_gradient": finite_grad,
        "train_min_gap": train_min_gap,
        "train_min_speed": train_min_speed,
        "train_headway_min": train_headway_min,
        "train_headway_max": train_headway_max,
        "step_runtime_s": step_runtime_s,
        "failed": failed,
        "failure_reason": failure_reason,
        "requested_device": config.device,
        "actual_device": str(beta.device),
        "dtype": str(beta.dtype),
        "execution_mode": config.execution_mode,
        "train_scenario_count": len(training.scenario_names),
    }
    if metadata is not None:
        row["environment"] = metadata
    if include_detail:
        row["train_per_scenario"] = per_scenario_rows(training)
    if held_out is not None:
        heldout_min_gap, heldout_min_speed = min_gap_and_speed(held_out.rollouts)
        heldout_headway_min, heldout_headway_max = headway_range(
            context.controller,
            beta,
            held_out.rollouts,
        )
        row.update(
            {
                "heldout_components": component_floats(held_out.aggregate),
                "heldout_weighted_components": weighted_component_floats(held_out.aggregate),
                "heldout_per_scenario": per_scenario_rows(held_out),
                "heldout_min_gap": heldout_min_gap,
                "heldout_min_speed": heldout_min_speed,
                "heldout_headway_min": heldout_headway_min,
                "heldout_headway_max": heldout_headway_max,
                "heldout_grad_enabled": heldout_grad_enabled,
                "heldout_scenario_count": len(held_out.scenario_names),
            }
        )
    return row


def per_scenario_rows(evaluation: Evaluation) -> list[dict[str, Any]]:
    rows = []
    for index, name in enumerate(evaluation.scenario_names):
        rows.append(
            {
                "scenario_name": name,
                "total": _item(evaluation.per_scenario.total, index),
                "progress": _item(evaluation.per_scenario.progress, index),
                "safety": _item(evaluation.per_scenario.safety, index),
                "jerk": _item(evaluation.per_scenario.jerk, index),
                "weighted_progress": _item(evaluation.per_scenario.weighted_progress, index),
                "weighted_safety": _item(evaluation.per_scenario.weighted_safety, index),
                "weighted_jerk": _item(evaluation.per_scenario.weighted_jerk, index),
            }
        )
    return rows


def headway_range(
    controller: StructuredHeadwayController,
    beta: torch.Tensor,
    rollouts: Sequence[RolloutResult],
) -> tuple[float, float]:
    values = []
    for result in rollouts:
        inputs = torch.stack(
            [result.follower_v[:-1], result.delta_v[:-1], result.gap[:-1]],
            dim=-1,
        )
        values.append(controller(beta, inputs))
    combined = torch.cat([value.reshape(-1) for value in values])
    return (
        float(torch.min(combined).detach().cpu()),
        float(torch.max(combined).detach().cpu()),
    )


def relative_change(final: float, initial: float, *, epsilon: float = 1e-12) -> float:
    return (final - initial) / max(abs(initial), epsilon)


def select_shared_learning_rate(
    summaries: Sequence[dict[str, Any]],
) -> tuple[float | None, list[dict[str, Any]]]:
    eligible = sorted(
        (
            row
            for row in summaries
            if int(row["finite_run_count"]) == int(row["run_count"])
            and float(row["median_relative_training_change"]) < 0.0
        ),
        key=lambda row: float(row["learning_rate"]),
    )
    if not eligible:
        return None, []
    selected_index = len(eligible) - 1
    comparisons: list[dict[str, Any]] = []
    while selected_index > 0:
        larger = eligible[selected_index]
        smaller = eligible[selected_index - 1]
        larger_median = float(larger["median_relative_training_change"])
        smaller_median = float(smaller["median_relative_training_change"])
        best = min(larger_median, smaller_median)
        near_tie = abs(larger_median - smaller_median) <= 0.02 * abs(best)
        comparisons.append(
            {
                "larger_lr": float(larger["learning_rate"]),
                "smaller_lr": float(smaller["learning_rate"]),
                "larger_median": larger_median,
                "smaller_median": smaller_median,
                "near_tie": near_tie,
            }
        )
        if not near_tie:
            break
        selected_index -= 1
    return float(eligible[selected_index]["learning_rate"]), comparisons


def summarize_run(run: OptimizationRun) -> dict[str, Any]:
    first = run.rows[0]
    final = run.rows[-1]
    initial_train = float(first["train_components"]["total"])
    final_train = float(final["train_components"]["total"])
    summary = {
        "run_id": first["run_id"],
        "K": first["K"],
        "horizon_label": first["horizon_label"],
        "initialization_id": first["initialization_id"],
        "T_init": first["T_init"],
        "seed": first["seed"],
        "learning_rate": first["learning_rate"],
        "failed": run.failed,
        "failure_reason": run.failure_reason,
        "updates_completed": run.updates_completed,
        "runtime_s": run.runtime_s,
        "initial_train_total": initial_train,
        "final_train_total": final_train,
        "final_train_relative_change": relative_change(final_train, initial_train),
        "final_beta": final["beta"],
        "final_train_headway_min": final["train_headway_min"],
        "final_train_headway_max": final["train_headway_max"],
    }
    if "heldout_components" in first and "heldout_components" in final:
        initial_heldout = float(first["heldout_components"]["total"])
        final_heldout = float(final["heldout_components"]["total"])
        summary.update(
            {
                "initial_heldout_total": initial_heldout,
                "final_heldout_total": final_heldout,
                "final_heldout_relative_change": relative_change(final_heldout, initial_heldout),
                "final_heldout_headway_min": final["heldout_headway_min"],
                "final_heldout_headway_max": final["heldout_headway_max"],
            }
        )
    summary.update(convergence_metrics(run.rows, failed=run.failed))
    summary["component_changes"] = {
        name: float(final["train_components"][name]) - float(first["train_components"][name])
        for name in ("progress", "safety", "jerk")
    }
    return summary


def convergence_metrics(rows: Sequence[dict[str, Any]], *, failed: bool) -> dict[str, Any]:
    if failed or len(rows) < 2:
        return {
            "normalized_training_auc": None,
            "first_update_50pct": None,
            "first_update_90pct": None,
            "max_positive_rebound_after_90pct": None,
            "tail_std_updates_451_500": None,
        }
    updates = [int(row["update"]) for row in rows]
    values = [float(row["train_components"]["total"]) for row in rows]
    initial = values[0]
    final = values[-1]
    normalized = [value / max(abs(initial), 1e-12) for value in values]
    auc = sum(
        (updates[index] - updates[index - 1]) * (normalized[index] + normalized[index - 1]) / 2.0
        for index in range(1, len(rows))
    ) / max(updates[-1] - updates[0], 1)
    achieved = initial - final
    update_50 = _first_threshold(updates, values, initial, achieved, 0.5)
    update_90 = _first_threshold(updates, values, initial, achieved, 0.9)
    rebound = None
    if update_90 is not None:
        index_90 = updates.index(update_90)
        rebound = max(0.0, max(values[index_90:]) - values[index_90])
    tail = [value for update, value in zip(updates, values, strict=True) if 451 <= update <= 500]
    return {
        "normalized_training_auc": auc,
        "first_update_50pct": update_50,
        "first_update_90pct": update_90,
        "max_positive_rebound_after_90pct": rebound,
        "tail_std_updates_451_500": statistics.pstdev(tail) if tail else None,
    }


def aggregate_horizons(run_summaries: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    aggregates = []
    for horizon in sorted({int(row["K"]) for row in run_summaries}):
        rows = [row for row in run_summaries if int(row["K"]) == horizon]
        heldout = [
            float(row["final_heldout_relative_change"])
            for row in rows
            if not row["failed"] and "final_heldout_relative_change" in row
        ]
        aggregates.append(
            {
                "K": horizon,
                "horizon_label": "T" if horizon == 80 else str(horizon),
                "run_count": len(rows),
                "failure_count": sum(bool(row["failed"]) for row in rows),
                "finite_heldout_count": len(heldout),
                "median_final_heldout_relative_change": statistics.median(heldout) if heldout else None,
                "mean_final_heldout_relative_change": statistics.fmean(heldout) if heldout else None,
                "min_final_heldout_relative_change": min(heldout) if heldout else None,
                "max_final_heldout_relative_change": max(heldout) if heldout else None,
                "iqr_final_heldout_relative_change": _iqr(heldout) if heldout else None,
                "secondary": _aggregate_secondary(rows),
            }
        )
    return aggregates


def rank_horizons(aggregates: Sequence[dict[str, Any]]) -> list[int]:
    valid = [
        row for row in aggregates if row["median_final_heldout_relative_change"] is not None
    ]
    return [
        int(row["K"])
        for row in sorted(valid, key=lambda row: float(row["median_final_heldout_relative_change"]))
    ]


def operational_ties(aggregates: Sequence[dict[str, Any]]) -> list[list[int]]:
    ties = []
    for index, left in enumerate(aggregates):
        if left["median_final_heldout_relative_change"] is None:
            continue
        for right in aggregates[index + 1 :]:
            if right["median_final_heldout_relative_change"] is None:
                continue
            median_close = abs(
                float(left["median_final_heldout_relative_change"])
                - float(right["median_final_heldout_relative_change"])
            ) <= 0.01
            ranges_overlap = max(
                float(left["min_final_heldout_relative_change"]),
                float(right["min_final_heldout_relative_change"]),
            ) <= min(
                float(left["max_final_heldout_relative_change"]),
                float(right["max_final_heldout_relative_change"]),
            )
            if median_close and ranges_overlap:
                ties.append([int(left["K"]), int(right["K"])])
    return ties


def spearman_against_milestone1(d2_ranking: Sequence[int]) -> float | None:
    milestone1 = [80, 10, 6, 3, 1]
    if set(d2_ranking) != set(milestone1):
        return None
    ranks_a = {horizon: index + 1 for index, horizon in enumerate(milestone1)}
    ranks_b = {horizon: index + 1 for index, horizon in enumerate(d2_ranking)}
    squared = sum((ranks_a[horizon] - ranks_b[horizon]) ** 2 for horizon in milestone1)
    n = len(milestone1)
    return 1.0 - 6.0 * squared / (n * (n * n - 1))


def classify_h1(d2_ranking: Sequence[int], ties: Sequence[Sequence[int]]) -> str:
    if len(d2_ranking) != 5:
        return "unresolved"
    rho = spearman_against_milestone1(d2_ranking)
    top_overlap = len(set(d2_ranking[:3]) & {80, 10, 6})
    full_tied_with_best = any(80 in pair and d2_ranking[0] in pair for pair in ties)
    if rho is not None and rho >= 0.8 and top_overlap == 3 and (
        d2_ranking[0] == 80 or full_tied_with_best
    ):
        return "supported"
    if rho is not None and rho <= 0.0:
        return "contradicted_or_weakened"
    return "unresolved"


def compare_rows(
    left: dict[str, Any],
    right: dict[str, Any],
    *,
    scalar_atol: float = 1e-8,
    scalar_rtol: float = 1e-7,
    beta_atol: float = 1e-7,
    beta_rtol: float = 1e-6,
) -> dict[str, Any]:
    scalar_diffs = []
    scalar_rel_diffs = []
    for scope in ("train_components", "heldout_components"):
        for field in ("total", "progress", "safety", "jerk"):
            a = float(left[scope][field])
            b = float(right[scope][field])
            scalar_diffs.append(abs(a - b))
            scalar_rel_diffs.append(abs(a - b) / max(abs(a), abs(b), 1e-12))
    beta_left = torch.tensor(left["beta"], dtype=torch.float64)
    beta_right = torch.tensor(right["beta"], dtype=torch.float64)
    beta_abs = float(torch.max(torch.abs(beta_left - beta_right)))
    beta_rel = beta_abs / max(
        float(torch.max(torch.abs(beta_left))),
        float(torch.max(torch.abs(beta_right))),
        1e-12,
    )
    scalar_abs = max(scalar_diffs)
    scalar_rel = max(scalar_rel_diffs)
    flags_match = (
        bool(left["failed"]) == bool(right["failed"])
        and (float(left["train_components"]["total"]) < float(left["initial_train_total"]))
        == (float(right["train_components"]["total"]) < float(right["initial_train_total"]))
    )
    return {
        "scalar_max_abs_diff": scalar_abs,
        "scalar_max_rel_diff": scalar_rel,
        "beta_max_abs_diff": beta_abs,
        "beta_max_rel_diff": beta_rel,
        "flags_match": flags_match,
        "passed": (
            (scalar_abs <= scalar_atol or scalar_rel <= scalar_rtol)
            and (beta_abs <= beta_atol or beta_rel <= beta_rtol)
            and flags_match
        ),
    }


def _split_rollout(result: RolloutResult) -> list[RolloutResult]:
    return [
        RolloutResult(
            leader_x=result.leader_x[index],
            leader_v=result.leader_v[index],
            follower_x=result.follower_x[index],
            follower_v=result.follower_v[index],
            follower_a=result.follower_a[index],
            gap=result.gap[index],
            delta_v=result.delta_v[index],
        )
        for index in range(result.follower_v.shape[0])
    ]


def _item(tensor: torch.Tensor, index: int) -> float:
    return float(tensor[index].detach().cpu())


def _first_threshold(
    updates: Sequence[int],
    values: Sequence[float],
    initial: float,
    achieved: float,
    fraction: float,
) -> int | None:
    if achieved <= 0.0:
        return None
    target = initial - fraction * achieved
    return next((update for update, value in zip(updates, values, strict=True) if value <= target), None)


def _iqr(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0
    ordered = sorted(values)
    lower = statistics.median(ordered[: len(ordered) // 2])
    upper = statistics.median(ordered[(len(ordered) + 1) // 2 :])
    return upper - lower


def _aggregate_secondary(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    fields = (
        "final_train_relative_change",
        "normalized_training_auc",
        "first_update_50pct",
        "first_update_90pct",
        "max_positive_rebound_after_90pct",
        "tail_std_updates_451_500",
        "runtime_s",
    )
    result = {}
    for field in fields:
        values = [float(row[field]) for row in rows if row.get(field) is not None]
        result[field] = {
            "count": len(values),
            "median": statistics.median(values) if values else None,
            "mean": statistics.fmean(values) if values else None,
            "min": min(values) if values else None,
            "max": max(values) if values else None,
        }
    return result
