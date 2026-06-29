"""Milestone 3 small-MLP training and analysis utilities."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import math
import statistics
import time
from typing import Any, Literal, Mapping, Sequence

import torch

from differential_sim.batched_temporal_gradients import (
    BatchedScenarioData,
    batched_rollout_objective,
    build_batched_scenarios,
    rollout_batched_scenarios_with_controller,
    split_batched_rollout,
)
from differential_sim.controllers import InputNormalization, SmallMLPHeadwayController
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


HORIZONS = (6, 10, 20, 35, 50, 80)
MODEL_SEEDS = (1000, 1001, 1002, 1003, 1004, 1005)
LR_CANDIDATES = (1e-4, 3e-4, 1e-3, 3e-3, 1e-2)
BUDGET_CANDIDATES = (200, 400, 600, 800, 1000, 1200)
DETAIL_INTERVAL = 30
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


@dataclass(frozen=True)
class TrainingConfig:
    learning_rate: float
    updates: int
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    weight_decay: float = 0.0
    amsgrad: bool = False
    detail_interval: int = DETAIL_INTERVAL
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


@dataclass
class TrainingRun:
    rows: list[dict[str, Any]]
    failed: bool
    failure_reason: str
    updates_completed: int
    runtime_s: float
    final_state: dict[str, torch.Tensor]
    snapshots: dict[int, dict[str, torch.Tensor]]


def build_small_model(
    normalization: InputNormalization,
    *,
    seed: int,
    dtype: torch.dtype = DTYPE,
    device: torch.device | str = "cpu",
) -> SmallMLPHeadwayController:
    if dtype != torch.float64:
        raise ValueError("Milestone 3 requires torch.float64")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(seed))
        model = SmallMLPHeadwayController(
            normalization,
            hidden_width=16,
            dtype=dtype,
            device=device,
        )
    return model


def clone_state_dict(
    state: Mapping[str, torch.Tensor],
    *,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = DTYPE,
) -> dict[str, torch.Tensor]:
    return {
        name: tensor.detach().clone().to(device=device, dtype=dtype)
        for name, tensor in state.items()
    }


def state_dict_hash(state: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def model_initialization_record(
    model: SmallMLPHeadwayController,
    *,
    seed: int,
) -> dict[str, Any]:
    state = model.state_dict()
    parameters = dict(model.named_parameters())
    return {
        "seed": seed,
        "state_hash": state_dict_hash(state),
        "parameter_count": sum(parameter.numel() for parameter in parameters.values()),
        "parameter_names": list(parameters),
        "parameter_shapes": {name: list(parameter.shape) for name, parameter in parameters.items()},
        "parameters": {
            name: [float(value) for value in parameter.detach().cpu().reshape(-1)]
            for name, parameter in parameters.items()
        },
    }


def flatten_parameters(model: torch.nn.Module) -> torch.Tensor:
    values = [parameter.detach().reshape(-1) for parameter in model.parameters()]
    return torch.cat(values) if values else torch.empty(0, dtype=DTYPE)


def flatten_gradients(model: torch.nn.Module) -> torch.Tensor:
    values = []
    for parameter in model.parameters():
        if parameter.grad is None:
            values.append(torch.zeros_like(parameter).reshape(-1))
        else:
            values.append(parameter.grad.detach().reshape(-1))
    return torch.cat(values) if values else torch.empty(0, dtype=DTYPE)


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
    model: SmallMLPHeadwayController,
    horizon: int,
    mode: ExecutionMode,
    dtype: torch.dtype = DTYPE,
    device: torch.device | str = "cpu",
) -> Evaluation:
    adapter = torch.empty(0, dtype=dtype, device=device)
    if mode == "scenario-batched":
        rollout = rollout_batched_scenarios_with_controller(
            split.batched,
            beta=adapter,
            controller=model,
            base_params=context.base_params,
            rollout_config=context.rollout_config,
            horizon_k=horizon,
            dtype=dtype,
            device=device,
        )
        aggregate, per_scenario = batched_rollout_objective(rollout, context.objective_config)
        rollouts = tuple(split_batched_rollout(rollout))
    else:
        rollouts = tuple(
            rollout_with_controller(
                leader=leader,
                beta=adapter,
                controller=model,
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
    model: SmallMLPHeadwayController,
    mode: ExecutionMode,
    dtype: torch.dtype = DTYPE,
    device: torch.device | str = "cpu",
) -> tuple[Evaluation, bool]:
    with torch.no_grad():
        evaluation = evaluate_split(
            context,
            split,
            model=model,
            horizon=80,
            mode=mode,
            dtype=dtype,
            device=device,
        )
        grad_enabled = torch.is_grad_enabled()
    return evaluation, grad_enabled


def run_training(
    context: ParityContext,
    train: PreparedSplit,
    held_out: PreparedSplit | None,
    *,
    initial_state: Mapping[str, torch.Tensor],
    seed: int,
    horizon: int,
    config: TrainingConfig,
    stage: str,
    snapshot_updates: Sequence[int] = (),
    include_metadata: bool = True,
) -> TrainingRun:
    _validate_training_config(config)
    if horizon not in HORIZONS:
        raise ValueError(f"Milestone 3 horizon must be one of {HORIZONS}")
    device = torch.device(config.device)
    model = build_small_model(context.normalization, seed=seed, dtype=config.dtype, device=device)
    model.load_state_dict(clone_state_dict(initial_state, device=device, dtype=config.dtype))
    initial_hash = state_dict_hash(model.state_dict())
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        betas=config.betas,
        eps=config.eps,
        weight_decay=config.weight_decay,
        amsgrad=config.amsgrad,
    )
    metadata = asdict(device_metadata(device, dtype=config.dtype)) if include_metadata else None
    rows: list[dict[str, Any]] = []
    snapshots: dict[int, dict[str, torch.Tensor]] = {}
    requested_snapshots = set(int(update) for update in snapshot_updates)
    failed = False
    failure_reason = ""
    updates_completed = 0
    prior_update_norm = 0.0
    run_start = time.perf_counter()

    for update in range(config.updates + 1):
        if update in requested_snapshots:
            snapshots[update] = clone_state_dict(model.state_dict())
        optimizer.zero_grad(set_to_none=True)
        step_start = time.perf_counter()
        training = evaluate_split(
            context,
            train,
            model=model,
            horizon=horizon,
            mode=config.execution_mode,
            dtype=config.dtype,
            device=device,
        )
        finite_loss = bool(torch.isfinite(training.aggregate.total).detach().cpu())
        finite_grad = False
        gradient_norm = math.nan
        if finite_loss:
            training.aggregate.total.backward()
            gradient = flatten_gradients(model)
            finite_grad = bool(torch.all(torch.isfinite(gradient)).detach().cpu())
            gradient_norm = float(torch.linalg.vector_norm(gradient).detach().cpu())
        step_runtime_s = time.perf_counter() - step_start

        if not finite_loss:
            failed = True
            failure_reason = "nonfinite_loss"
        elif not finite_grad:
            failed = True
            failure_reason = "nonfinite_gradient"

        scheduled_detail = update % config.detail_interval == 0 or update == config.updates
        heldout_evaluation = None
        heldout_grad_enabled = None
        if scheduled_detail and held_out is not None:
            heldout_evaluation, heldout_grad_enabled = evaluate_held_out(
                context,
                held_out,
                model=model,
                mode=config.execution_mode,
                dtype=config.dtype,
                device=device,
            )

        rows.append(
            training_row(
                context,
                training,
                heldout_evaluation,
                heldout_grad_enabled=heldout_grad_enabled,
                model=model,
                seed=seed,
                horizon=horizon,
                update=update,
                initial_state_hash=initial_hash,
                gradient_norm=gradient_norm,
                update_norm=prior_update_norm,
                finite_loss=finite_loss,
                finite_gradient=finite_grad,
                failed=failed,
                failure_reason=failure_reason,
                step_runtime_s=step_runtime_s,
                cumulative_runtime_s=time.perf_counter() - run_start,
                config=config,
                stage=stage,
                metadata=metadata,
                include_detail=scheduled_detail,
            )
        )
        if failed or update == config.updates:
            break

        before = flatten_parameters(model).clone()
        optimizer.step()
        after = flatten_parameters(model)
        prior_update_norm = float(torch.linalg.vector_norm(after - before).detach().cpu())
        updates_completed += 1

    return TrainingRun(
        rows=rows,
        failed=failed,
        failure_reason=failure_reason,
        updates_completed=updates_completed,
        runtime_s=time.perf_counter() - run_start,
        final_state=clone_state_dict(model.state_dict()),
        snapshots=snapshots,
    )


def training_row(
    context: ParityContext,
    training: Evaluation,
    held_out: Evaluation | None,
    *,
    heldout_grad_enabled: bool | None,
    model: SmallMLPHeadwayController,
    seed: int,
    horizon: int,
    update: int,
    initial_state_hash: str,
    gradient_norm: float,
    update_norm: float,
    finite_loss: bool,
    finite_gradient: bool,
    failed: bool,
    failure_reason: str,
    step_runtime_s: float,
    cumulative_runtime_s: float,
    config: TrainingConfig,
    stage: str,
    metadata: Mapping[str, Any] | None,
    include_detail: bool,
) -> dict[str, Any]:
    train_min_gap, train_min_speed = min_gap_and_speed(training.rollouts)
    train_headway_min, train_headway_max = headway_range(model, training.rollouts)
    row: dict[str, Any] = {
        "milestone": "3_small_model_training",
        "stage": stage,
        "run_id": f"seed{seed}_K{horizon}",
        "model_seed": seed,
        "initial_state_hash": initial_state_hash,
        "K": horizon,
        "horizon_label": "T" if horizon == 80 else str(horizon),
        "update": update,
        "architecture": "3-16-tanh-1-bounded-sigmoid",
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "optimizer": "Adam",
        "learning_rate": config.learning_rate,
        "adam_betas": list(config.betas),
        "adam_eps": config.eps,
        "weight_decay": config.weight_decay,
        "amsgrad": config.amsgrad,
        "updates_requested": config.updates,
        "train_components": component_floats(training.aggregate),
        "train_weighted_components": weighted_component_floats(training.aggregate),
        "gradient_norm": gradient_norm,
        "parameter_update_norm": update_norm,
        "parameter_norm": float(torch.linalg.vector_norm(flatten_parameters(model)).detach().cpu()),
        "finite_loss": finite_loss,
        "finite_gradient": finite_gradient,
        "train_min_gap": train_min_gap,
        "train_min_speed": train_min_speed,
        "train_headway_min": train_headway_min,
        "train_headway_max": train_headway_max,
        "step_runtime_s": step_runtime_s,
        "cumulative_runtime_s": cumulative_runtime_s,
        "failed": failed,
        "failure_reason": failure_reason,
        "requested_device": config.device,
        "actual_device": str(next(model.parameters()).device),
        "dtype": str(next(model.parameters()).dtype),
        "execution_mode": config.execution_mode,
        "train_scenario_count": len(training.scenario_names),
    }
    if stage == "execution_check":
        row["flat_parameters"] = [
            float(value) for value in flatten_parameters(model).detach().cpu()
        ]
        row["flat_gradients"] = [
            float(value) for value in flatten_gradients(model).detach().cpu()
        ]
    if metadata is not None:
        row["environment"] = dict(metadata)
    if include_detail:
        row["train_per_scenario"] = per_scenario_rows(training)
    if held_out is not None:
        heldout_min_gap, heldout_min_speed = min_gap_and_speed(held_out.rollouts)
        heldout_headway_min, heldout_headway_max = headway_range(model, held_out.rollouts)
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
    return [
        {
            "scenario_name": name,
            **{
                field: float(getattr(evaluation.per_scenario, field)[index].detach().cpu())
                for field in COMPONENT_FIELDS
            },
        }
        for index, name in enumerate(evaluation.scenario_names)
    ]


def headway_range(
    model: SmallMLPHeadwayController,
    rollouts: Sequence[RolloutResult],
) -> tuple[float, float]:
    values = []
    adapter = torch.empty(0, dtype=next(model.parameters()).dtype, device=next(model.parameters()).device)
    for result in rollouts:
        inputs = torch.stack(
            [result.follower_v[:-1], result.delta_v[:-1], result.gap[:-1]],
            dim=-1,
        )
        values.append(model(adapter, inputs))
    combined = torch.cat([value.reshape(-1) for value in values])
    return float(torch.min(combined).detach().cpu()), float(torch.max(combined).detach().cpu())


def relative_change(final: float, initial: float, *, epsilon: float = 1e-12) -> float:
    return (final - initial) / max(abs(initial), epsilon)


def summarize_run(run: TrainingRun) -> dict[str, Any]:
    first = run.rows[0]
    final = run.rows[-1]
    summary: dict[str, Any] = {
        "run_id": first["run_id"],
        "model_seed": first["model_seed"],
        "initial_state_hash": first["initial_state_hash"],
        "K": first["K"],
        "horizon_label": first["horizon_label"],
        "learning_rate": first["learning_rate"],
        "updates_requested": first["updates_requested"],
        "updates_completed": run.updates_completed,
        "failed": run.failed,
        "failure_reason": run.failure_reason,
        "runtime_s": run.runtime_s,
        "initial_train_total": float(first["train_components"]["total"]),
        "final_train_total": float(final["train_components"]["total"]),
        "final_state_hash": state_dict_hash(run.final_state),
        "final_train_headway_min": final["train_headway_min"],
        "final_train_headway_max": final["train_headway_max"],
    }
    summary["final_train_relative_change"] = relative_change(
        summary["final_train_total"],
        summary["initial_train_total"],
    )
    if "heldout_components" in first and "heldout_components" in final:
        summary.update(
            {
                "initial_heldout_total": float(first["heldout_components"]["total"]),
                "final_heldout_total": float(final["heldout_components"]["total"]),
                "final_heldout_headway_min": final["heldout_headway_min"],
                "final_heldout_headway_max": final["heldout_headway_max"],
            }
        )
        summary["final_heldout_relative_change"] = relative_change(
            summary["final_heldout_total"],
            summary["initial_heldout_total"],
        )
    summary["component_changes"] = {
        name: float(final["train_components"][name]) - float(first["train_components"][name])
        for name in ("progress", "safety", "jerk")
    }
    summary["weighted_component_changes"] = {
        name: float(final["train_weighted_components"][name])
        - float(first["train_weighted_components"][name])
        for name in ("progress", "safety", "jerk")
    }
    summary.update(convergence_metrics(run.rows, failed=run.failed))
    return summary


def convergence_metrics(rows: Sequence[dict[str, Any]], *, failed: bool) -> dict[str, Any]:
    if failed or len(rows) < 2:
        return {
            "normalized_training_auc": None,
            "first_update_50pct": None,
            "first_update_90pct": None,
            "max_positive_rebound_after_90pct": None,
            "tail_std_final_10pct": None,
            "tail_mean_abs_relative_change_100": None,
        }
    updates = [int(row["update"]) for row in rows]
    values = [float(row["train_components"]["total"]) for row in rows]
    initial, final = values[0], values[-1]
    normalized = [value / max(abs(initial), 1e-12) for value in values]
    auc = sum(
        (updates[index] - updates[index - 1]) * (normalized[index] + normalized[index - 1]) / 2.0
        for index in range(1, len(rows))
    ) / max(updates[-1], 1)
    achieved = initial - final
    update_50 = _first_threshold(updates, values, initial, achieved, 0.5)
    update_90 = _first_threshold(updates, values, initial, achieved, 0.9)
    rebound = None
    if update_90 is not None:
        index_90 = updates.index(update_90)
        rebound = max(0.0, max(values[index_90:]) - values[index_90])
    tail_start = math.ceil(0.9 * updates[-1])
    tail_values = [value for update, value in zip(updates, values, strict=True) if update >= tail_start]
    last_100 = [
        (previous, current)
        for previous, current, update in zip(values[:-1], values[1:], updates[1:], strict=True)
        if update > updates[-1] - 100
    ]
    mean_abs_relative = (
        statistics.fmean(abs(current - previous) / max(abs(previous), 1e-12) for previous, current in last_100)
        if last_100
        else None
    )
    return {
        "normalized_training_auc": auc,
        "first_update_50pct": update_50,
        "first_update_90pct": update_90,
        "max_positive_rebound_after_90pct": rebound,
        "tail_std_final_10pct": statistics.pstdev(tail_values) if tail_values else None,
        "tail_mean_abs_relative_change_100": mean_abs_relative,
    }


def select_shared_learning_rate(
    candidate_summaries: Sequence[dict[str, Any]],
) -> tuple[float | None, list[dict[str, Any]]]:
    eligible = [
        row
        for row in candidate_summaries
        if int(row["finite_run_count"]) == int(row["run_count"])
        and all(float(item["median_relative_training_change"]) < 0.0 for item in row["by_horizon"])
    ]
    if not eligible:
        return None, []
    ordered = sorted(eligible, key=lambda row: float(row["score"]))
    selected = ordered[0]
    comparisons = []
    for candidate in ordered[1:]:
        best_score = float(selected["score"])
        score = float(candidate["score"])
        near_tie = abs(score - best_score) <= 0.02 * abs(best_score)
        comparisons.append(
            {
                "selected_lr_before_tie": float(selected["learning_rate"]),
                "candidate_lr": float(candidate["learning_rate"]),
                "selected_score": best_score,
                "candidate_score": score,
                "near_tie": near_tie,
            }
        )
        if near_tie and float(candidate["learning_rate"]) < float(selected["learning_rate"]):
            selected = candidate
    return float(selected["learning_rate"]), comparisons


def select_update_budget(
    run_rows: Sequence[dict[str, Any]],
    *,
    candidates: Sequence[int] = BUDGET_CANDIDATES,
) -> tuple[int | None, list[dict[str, Any]]]:
    assessments = []
    for budget in candidates:
        by_horizon = []
        qualifies = True
        for horizon in HORIZONS:
            runs = _rows_by_run(run_rows, horizon=horizon)
            remaining_fractions = []
            stable_changes = []
            finite_count = 0
            seed_qualified = 0
            for rows in runs.values():
                row_budget = _row_at(rows, budget)
                row_final = _row_at(rows, candidates[-1])
                initial = float(rows[0]["train_components"]["total"])
                current = float(row_budget["train_components"]["total"])
                final = float(row_final["train_components"]["total"])
                achieved = initial - final
                remaining = current - final
                fraction = remaining / max(abs(achieved), 1e-12)
                remaining_fractions.append(fraction)
                if fraction <= 0.05:
                    seed_qualified += 1
                tail = [
                    float(row["train_components"]["total"])
                    for row in rows
                    if budget - 100 <= int(row["update"]) <= budget
                ]
                changes = [
                    abs(current_value - previous) / max(abs(previous), 1e-12)
                    for previous, current_value in zip(tail[:-1], tail[1:], strict=True)
                ]
                stable_changes.append(statistics.fmean(changes) if changes else math.inf)
                if all(
                    bool(row["finite_loss"]) and bool(row["finite_gradient"]) and not bool(row["failed"])
                    for row in rows
                    if int(row["update"]) <= budget
                ):
                    finite_count += 1
            median_remaining = statistics.median(remaining_fractions)
            median_stability = statistics.median(stable_changes)
            horizon_qualifies = (
                finite_count == len(MODEL_SEEDS)
                and median_remaining <= 0.02
                and seed_qualified >= 5
                and median_stability <= 1e-4
            )
            qualifies = qualifies and horizon_qualifies
            by_horizon.append(
                {
                    "K": horizon,
                    "finite_run_count": finite_count,
                    "median_remaining_fraction": median_remaining,
                    "seed_remaining_fraction_count": seed_qualified,
                    "median_tail_mean_abs_relative_change": median_stability,
                    "qualifies": horizon_qualifies,
                }
            )
        assessments.append({"budget": budget, "qualifies": qualifies, "by_horizon": by_horizon})
        if qualifies:
            return budget, assessments
    return None, assessments


def aggregate_horizons(run_summaries: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    aggregates = []
    for horizon in HORIZONS:
        rows = [row for row in run_summaries if int(row["K"]) == horizon]
        values = [
            float(row["final_heldout_relative_change"])
            for row in rows
            if not row["failed"] and row.get("final_heldout_relative_change") is not None
        ]
        aggregates.append(
            {
                "K": horizon,
                "run_count": len(rows),
                "failure_count": sum(bool(row["failed"]) for row in rows),
                "finite_heldout_count": len(values),
                "median_final_heldout_relative_change": statistics.median(values) if values else None,
                "mean_final_heldout_relative_change": statistics.fmean(values) if values else None,
                "min_final_heldout_relative_change": min(values) if values else None,
                "max_final_heldout_relative_change": max(values) if values else None,
                "iqr_final_heldout_relative_change": _iqr(values) if values else None,
                "secondary": _aggregate_secondary(rows),
            }
        )
    return aggregates


def classify_h2(
    run_summaries: Sequence[dict[str, Any]],
    aggregates: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    by_horizon = {int(row["K"]): row for row in aggregates}
    valid_truncated = [
        horizon
        for horizon in (6, 10, 20, 35, 50)
        if by_horizon[horizon]["median_final_heldout_relative_change"] is not None
    ]
    if not valid_truncated:
        return {"classification": "unresolved", "reason": "no_finite_truncated_result"}
    best_truncated = min(
        valid_truncated,
        key=lambda horizon: float(by_horizon[horizon]["median_final_heldout_relative_change"]),
    )
    paired = []
    for seed in MODEL_SEEDS:
        full = next(
            (
                row
                for row in run_summaries
                if int(row["K"]) == 80 and int(row["model_seed"]) == seed and not row["failed"]
            ),
            None,
        )
        truncated = next(
            (
                row
                for row in run_summaries
                if int(row["K"]) == best_truncated
                and int(row["model_seed"]) == seed
                and not row["failed"]
            ),
            None,
        )
        if full is not None and truncated is not None:
            paired.append(
                float(full["final_heldout_relative_change"])
                - float(truncated["final_heldout_relative_change"])
            )
    failures_ok = all(int(row["failure_count"]) <= 1 for row in aggregates)
    if len(paired) < 5 or not failures_ok:
        return {
            "classification": "unresolved",
            "reason": "insufficient_pairs_or_failures",
            "best_truncated_horizon": best_truncated,
            "paired_differences": paired,
        }
    median_difference = statistics.median(paired)
    full_wins = sum(value < -0.01 for value in paired)
    truncated_wins = sum(value > 0.01 for value in paired)
    full_median = float(by_horizon[80]["median_final_heldout_relative_change"])
    truncated_median = float(by_horizon[best_truncated]["median_final_heldout_relative_change"])
    ranges_overlap = max(
        float(by_horizon[80]["min_final_heldout_relative_change"]),
        float(by_horizon[best_truncated]["min_final_heldout_relative_change"]),
    ) <= min(
        float(by_horizon[80]["max_final_heldout_relative_change"]),
        float(by_horizon[best_truncated]["max_final_heldout_relative_change"]),
    )
    if full_median < 0.0 and (
        (median_difference <= -0.01 and full_wins >= 4)
        or (
            abs(median_difference) <= 0.01
            and ranges_overlap
            and full_wins <= 4
            and truncated_wins <= 4
        )
    ):
        classification = "supported"
        reason = "full_better_or_operationally_tied"
    elif (
        median_difference >= 0.01
        and truncated_wins >= 4
        or (full_median >= 0.0 and truncated_median < 0.0)
        or all(
            row["median_final_heldout_relative_change"] is not None
            and float(row["median_final_heldout_relative_change"]) >= 0.0
            for row in aggregates
        )
    ):
        classification = "contradicted"
        reason = "best_truncated_better_or_no_transfer"
    else:
        classification = "unresolved"
        reason = "paired_evidence_ambiguous"
    return {
        "classification": classification,
        "reason": reason,
        "best_truncated_horizon": best_truncated,
        "paired_differences": paired,
        "median_paired_difference": median_difference,
        "full_win_count": full_wins,
        "truncated_win_count": truncated_wins,
        "ranges_overlap": ranges_overlap,
    }


def gradient_field_diagnostics(
    context: ParityContext,
    train: PreparedSplit,
    *,
    seed: int,
    reference_update: int,
    reference_state: Mapping[str, torch.Tensor],
) -> list[dict[str, Any]]:
    gradients: dict[int, torch.Tensor] = {}
    runtimes: dict[int, float] = {}
    state_hash = state_dict_hash(reference_state)
    for horizon in HORIZONS:
        model = build_small_model(context.normalization, seed=seed)
        model.load_state_dict(clone_state_dict(reference_state))
        model.zero_grad(set_to_none=True)
        start = time.perf_counter()
        evaluation = evaluate_split(
            context,
            train,
            model=model,
            horizon=horizon,
            mode="scenario-batched",
        )
        evaluation.aggregate.total.backward()
        runtimes[horizon] = time.perf_counter() - start
        gradients[horizon] = flatten_gradients(model).cpu()
    rows = []
    for left in HORIZONS:
        for right in HORIZONS:
            left_norm = float(torch.linalg.vector_norm(gradients[left]))
            right_norm = float(torch.linalg.vector_norm(gradients[right]))
            cosine = None
            if left_norm > 0.0 and right_norm > 0.0:
                cosine = float(torch.dot(gradients[left], gradients[right]) / (left_norm * right_norm))
            rows.append(
                {
                    "milestone": "3_small_model_training",
                    "stage": "gradient_field_diagnostics",
                    "model_seed": seed,
                    "reference_horizon": 80,
                    "reference_update": reference_update,
                    "reference_state_hash": state_hash,
                    "left_K": left,
                    "right_K": right,
                    "left_gradient_norm": left_norm,
                    "right_gradient_norm": right_norm,
                    "cosine": cosine,
                    "left_runtime_s": runtimes[left],
                    "right_runtime_s": runtimes[right],
                }
            )
    return rows


def compare_training_rows(
    left: dict[str, Any],
    right: dict[str, Any],
    *,
    scalar_atol: float = 1e-8,
    scalar_rtol: float = 1e-7,
    vector_atol: float = 1e-7,
    vector_rtol: float = 1e-6,
) -> dict[str, Any]:
    scalar_abs = 0.0
    scalar_rel = 0.0
    for scope in ("train_components", "heldout_components"):
        for field in ("total", "progress", "safety", "jerk"):
            a, b = float(left[scope][field]), float(right[scope][field])
            diff = abs(a - b)
            scalar_abs = max(scalar_abs, diff)
            scalar_rel = max(scalar_rel, diff / max(abs(a), abs(b), 1e-12))
    vector_fields = ("flat_parameters", "flat_gradients")
    vector_abs = 0.0
    vector_rel = 0.0
    for field in vector_fields:
        a = torch.tensor(left[field], dtype=DTYPE)
        b = torch.tensor(right[field], dtype=DTYPE)
        diff = float(torch.max(torch.abs(a - b)))
        denom = max(float(torch.max(torch.abs(a))), float(torch.max(torch.abs(b))), 1e-12)
        vector_abs = max(vector_abs, diff)
        vector_rel = max(vector_rel, diff / denom)
    left_improved = float(left["train_components"]["total"]) < float(left["initial_train_total"])
    right_improved = float(right["train_components"]["total"]) < float(right["initial_train_total"])
    flags_match = (
        bool(left["failed"]) == bool(right["failed"])
        and left_improved == right_improved
        and left["heldout_grad_enabled"] is False
        and right["heldout_grad_enabled"] is False
    )
    return {
        "scalar_max_abs_diff": scalar_abs,
        "scalar_max_rel_diff": scalar_rel,
        "vector_max_abs_diff": vector_abs,
        "vector_max_rel_diff": vector_rel,
        "flags_match": flags_match,
        "passed": (
            (scalar_abs <= scalar_atol or scalar_rel <= scalar_rtol)
            and (vector_abs <= vector_atol or vector_rel <= vector_rtol)
            and flags_match
        ),
    }


def _validate_training_config(config: TrainingConfig) -> None:
    if config.learning_rate <= 0.0:
        raise ValueError("learning_rate must be positive")
    if config.updates < 0:
        raise ValueError("updates must be nonnegative")
    if config.detail_interval != DETAIL_INTERVAL:
        raise ValueError(f"Milestone 3 requires detail_interval={DETAIL_INTERVAL}")
    if config.device != "cpu":
        raise ValueError("Milestone 3 main training requires CPU")
    if config.dtype != torch.float64:
        raise ValueError("Milestone 3 requires torch.float64")
    if config.execution_mode not in ("scenario-batched", "unbatched"):
        raise ValueError("unsupported execution mode")
    if config.betas != (0.9, 0.999) or config.eps != 1e-8:
        raise ValueError("Milestone 3 requires approved Adam parameters")
    if config.weight_decay != 0.0 or config.amsgrad:
        raise ValueError("Milestone 3 prohibits weight decay and AMSGrad")


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


def _rows_by_run(
    rows: Sequence[dict[str, Any]],
    *,
    horizon: int,
) -> dict[str, list[dict[str, Any]]]:
    run_ids = sorted({str(row["run_id"]) for row in rows if int(row["K"]) == horizon})
    return {
        run_id: sorted(
            [row for row in rows if row["run_id"] == run_id],
            key=lambda row: int(row["update"]),
        )
        for run_id in run_ids
    }


def _row_at(rows: Sequence[dict[str, Any]], update: int) -> dict[str, Any]:
    return next(row for row in rows if int(row["update"]) == update)


def _iqr(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0
    ordered = sorted(values)
    lower = statistics.median(ordered[: len(ordered) // 2])
    upper = statistics.median(ordered[(len(ordered) + 1) // 2 :])
    return upper - lower


def _aggregate_secondary(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    result = {}
    for field in (
        "final_train_relative_change",
        "normalized_training_auc",
        "first_update_50pct",
        "first_update_90pct",
        "max_positive_rebound_after_90pct",
        "tail_std_final_10pct",
        "tail_mean_abs_relative_change_100",
        "runtime_s",
    ):
        values = [float(row[field]) for row in rows if row.get(field) is not None]
        result[field] = {
            "count": len(values),
            "median": statistics.median(values) if values else None,
            "mean": statistics.fmean(values) if values else None,
            "min": min(values) if values else None,
            "max": max(values) if values else None,
        }
    return result
