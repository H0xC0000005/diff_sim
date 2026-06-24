"""Run Milestone 2 D.1b scenario-batching correctness and cost checks."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
import json
import math
from pathlib import Path
import statistics
import sys
import time
from typing import Any, Literal, Sequence

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from differential_sim.batched_temporal_gradients import (
    BatchedScenarioData,
    batched_rollout_objective,
    build_batched_scenarios,
    rollout_batched_scenarios_with_controller,
    split_batched_rollout,
)
from differential_sim.device_parity import (
    DTYPE,
    ParityContext,
    ParityTolerance,
    build_parity_context,
    device_metadata,
    json_safe,
    max_result_difference,
    peak_memory_if_cuda,
    reset_peak_memory_if_cuda,
    scalar_abs_rel,
    synchronize_if_cuda,
    tolerance_passed,
    vector_abs_rel,
    write_json,
    write_jsonl,
)
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


ExecutionMode = Literal["unbatched", "scenario-batched"]
DEFAULT_OUTPUT_DIR = ROOT / "reports" / "milestone2" / "infrastructure" / "batching"
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
class PreparedSplit:
    names: tuple[str, ...]
    leaders: tuple[LeaderProfile, ...]
    batched: BatchedScenarioData


@dataclass(frozen=True)
class ExecutionSnapshot:
    device: str
    mode: ExecutionMode
    horizon: int
    initialization_id: int
    update: int
    beta: torch.Tensor
    aggregate: ObjectiveComponents
    per_scenario: ObjectiveComponents
    rollouts: tuple[RolloutResult, ...]
    grad: torch.Tensor
    heldout_aggregate: ObjectiveComponents
    heldout_per_scenario: ObjectiveComponents
    heldout_rollouts: tuple[RolloutResult, ...]
    heldout_grad_enabled: bool
    improved: bool = False
    failed: bool = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--execution-mode",
        choices=("unbatched", "scenario-batched", "compare"),
        default="compare",
    )
    parser.add_argument("--device", default="cpu,cuda")
    parser.add_argument("--probe-lr", type=float, default=0.03)
    parser.add_argument("--probe-updates", type=int, default=10)
    parser.add_argument("--warmup-repeats", type=int, default=3)
    parser.add_argument("--timed-repeats", type=int, default=10)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def prepare_split(
    configs,
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> PreparedSplit:
    names = tuple(name for name, _ in configs)
    leaders = tuple(leader_profile(config, dtype=dtype, device=device) for _, config in configs)
    batched = build_batched_scenarios(configs, dtype=dtype, device=device)
    return PreparedSplit(names=names, leaders=leaders, batched=batched)


def aggregate_per_scenario(components: Sequence[ObjectiveComponents]) -> ObjectiveComponents:
    return ObjectiveComponents(
        **{
            field: torch.stack([getattr(component, field) for component in components])
            for field in COMPONENT_FIELDS
        }
    )


def evaluate_prepared(
    context: ParityContext,
    split: PreparedSplit,
    *,
    mode: ExecutionMode,
    beta: torch.Tensor,
    horizon: int,
    device: torch.device,
) -> tuple[ObjectiveComponents, ObjectiveComponents, tuple[RolloutResult, ...]]:
    if mode == "unbatched":
        rollouts = tuple(
            rollout_with_controller(
                leader=leader,
                beta=beta,
                controller=context.controller,
                base_params=context.base_params,
                rollout_config=context.rollout_config,
                horizon_k=horizon,
                dtype=DTYPE,
                device=device,
            )
            for leader in split.leaders
        )
        individual = [rollout_objective(result, context.objective_config) for result in rollouts]
        return mean_scenario_objective(rollouts, context.objective_config), aggregate_per_scenario(individual), rollouts
    rollout = rollout_batched_scenarios_with_controller(
        split.batched,
        beta=beta,
        controller=context.controller,
        base_params=context.base_params,
        rollout_config=context.rollout_config,
        horizon_k=horizon,
        dtype=DTYPE,
        device=device,
    )
    aggregate, per_scenario = batched_rollout_objective(rollout, context.objective_config)
    return aggregate, per_scenario, tuple(split_batched_rollout(rollout))


def make_snapshot(
    context: ParityContext,
    train: PreparedSplit,
    held_out: PreparedSplit,
    *,
    mode: ExecutionMode,
    beta: torch.Tensor,
    horizon: int,
    initialization_id: int,
    update: int,
    device: torch.device,
) -> ExecutionSnapshot:
    beta_work = beta.detach().clone().to(device=device, dtype=DTYPE).requires_grad_(True)
    aggregate, per_scenario, rollouts = evaluate_prepared(
        context,
        train,
        mode=mode,
        beta=beta_work,
        horizon=horizon,
        device=device,
    )
    grad = torch.autograd.grad(aggregate.total, beta_work)[0]
    with torch.no_grad():
        heldout_aggregate, heldout_per_scenario, heldout_rollouts = evaluate_prepared(
            context,
            held_out,
            mode=mode,
            beta=beta_work.detach(),
            horizon=80,
            device=device,
        )
        heldout_grad_enabled = torch.is_grad_enabled()
    return ExecutionSnapshot(
        device=str(device),
        mode=mode,
        horizon=horizon,
        initialization_id=initialization_id,
        update=update,
        beta=beta_work.detach(),
        aggregate=aggregate,
        per_scenario=per_scenario,
        rollouts=rollouts,
        grad=grad.detach(),
        heldout_aggregate=heldout_aggregate,
        heldout_per_scenario=heldout_per_scenario,
        heldout_rollouts=heldout_rollouts,
        heldout_grad_enabled=heldout_grad_enabled,
    )


def max_component_difference(
    left: ObjectiveComponents,
    right: ObjectiveComponents,
) -> tuple[float, float]:
    max_abs = 0.0
    max_rel = 0.0
    for field in COMPONENT_FIELDS:
        abs_diff, rel_diff = vector_abs_rel(getattr(left, field), getattr(right, field))
        max_abs = max(max_abs, abs_diff)
        max_rel = max(max_rel, rel_diff)
    return max_abs, max_rel


def compare_execution_snapshots(
    left: ExecutionSnapshot,
    right: ExecutionSnapshot,
    *,
    comparison: str,
    stage: str,
    t_init: float,
    tolerance: ParityTolerance,
) -> dict[str, Any]:
    train_abs, train_rel = scalar_abs_rel(
        float(left.aggregate.total.detach().cpu()),
        float(right.aggregate.total.detach().cpu()),
    )
    heldout_abs, heldout_rel = scalar_abs_rel(
        float(left.heldout_aggregate.total.detach().cpu()),
        float(right.heldout_aggregate.total.detach().cpu()),
    )
    per_abs, per_rel = max_component_difference(left.per_scenario, right.per_scenario)
    heldout_per_abs, heldout_per_rel = max_component_difference(
        left.heldout_per_scenario,
        right.heldout_per_scenario,
    )
    beta_abs, beta_rel = vector_abs_rel(left.beta, right.beta)
    grad_abs, grad_rel = vector_abs_rel(left.grad, right.grad)
    train_trajectory_abs = max_result_difference(left.rollouts, right.rollouts)
    heldout_trajectory_abs = max_result_difference(left.heldout_rollouts, right.heldout_rollouts)
    improvement_flags_match = left.improved == right.improved
    failure_flags_match = left.failed == right.failed
    left_min_gap, left_min_speed = min_gap_and_speed(left.rollouts)
    right_min_gap, right_min_speed = min_gap_and_speed(right.rollouts)
    left_heldout_min_gap, left_heldout_min_speed = min_gap_and_speed(left.heldout_rollouts)
    right_heldout_min_gap, right_heldout_min_speed = min_gap_and_speed(right.heldout_rollouts)
    min_gap_abs = abs(left_min_gap - right_min_gap)
    min_speed_abs = abs(left_min_speed - right_min_speed)
    heldout_min_gap_abs = abs(left_heldout_min_gap - right_heldout_min_gap)
    heldout_min_speed_abs = abs(left_heldout_min_speed - right_heldout_min_speed)
    passed = (
        train_trajectory_abs <= tolerance.trajectory_atol
        and heldout_trajectory_abs <= tolerance.trajectory_atol
        and tolerance_passed(train_abs, train_rel, atol=tolerance.scalar_atol, rtol=tolerance.scalar_rtol)
        and tolerance_passed(
            heldout_abs,
            heldout_rel,
            atol=tolerance.scalar_atol,
            rtol=tolerance.scalar_rtol,
        )
        and tolerance_passed(per_abs, per_rel, atol=tolerance.scalar_atol, rtol=tolerance.scalar_rtol)
        and tolerance_passed(
            heldout_per_abs,
            heldout_per_rel,
            atol=tolerance.scalar_atol,
            rtol=tolerance.scalar_rtol,
        )
        and tolerance_passed(beta_abs, beta_rel, atol=tolerance.vector_atol, rtol=tolerance.vector_rtol)
        and tolerance_passed(grad_abs, grad_rel, atol=tolerance.vector_atol, rtol=tolerance.vector_rtol)
        and left.heldout_grad_enabled is False
        and right.heldout_grad_enabled is False
        and min_gap_abs <= tolerance.trajectory_atol
        and min_speed_abs <= tolerance.trajectory_atol
        and heldout_min_gap_abs <= tolerance.trajectory_atol
        and heldout_min_speed_abs <= tolerance.trajectory_atol
        and (stage == "update0" or (improvement_flags_match and failure_flags_match))
    )
    return {
        "milestone": "2_d1b_gpu_scenario_batching",
        "comparison": comparison,
        "stage": stage,
        "K": left.horizon,
        "horizon_label": "T" if left.horizon == 80 else str(left.horizon),
        "initialization_id": left.initialization_id,
        "T_init": t_init,
        "update": left.update,
        "left_device": left.device,
        "left_execution_mode": left.mode,
        "right_device": right.device,
        "right_execution_mode": right.mode,
        "passed": passed,
        "train_trajectory_max_abs_diff": train_trajectory_abs,
        "heldout_trajectory_max_abs_diff": heldout_trajectory_abs,
        "train_total_abs_diff": train_abs,
        "train_total_rel_diff": train_rel,
        "heldout_total_abs_diff": heldout_abs,
        "heldout_total_rel_diff": heldout_rel,
        "per_scenario_component_max_abs_diff": per_abs,
        "per_scenario_component_max_rel_diff": per_rel,
        "heldout_per_scenario_component_max_abs_diff": heldout_per_abs,
        "heldout_per_scenario_component_max_rel_diff": heldout_per_rel,
        "beta_max_abs_diff": beta_abs,
        "beta_max_rel_diff": beta_rel,
        "grad_max_abs_diff": grad_abs,
        "grad_max_rel_diff": grad_rel,
        "min_gap_abs_diff": min_gap_abs,
        "min_speed_abs_diff": min_speed_abs,
        "heldout_min_gap_abs_diff": heldout_min_gap_abs,
        "heldout_min_speed_abs_diff": heldout_min_speed_abs,
        "improvement_flags_match": improvement_flags_match if stage != "update0" else True,
        "failure_flags_match": failure_flags_match if stage != "update0" else True,
        "left": snapshot_summary(left),
        "right": snapshot_summary(right),
    }


def snapshot_summary(snapshot: ExecutionSnapshot) -> dict[str, Any]:
    return {
        "device_actual": snapshot.device,
        "execution_mode": snapshot.mode,
        "beta": snapshot.beta,
        "train_components": component_floats(snapshot.aggregate),
        "train_weighted_components": weighted_component_floats(snapshot.aggregate),
        "train_per_scenario": {
            field: getattr(snapshot.per_scenario, field) for field in COMPONENT_FIELDS
        },
        "train_grad": snapshot.grad,
        "train_grad_norm": float(torch.linalg.vector_norm(snapshot.grad).detach().cpu()),
        "heldout_components": component_floats(snapshot.heldout_aggregate),
        "heldout_weighted_components": weighted_component_floats(snapshot.heldout_aggregate),
        "heldout_per_scenario": {
            field: getattr(snapshot.heldout_per_scenario, field) for field in COMPONENT_FIELDS
        },
        "heldout_grad_enabled": snapshot.heldout_grad_enabled,
    }


def run_optimizer_probe(
    context: ParityContext,
    train: PreparedSplit,
    held_out: PreparedSplit,
    *,
    mode: ExecutionMode,
    beta_initial: torch.Tensor,
    horizon: int,
    initialization_id: int,
    device: torch.device,
    probe_lr: float,
    probe_updates: int,
) -> tuple[ExecutionSnapshot, ExecutionSnapshot, dict[str, Any]]:
    beta = beta_initial.detach().clone().to(device=device, dtype=DTYPE).requires_grad_(True)
    optimizer = torch.optim.Adam(
        [beta],
        lr=probe_lr,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.0,
        amsgrad=False,
    )
    initial = make_snapshot(
        context,
        train,
        held_out,
        mode=mode,
        beta=beta,
        horizon=horizon,
        initialization_id=initialization_id,
        update=0,
        device=device,
    )
    reset_peak_memory_if_cuda(device)
    failed = False
    failure_reason = ""
    completed = 0
    for _ in range(probe_updates):
        optimizer.zero_grad(set_to_none=True)
        aggregate, _, _ = evaluate_prepared(
            context,
            train,
            mode=mode,
            beta=beta,
            horizon=horizon,
            device=device,
        )
        aggregate.total.backward()
        if not bool(torch.isfinite(aggregate.total).detach().cpu()):
            failed = True
            failure_reason = "nonfinite_loss"
            break
        if beta.grad is None or not bool(torch.all(torch.isfinite(beta.grad)).detach().cpu()):
            failed = True
            failure_reason = "nonfinite_grad"
            break
        optimizer.step()
        completed += 1
    final = make_snapshot(
        context,
        train,
        held_out,
        mode=mode,
        beta=beta.detach(),
        horizon=horizon,
        initialization_id=initialization_id,
        update=completed,
        device=device,
    )
    initial_total = float(initial.aggregate.total.detach().cpu())
    final_total = float(final.aggregate.total.detach().cpu())
    final = replace(
        final,
        improved=final_total < initial_total,
        failed=failed,
    )
    return initial, final, {
        "device_actual": str(device),
        "execution_mode": mode,
        "probe_lr": probe_lr,
        "probe_updates_requested": probe_updates,
        "probe_updates_completed": completed,
        "failed": failed,
        "failure_reason": failure_reason,
        "improved": final_total < initial_total,
        "peak_memory_bytes": peak_memory_if_cuda(device),
    }


def time_training_case(
    context: ParityContext,
    train: PreparedSplit,
    *,
    mode: ExecutionMode,
    beta_initial: torch.Tensor,
    horizon: int,
    initialization_id: int,
    t_init: float,
    device: torch.device,
    warmup_repeats: int,
    timed_repeats: int,
) -> list[dict[str, Any]]:
    def execute() -> None:
        beta = beta_initial.detach().clone().to(device=device, dtype=DTYPE).requires_grad_(True)
        aggregate, _, _ = evaluate_prepared(
            context,
            train,
            mode=mode,
            beta=beta,
            horizon=horizon,
            device=device,
        )
        aggregate.total.backward()

    for _ in range(warmup_repeats):
        execute()
    synchronize_if_cuda(device)
    reset_peak_memory_if_cuda(device)
    rows: list[dict[str, Any]] = []
    for repeat in range(timed_repeats):
        synchronize_if_cuda(device)
        start = time.perf_counter()
        execute()
        synchronize_if_cuda(device)
        rows.append(
            timing_row(
                device=device,
                mode=mode,
                operation="train_objective_backward",
                horizon=horizon,
                initialization_id=initialization_id,
                t_init=t_init,
                repeat=repeat,
                runtime_s=time.perf_counter() - start,
                warmup_repeats=warmup_repeats,
                timed_repeats=timed_repeats,
                peak_memory_bytes=peak_memory_if_cuda(device),
            )
        )
    return rows


def time_heldout_case(
    context: ParityContext,
    held_out: PreparedSplit,
    *,
    mode: ExecutionMode,
    beta_initial: torch.Tensor,
    horizon: int,
    initialization_id: int,
    t_init: float,
    device: torch.device,
    warmup_repeats: int,
    timed_repeats: int,
) -> list[dict[str, Any]]:
    def execute() -> None:
        with torch.no_grad():
            evaluate_prepared(
                context,
                held_out,
                mode=mode,
                beta=beta_initial.to(device=device, dtype=DTYPE),
                horizon=80,
                device=device,
            )

    for _ in range(warmup_repeats):
        execute()
    synchronize_if_cuda(device)
    reset_peak_memory_if_cuda(device)
    rows: list[dict[str, Any]] = []
    for repeat in range(timed_repeats):
        synchronize_if_cuda(device)
        start = time.perf_counter()
        execute()
        synchronize_if_cuda(device)
        rows.append(
            timing_row(
                device=device,
                mode=mode,
                operation="heldout_no_grad_forward",
                horizon=horizon,
                initialization_id=initialization_id,
                t_init=t_init,
                repeat=repeat,
                runtime_s=time.perf_counter() - start,
                warmup_repeats=warmup_repeats,
                timed_repeats=timed_repeats,
                peak_memory_bytes=peak_memory_if_cuda(device),
            )
        )
    return rows


def timing_row(
    *,
    device: torch.device,
    mode: ExecutionMode,
    operation: str,
    horizon: int,
    initialization_id: int,
    t_init: float,
    repeat: int,
    runtime_s: float,
    warmup_repeats: int,
    timed_repeats: int,
    peak_memory_bytes: int | None,
) -> dict[str, Any]:
    return {
        "milestone": "2_d1b_gpu_scenario_batching",
        "device_actual": str(device),
        "execution_mode": mode,
        "operation": operation,
        "K": horizon,
        "horizon_label": "T" if horizon == 80 else str(horizon),
        "initialization_id": initialization_id,
        "T_init": t_init,
        "repeat": repeat,
        "runtime_s": runtime_s,
        "warmup_repeats": warmup_repeats,
        "timed_repeats": timed_repeats,
        "peak_memory_bytes": peak_memory_bytes,
    }


def summarize_timing(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str, int], list[dict[str, Any]]] = {}
    for row in rows:
        key = (row["device_actual"], row["execution_mode"], row["operation"], int(row["K"]))
        groups.setdefault(key, []).append(row)
    summary: list[dict[str, Any]] = []
    for (device, mode, operation, horizon), items in sorted(groups.items()):
        runtimes = [float(item["runtime_s"]) for item in items]
        summary.append(
            {
                "device_actual": device,
                "execution_mode": mode,
                "operation": operation,
                "K": horizon,
                "sample_count": len(runtimes),
                "median_runtime_s": statistics.median(runtimes),
                "mean_runtime_s": statistics.fmean(runtimes),
                "min_runtime_s": min(runtimes),
                "max_runtime_s": max(runtimes),
                "max_peak_memory_bytes": max(
                    (int(item["peak_memory_bytes"] or 0) for item in items),
                    default=0,
                ),
            }
        )
    lookup = {
        (row["device_actual"], row["operation"], row["K"], row["execution_mode"]): row
        for row in summary
    }
    for row in summary:
        if row["execution_mode"] != "scenario-batched":
            continue
        unbatched = lookup.get((row["device_actual"], row["operation"], row["K"], "unbatched"))
        row["batched_over_unbatched_median_ratio"] = (
            row["median_runtime_s"] / unbatched["median_runtime_s"] if unbatched else math.nan
        )
    return summary


def max_differences(rows: Sequence[dict[str, Any]]) -> dict[str, float]:
    fields = (
        "train_trajectory_max_abs_diff",
        "heldout_trajectory_max_abs_diff",
        "train_total_abs_diff",
        "train_total_rel_diff",
        "heldout_total_abs_diff",
        "heldout_total_rel_diff",
        "per_scenario_component_max_abs_diff",
        "per_scenario_component_max_rel_diff",
        "heldout_per_scenario_component_max_abs_diff",
        "heldout_per_scenario_component_max_rel_diff",
        "beta_max_abs_diff",
        "beta_max_rel_diff",
        "grad_max_abs_diff",
        "grad_max_rel_diff",
        "min_gap_abs_diff",
        "min_speed_abs_diff",
        "heldout_min_gap_abs_diff",
        "heldout_min_speed_abs_diff",
    )
    return {field: max(float(row[field]) for row in rows) for field in fields}


def summary_markdown(summary: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Milestone 2 D.1b GPU Scenario Batching Summary",
            "",
            f"- correctness passed: `{summary['passed']}`",
            f"- parity rows: `{summary['parity_row_count']}`",
            f"- timing samples: `{summary['timing_sample_count']}`",
            f"- horizons: `{summary['horizons']}`",
            f"- initializations: `{summary['initialization_count']}`",
            f"- warm-up repetitions: `{summary['warmup_repeats']}`",
            f"- timed repetitions: `{summary['timed_repeats']}`",
            "",
            "## Maximum Differences",
            "",
            "```json",
            json.dumps(json_safe(summary["max_differences"]), indent=2, sort_keys=True),
            "```",
            "",
            "## Timing",
            "",
            "```json",
            json.dumps(json_safe(summary["timing_summary"]), indent=2, sort_keys=True),
            "```",
            "",
            "## D.2 Boundary",
            "",
            "These results are descriptive infrastructure evidence only. This report does not select or recommend a D.2 device or execution policy.",
            "",
        ]
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.probe_lr <= 0.0:
        raise SystemExit("--probe-lr must be positive")
    if args.probe_updates < 1:
        raise SystemExit("--probe-updates must be >= 1")
    if args.warmup_repeats < 0:
        raise SystemExit("--warmup-repeats must be >= 0")
    if args.timed_repeats < 1:
        raise SystemExit("--timed-repeats must be >= 1")

    devices = tuple(item.strip() for item in args.device.split(",") if item.strip())
    if not devices:
        raise SystemExit("--device must name at least one device")
    modes: tuple[ExecutionMode, ...]
    if args.execution_mode == "compare":
        modes = ("unbatched", "scenario-batched")
    else:
        modes = (args.execution_mode,)

    limits = {}
    warmup_repeats = args.warmup_repeats
    timed_repeats = args.timed_repeats
    probe_updates = args.probe_updates
    if args.smoke:
        limits = {"train_limit": 1, "held_out_limit": 1, "horizon_limit": 1, "init_limit": 1}
        warmup_repeats = 0
        timed_repeats = 1
        probe_updates = min(probe_updates, 1)
        devices = ("cpu",)
    if "cuda" in devices and not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable to PyTorch; rerun with escalation if sandboxing is suspected")
    if args.execution_mode == "compare" and set(devices) != {"cpu", "cuda"} and not args.smoke:
        raise SystemExit("full compare mode requires --device cpu,cuda")

    context = build_parity_context(**limits)
    tolerance = ParityTolerance()
    prepared: dict[str, tuple[PreparedSplit, PreparedSplit]] = {}
    metadata: dict[str, Any] = {}
    for device_name in devices:
        device = torch.device(device_name)
        prepared[device_name] = (
            prepare_split(context.train, dtype=DTYPE, device=device),
            prepare_split(context.held_out, dtype=DTYPE, device=device),
        )
        metadata[device_name] = device_metadata(device_name)

    initial_snapshots: dict[tuple[str, ExecutionMode, int, int], ExecutionSnapshot] = {}
    final_snapshots: dict[tuple[str, ExecutionMode, int, int], ExecutionSnapshot] = {}
    probe_info: list[dict[str, Any]] = []
    timing_rows: list[dict[str, Any]] = []
    for init_id, beta_initial in enumerate(context.init_betas_cpu):
        t_init = context.t_init_values[init_id]
        for horizon in context.horizons:
            for device_index, device_name in enumerate(devices):
                device = torch.device(device_name)
                train, held_out = prepared[device_name]
                ordered_modes = modes if (init_id + horizon + device_index) % 2 == 0 else tuple(reversed(modes))
                for mode in ordered_modes:
                    initial, final, info = run_optimizer_probe(
                        context,
                        train,
                        held_out,
                        mode=mode,
                        beta_initial=beta_initial,
                        horizon=horizon,
                        initialization_id=init_id,
                        device=device,
                        probe_lr=args.probe_lr,
                        probe_updates=probe_updates,
                    )
                    initial_snapshots[(device_name, mode, horizon, init_id)] = initial
                    final_snapshots[(device_name, mode, horizon, init_id)] = final
                    info.update({"K": horizon, "initialization_id": init_id, "T_init": t_init})
                    probe_info.append(info)
                    timing_rows.extend(
                        time_training_case(
                            context,
                            train,
                            mode=mode,
                            beta_initial=beta_initial,
                            horizon=horizon,
                            initialization_id=init_id,
                            t_init=t_init,
                            device=device,
                            warmup_repeats=warmup_repeats,
                            timed_repeats=timed_repeats,
                        )
                    )
                    timing_rows.extend(
                        time_heldout_case(
                            context,
                            held_out,
                            mode=mode,
                            beta_initial=beta_initial,
                            horizon=horizon,
                            initialization_id=init_id,
                            t_init=t_init,
                            device=device,
                            warmup_repeats=warmup_repeats,
                            timed_repeats=timed_repeats,
                        )
                    )

    parity_rows: list[dict[str, Any]] = []
    if args.execution_mode == "compare":
        comparisons = []
        if "cpu" in devices:
            comparisons.append(("cpu_unbatched_vs_batched", ("cpu", "unbatched"), ("cpu", "scenario-batched")))
        if "cuda" in devices:
            comparisons.append(
                ("cuda_unbatched_vs_batched", ("cuda", "unbatched"), ("cuda", "scenario-batched"))
            )
        if "cpu" in devices and "cuda" in devices:
            comparisons.append(
                ("batched_cpu_vs_cuda", ("cpu", "scenario-batched"), ("cuda", "scenario-batched"))
            )
        for init_id, _ in enumerate(context.init_betas_cpu):
            t_init = context.t_init_values[init_id]
            for horizon in context.horizons:
                for comparison, left_key, right_key in comparisons:
                    for stage, snapshots in (("update0", initial_snapshots), ("probe_final", final_snapshots)):
                        parity_rows.append(
                            compare_execution_snapshots(
                                snapshots[(left_key[0], left_key[1], horizon, init_id)],
                                snapshots[(right_key[0], right_key[1], horizon, init_id)],
                                comparison=comparison,
                                stage=stage,
                                t_init=t_init,
                                tolerance=tolerance,
                            )
                        )

    timing_summary = summarize_timing(timing_rows)
    summary = {
        "milestone": "2_d1b_gpu_scenario_batching",
        "passed": all(bool(row["passed"]) for row in parity_rows) if parity_rows else True,
        "parity_row_count": len(parity_rows),
        "timing_sample_count": len(timing_rows),
        "probe_lr": args.probe_lr,
        "probe_updates": probe_updates,
        "warmup_repeats": warmup_repeats,
        "timed_repeats": timed_repeats,
        "horizons": list(context.horizons),
        "initialization_count": len(context.init_betas_cpu),
        "train_scenario_count": len(context.train),
        "held_out_scenario_count": len(context.held_out),
        "metadata": metadata,
        "max_differences": max_differences(parity_rows) if parity_rows else {},
        "probe_info": probe_info,
        "timing_summary": timing_summary,
        "d2_policy_selected": False,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "parity.jsonl", parity_rows)
    write_jsonl(args.output_dir / "timing.jsonl", timing_rows)
    write_json(args.output_dir / "summary.json", summary)
    (args.output_dir / "summary.md").write_text(summary_markdown(summary))
    return summary


def main() -> None:
    summary = run(parse_args())
    print(
        json.dumps(
            {
                "passed": summary["passed"],
                "parity_row_count": summary["parity_row_count"],
                "timing_sample_count": summary["timing_sample_count"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
