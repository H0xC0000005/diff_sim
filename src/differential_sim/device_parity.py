"""CPU/CUDA parity utilities for Milestone 2 D.1 infrastructure checks."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from importlib import metadata
import json
import math
import platform
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import torch

from differential_sim.controllers import (
    InputNormalization,
    StructuredHeadwayController,
    input_normalization_from_scenarios,
    noisy_center_betas,
    tensor_to_list,
)
from differential_sim.milestone1_diagnostics import (
    HORIZONS,
    build_controller,
    diagnostic_scenarios,
    held_out_scenarios,
    min_gap_and_speed,
    scenario_configs_to_dicts,
)
from differential_sim.objectives import (
    ObjectiveComponents,
    ObjectiveConfig,
    component_floats,
    default_idm_params,
    weighted_component_floats,
)
from differential_sim.rollout import RolloutResult
from differential_sim.scenarios import ScenarioConfig
from differential_sim.temporal_gradients import (
    ScenarioRollout,
    default_rollout_config,
    objective_for_scenarios,
)


DTYPE = torch.float64
TRAIN_SCENARIO_COUNT = 14
HELD_OUT_SCENARIO_COUNT = 8
UPDATE_COUNT_OPTIONS = (100, 200, 300, 500)


@dataclass(frozen=True)
class ParityTolerance:
    trajectory_atol: float = 1e-8
    scalar_atol: float = 1e-8
    scalar_rtol: float = 1e-7
    vector_atol: float = 1e-7
    vector_rtol: float = 1e-6


@dataclass(frozen=True)
class DeviceMetadata:
    requested: str
    actual: str
    dtype: str
    python: str
    platform: str
    torch_version: str
    diffidm_version: str | None
    cuda_available: bool
    torch_cuda_version: str | None
    cuda_device_count: int
    gpu_name: str | None
    gpu_compute_capability: str | None
    gpu_total_memory_bytes: int | None


@dataclass(frozen=True)
class ParityContext:
    train: list[tuple[str, ScenarioConfig]]
    held_out: list[tuple[str, ScenarioConfig]]
    normalization: InputNormalization
    controller: StructuredHeadwayController
    base_params: Any
    rollout_config: Any
    objective_config: ObjectiveConfig
    horizons: tuple[int, ...]
    init_betas_cpu: list[torch.Tensor]
    t_init_values: tuple[float, ...]


@dataclass(frozen=True)
class Snapshot:
    device: str
    horizon: int
    initialization_id: int
    update: int
    beta: torch.Tensor
    components: ObjectiveComponents
    rollouts: list[ScenarioRollout]
    grad: torch.Tensor
    train_runtime_s: float
    heldout_components: ObjectiveComponents
    heldout_rollouts: list[ScenarioRollout]
    heldout_grad_enabled: bool


def json_safe(value: Any) -> Any:
    """Convert tensors, dataclasses, paths, and numeric scalars to JSON-safe values."""

    if isinstance(value, torch.Tensor):
        if value.ndim == 0:
            return float(value.detach().cpu().item())
        return tensor_to_list(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    if isinstance(value, float):
        if math.isfinite(value):
            return value
        return str(value)
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if hasattr(value, "__dataclass_fields__"):
        return json_safe(asdict(value))
    return str(value)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_safe(value), indent=2, sort_keys=True) + "\n")


def write_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(json_safe(row), sort_keys=True) + "\n" for row in rows))


def resolve_device(device: str) -> torch.device:
    resolved = torch.device(device)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False")
    return resolved


def device_metadata(device: str | torch.device, *, dtype: torch.dtype = DTYPE) -> DeviceMetadata:
    resolved = resolve_device(str(device))
    diffidm_version: str | None
    try:
        diffidm_version = metadata.version("diffidm")
    except metadata.PackageNotFoundError:
        diffidm_version = None

    gpu_name = None
    gpu_compute_capability = None
    gpu_total_memory_bytes = None
    if resolved.type == "cuda":
        index = resolved.index if resolved.index is not None else torch.cuda.current_device()
        props = torch.cuda.get_device_properties(index)
        gpu_name = props.name
        gpu_compute_capability = f"{props.major}.{props.minor}"
        gpu_total_memory_bytes = int(props.total_memory)

    return DeviceMetadata(
        requested=str(device),
        actual=str(resolved),
        dtype=str(dtype).replace("torch.", "torch."),
        python=sys.version,
        platform=f"{platform.system()} {platform.release()}",
        torch_version=torch.__version__,
        diffidm_version=diffidm_version,
        cuda_available=torch.cuda.is_available(),
        torch_cuda_version=torch.version.cuda,
        cuda_device_count=torch.cuda.device_count(),
        gpu_name=gpu_name,
        gpu_compute_capability=gpu_compute_capability,
        gpu_total_memory_bytes=gpu_total_memory_bytes,
    )


def build_parity_context(
    *,
    dtype: torch.dtype = DTYPE,
    train_limit: int | None = None,
    held_out_limit: int | None = None,
    horizon_limit: int | None = None,
    init_limit: int | None = None,
) -> ParityContext:
    rollout_config = default_rollout_config()
    base_params = default_idm_params(time_headway=1.4)
    train = diagnostic_scenarios()
    held_out = held_out_scenarios()
    horizons = tuple(HORIZONS)
    t_init_values = (0.9, 1.2, 1.4, 1.6, 1.9, 2.2)
    if train_limit is not None:
        train = train[:train_limit]
    if held_out_limit is not None:
        held_out = held_out[:held_out_limit]
    if horizon_limit is not None:
        horizons = horizons[:horizon_limit]
    if init_limit is not None:
        t_init_values = t_init_values[:init_limit]

    normalization = input_normalization_from_scenarios(
        [config for _, config in train],
        base_params,
        rollout_config,
        dtype=dtype,
        device="cpu",
    )
    controller = build_controller("normalized", normalization)
    init_betas_cpu = noisy_center_betas(
        centers=t_init_values,
        seeds=tuple(range(len(t_init_values))),
        dtype=dtype,
        device="cpu",
    )
    return ParityContext(
        train=train,
        held_out=held_out,
        normalization=normalization,
        controller=controller,
        base_params=base_params,
        rollout_config=rollout_config,
        objective_config=ObjectiveConfig(),
        horizons=horizons,
        init_betas_cpu=init_betas_cpu,
        t_init_values=t_init_values,
    )


def synchronize_if_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def reset_peak_memory_if_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def peak_memory_if_cuda(device: torch.device) -> int | None:
    if device.type != "cuda":
        return None
    return int(torch.cuda.max_memory_allocated(device))


def objective_snapshot(
    context: ParityContext,
    *,
    beta: torch.Tensor,
    horizon: int,
    initialization_id: int,
    update: int,
    device: str | torch.device,
    dtype: torch.dtype = DTYPE,
) -> Snapshot:
    resolved = resolve_device(str(device))
    beta_work = beta.detach().clone().to(device=resolved, dtype=dtype).requires_grad_(True)
    synchronize_if_cuda(resolved)
    start = time.perf_counter()
    components, rollouts = objective_for_scenarios(
        context.train,
        beta=beta_work,
        controller=context.controller,
        base_params=context.base_params,
        rollout_config=context.rollout_config,
        horizon_k=horizon,
        objective_config=context.objective_config,
        dtype=dtype,
        device=resolved,
    )
    grad = torch.autograd.grad(components.total, beta_work, allow_unused=False)[0]
    synchronize_if_cuda(resolved)
    runtime_s = time.perf_counter() - start

    with torch.no_grad():
        heldout_components, heldout_rollouts = objective_for_scenarios(
            context.held_out,
            beta=beta_work.detach(),
            controller=context.controller,
            base_params=context.base_params,
            rollout_config=context.rollout_config,
            horizon_k=80,
            objective_config=context.objective_config,
            dtype=dtype,
            device=resolved,
        )
        heldout_grad_enabled = torch.is_grad_enabled()

    return Snapshot(
        device=str(resolved),
        horizon=horizon,
        initialization_id=initialization_id,
        update=update,
        beta=beta_work.detach(),
        components=components,
        rollouts=rollouts,
        grad=grad.detach(),
        train_runtime_s=runtime_s,
        heldout_components=heldout_components,
        heldout_rollouts=heldout_rollouts,
        heldout_grad_enabled=heldout_grad_enabled,
    )


def snapshot_row(snapshot: Snapshot, *, stage: str, t_init: float) -> dict[str, Any]:
    min_gap_train, min_speed_train = min_gap_and_speed([rollout.result for rollout in snapshot.rollouts])
    min_gap_heldout, min_speed_heldout = min_gap_and_speed([rollout.result for rollout in snapshot.heldout_rollouts])
    return {
        "milestone": "2_d1_cpu_gpu_infrastructure",
        "stage": stage,
        "device_actual": snapshot.device,
        "dtype": str(snapshot.beta.dtype),
        "K": snapshot.horizon,
        "horizon_label": "T" if snapshot.horizon == 80 else str(snapshot.horizon),
        "initialization_id": snapshot.initialization_id,
        "T_init": t_init,
        "update": snapshot.update,
        "beta": snapshot.beta,
        "train_components": component_floats(snapshot.components),
        "train_weighted_components": weighted_component_floats(snapshot.components),
        "train_grad": snapshot.grad,
        "train_grad_norm": float(torch.linalg.vector_norm(snapshot.grad).detach().cpu().item()),
        "train_runtime_s": snapshot.train_runtime_s,
        "heldout_components": component_floats(snapshot.heldout_components),
        "heldout_weighted_components": weighted_component_floats(snapshot.heldout_components),
        "heldout_grad_enabled": snapshot.heldout_grad_enabled,
        "min_gap_train": min_gap_train,
        "min_speed_train": min_speed_train,
        "min_gap_heldout": min_gap_heldout,
        "min_speed_heldout": min_speed_heldout,
    }


def max_rollout_difference(a: Sequence[ScenarioRollout], b: Sequence[ScenarioRollout]) -> float:
    if len(a) != len(b):
        raise ValueError("rollout lists must have the same length")
    max_diff = 0.0
    for left, right in zip(a, b, strict=True):
        if left.scenario_name != right.scenario_name:
            raise ValueError(f"scenario mismatch: {left.scenario_name} != {right.scenario_name}")
        max_diff = max(max_diff, rollout_result_difference(left.result, right.result))
    return max_diff


def max_result_difference(results_a: Sequence[RolloutResult], results_b: Sequence[RolloutResult]) -> float:
    if len(results_a) != len(results_b):
        raise ValueError("result lists must have the same length")
    max_diff = 0.0
    for left, right in zip(results_a, results_b, strict=True):
        max_diff = max(max_diff, rollout_result_difference(left, right))
    return max_diff


def rollout_result_difference(left: RolloutResult, right: RolloutResult) -> float:
    max_diff = 0.0
    for field in ("follower_x", "follower_v", "follower_a", "gap", "delta_v"):
        left_tensor = getattr(left, field).detach().cpu()
        right_tensor = getattr(right, field).detach().cpu()
        diff = torch.max(torch.abs(left_tensor - right_tensor))
        max_diff = max(max_diff, float(diff.item()))
    return max_diff


def scalar_abs_rel(left: float, right: float) -> tuple[float, float]:
    abs_diff = abs(left - right)
    denom = max(abs(left), abs(right), 1e-300)
    return abs_diff, abs_diff / denom


def vector_abs_rel(left: torch.Tensor, right: torch.Tensor) -> tuple[float, float]:
    left_cpu = left.detach().cpu()
    right_cpu = right.detach().cpu()
    abs_diff = float(torch.max(torch.abs(left_cpu - right_cpu)).item())
    denom = float(torch.max(torch.maximum(torch.abs(left_cpu), torch.abs(right_cpu))).item())
    if denom == 0.0:
        return abs_diff, 0.0 if abs_diff == 0.0 else math.inf
    return abs_diff, abs_diff / denom


def tolerance_passed(abs_diff: float, rel_diff: float, *, atol: float, rtol: float | None = None) -> bool:
    return abs_diff <= atol or (rtol is not None and rel_diff <= rtol)


def compare_snapshots(
    cpu: Snapshot,
    cuda: Snapshot,
    *,
    tolerance: ParityTolerance = ParityTolerance(),
    stage: str,
    t_init: float,
) -> dict[str, Any]:
    train_abs, train_rel = scalar_abs_rel(
        float(cpu.components.total.detach().cpu().item()),
        float(cuda.components.total.detach().cpu().item()),
    )
    heldout_abs, heldout_rel = scalar_abs_rel(
        float(cpu.heldout_components.total.detach().cpu().item()),
        float(cuda.heldout_components.total.detach().cpu().item()),
    )
    beta_abs, beta_rel = vector_abs_rel(cpu.beta, cuda.beta)
    grad_abs, grad_rel = vector_abs_rel(cpu.grad, cuda.grad)
    train_traj_abs = max_rollout_difference(cpu.rollouts, cuda.rollouts)
    heldout_traj_abs = max_rollout_difference(cpu.heldout_rollouts, cuda.heldout_rollouts)
    passed = (
        train_traj_abs <= tolerance.trajectory_atol
        and heldout_traj_abs <= tolerance.trajectory_atol
        and tolerance_passed(train_abs, train_rel, atol=tolerance.scalar_atol, rtol=tolerance.scalar_rtol)
        and tolerance_passed(heldout_abs, heldout_rel, atol=tolerance.scalar_atol, rtol=tolerance.scalar_rtol)
        and tolerance_passed(beta_abs, beta_rel, atol=tolerance.vector_atol, rtol=tolerance.vector_rtol)
        and tolerance_passed(grad_abs, grad_rel, atol=tolerance.vector_atol, rtol=tolerance.vector_rtol)
        and cpu.heldout_grad_enabled is False
        and cuda.heldout_grad_enabled is False
    )
    return {
        "milestone": "2_d1_cpu_gpu_infrastructure",
        "stage": stage,
        "K": cpu.horizon,
        "horizon_label": "T" if cpu.horizon == 80 else str(cpu.horizon),
        "initialization_id": cpu.initialization_id,
        "T_init": t_init,
        "update": cpu.update,
        "passed": passed,
        "train_trajectory_max_abs_diff": train_traj_abs,
        "heldout_trajectory_max_abs_diff": heldout_traj_abs,
        "train_total_abs_diff": train_abs,
        "train_total_rel_diff": train_rel,
        "heldout_total_abs_diff": heldout_abs,
        "heldout_total_rel_diff": heldout_rel,
        "beta_max_abs_diff": beta_abs,
        "beta_max_rel_diff": beta_rel,
        "grad_max_abs_diff": grad_abs,
        "grad_max_rel_diff": grad_rel,
        "cpu": snapshot_row(cpu, stage=stage, t_init=t_init),
        "cuda": snapshot_row(cuda, stage=stage, t_init=t_init),
    }


def adam_probe_snapshot(
    context: ParityContext,
    *,
    beta_init: torch.Tensor,
    horizon: int,
    initialization_id: int,
    device: str | torch.device,
    probe_lr: float,
    probe_updates: int,
    dtype: torch.dtype = DTYPE,
) -> tuple[Snapshot, Snapshot, dict[str, Any]]:
    resolved = resolve_device(str(device))
    beta = beta_init.detach().clone().to(device=resolved, dtype=dtype).requires_grad_(True)
    optimizer = torch.optim.Adam(
        [beta],
        lr=probe_lr,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.0,
        amsgrad=False,
    )
    reset_peak_memory_if_cuda(resolved)
    initial = objective_snapshot(
        context,
        beta=beta,
        horizon=horizon,
        initialization_id=initialization_id,
        update=0,
        device=resolved,
        dtype=dtype,
    )
    runtimes: list[float] = []
    failed = False
    failure_reason = ""
    for _ in range(probe_updates):
        optimizer.zero_grad(set_to_none=True)
        synchronize_if_cuda(resolved)
        start = time.perf_counter()
        components, _ = objective_for_scenarios(
            context.train,
            beta=beta,
            controller=context.controller,
            base_params=context.base_params,
            rollout_config=context.rollout_config,
            horizon_k=horizon,
            objective_config=context.objective_config,
            dtype=dtype,
            device=resolved,
        )
        components.total.backward()
        synchronize_if_cuda(resolved)
        runtimes.append(time.perf_counter() - start)
        if not bool(torch.isfinite(components.total).detach().cpu().item()):
            failed = True
            failure_reason = "nonfinite_loss"
            break
        if beta.grad is None or not bool(torch.all(torch.isfinite(beta.grad)).detach().cpu().item()):
            failed = True
            failure_reason = "nonfinite_grad"
            break
        optimizer.step()
    final = objective_snapshot(
        context,
        beta=beta.detach(),
        horizon=horizon,
        initialization_id=initialization_id,
        update=len(runtimes),
        device=resolved,
        dtype=dtype,
    )
    info = {
        "device_actual": str(resolved),
        "probe_lr": probe_lr,
        "probe_updates_requested": probe_updates,
        "probe_updates_completed": len(runtimes),
        "failed": failed,
        "failure_reason": failure_reason,
        "mean_step_runtime_s": float(sum(runtimes) / len(runtimes)) if runtimes else math.nan,
        "peak_memory_bytes": peak_memory_if_cuda(resolved),
    }
    return initial, final, info


def run_cpu_cuda_parity(
    *,
    probe_lr: float = 0.03,
    probe_updates: int = 10,
    dtype: torch.dtype = DTYPE,
    output_dir: Path | None = None,
    train_limit: int | None = None,
    held_out_limit: int | None = None,
    horizon_limit: int | None = None,
    init_limit: int | None = None,
) -> dict[str, Any]:
    context = build_parity_context(
        dtype=dtype,
        train_limit=train_limit,
        held_out_limit=held_out_limit,
        horizon_limit=horizon_limit,
        init_limit=init_limit,
    )
    cpu_metadata = device_metadata("cpu", dtype=dtype)
    cuda_metadata = device_metadata("cuda", dtype=dtype)
    tolerance = ParityTolerance()
    rows: list[dict[str, Any]] = []
    runtime_rows: list[dict[str, Any]] = []

    for init_id, beta_cpu in enumerate(context.init_betas_cpu):
        t_init = context.t_init_values[init_id]
        for horizon in context.horizons:
            cpu_initial = objective_snapshot(
                context,
                beta=beta_cpu,
                horizon=horizon,
                initialization_id=init_id,
                update=0,
                device="cpu",
                dtype=dtype,
            )
            cuda_initial = objective_snapshot(
                context,
                beta=beta_cpu,
                horizon=horizon,
                initialization_id=init_id,
                update=0,
                device="cuda",
                dtype=dtype,
            )
            rows.append(
                compare_snapshots(
                    cpu_initial,
                    cuda_initial,
                    tolerance=tolerance,
                    stage="update0",
                    t_init=t_init,
                )
            )

            cpu_probe_initial, cpu_probe_final, cpu_info = adam_probe_snapshot(
                context,
                beta_init=beta_cpu,
                horizon=horizon,
                initialization_id=init_id,
                device="cpu",
                probe_lr=probe_lr,
                probe_updates=probe_updates,
                dtype=dtype,
            )
            cuda_probe_initial, cuda_probe_final, cuda_info = adam_probe_snapshot(
                context,
                beta_init=beta_cpu,
                horizon=horizon,
                initialization_id=init_id,
                device="cuda",
                probe_lr=probe_lr,
                probe_updates=probe_updates,
                dtype=dtype,
            )
            rows.append(
                compare_snapshots(
                    cpu_probe_initial,
                    cuda_probe_initial,
                    tolerance=tolerance,
                    stage="probe_update0",
                    t_init=t_init,
                )
            )
            final_row = compare_snapshots(
                cpu_probe_final,
                cuda_probe_final,
                tolerance=tolerance,
                stage="probe_final",
                t_init=t_init,
            )
            final_row["cpu_probe_info"] = cpu_info
            final_row["cuda_probe_info"] = cuda_info
            final_row["improvement_flags_match"] = _improvement_flags_match(cpu_probe_initial, cpu_probe_final, cuda_probe_initial, cuda_probe_final)
            final_row["passed"] = bool(final_row["passed"] and final_row["improvement_flags_match"])
            rows.append(final_row)
            runtime_rows.extend([cpu_info, cuda_info])

    summary = summarize_parity(
        rows,
        runtime_rows=runtime_rows,
        probe_lr=probe_lr,
        probe_updates=probe_updates,
        context=context,
        cpu_metadata=cpu_metadata,
        cuda_metadata=cuda_metadata,
    )
    result = {"rows": rows, "summary": summary}
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        write_jsonl(output_dir / "parity.jsonl", rows)
        write_json(output_dir / "runtime_summary.json", summary["runtime"])
        (output_dir / "summary.md").write_text(summary_markdown(summary))
    return result


def _improvement_flags_match(
    cpu_initial: Snapshot,
    cpu_final: Snapshot,
    cuda_initial: Snapshot,
    cuda_final: Snapshot,
) -> bool:
    cpu_improved = float(cpu_final.components.total.detach().cpu().item()) < float(cpu_initial.components.total.detach().cpu().item())
    cuda_improved = float(cuda_final.components.total.detach().cpu().item()) < float(cuda_initial.components.total.detach().cpu().item())
    return cpu_improved == cuda_improved


def summarize_parity(
    rows: Sequence[dict[str, Any]],
    *,
    runtime_rows: Sequence[dict[str, Any]],
    probe_lr: float,
    probe_updates: int,
    context: ParityContext,
    cpu_metadata: DeviceMetadata,
    cuda_metadata: DeviceMetadata,
) -> dict[str, Any]:
    finite_runtimes = [row["mean_step_runtime_s"] for row in runtime_rows if math.isfinite(row["mean_step_runtime_s"])]
    mean_runtime = float(sum(finite_runtimes) / len(finite_runtimes)) if finite_runtimes else math.nan
    max_peak_memory = max((row["peak_memory_bytes"] or 0 for row in runtime_rows), default=0)
    return {
        "milestone": "2_d1_cpu_gpu_infrastructure",
        "passed": all(bool(row["passed"]) for row in rows),
        "row_count": len(rows),
        "probe_lr": probe_lr,
        "probe_updates": probe_updates,
        "horizons": context.horizons,
        "initialization_count": len(context.init_betas_cpu),
        "train_scenario_count": len(context.train),
        "held_out_scenario_count": len(context.held_out),
        "scenarios": {
            "train": scenario_configs_to_dicts(context.train),
            "held_out": scenario_configs_to_dicts(context.held_out),
        },
        "metadata": {"cpu": cpu_metadata, "cuda": cuda_metadata},
        "max_differences": _max_difference_summary(rows),
        "runtime": {
            "mean_probe_step_runtime_s": mean_runtime,
            "max_cuda_peak_memory_bytes": max_peak_memory,
            "runtime_rows": runtime_rows,
            "estimated_gradient_update_cost_s": {str(n): mean_runtime * 5 * 6 * n for n in UPDATE_COUNT_OPTIONS},
        },
        "recommendations_not_frozen": {
            "N_updates": "choose after reviewing D.1 evidence",
            "cpu_cuda_evidence_policy": "choose after reviewing D.1 parity and cost",
        },
    }


def _max_difference_summary(rows: Sequence[dict[str, Any]]) -> dict[str, float]:
    fields = (
        "train_trajectory_max_abs_diff",
        "heldout_trajectory_max_abs_diff",
        "train_total_abs_diff",
        "train_total_rel_diff",
        "heldout_total_abs_diff",
        "heldout_total_rel_diff",
        "beta_max_abs_diff",
        "beta_max_rel_diff",
        "grad_max_abs_diff",
        "grad_max_rel_diff",
    )
    return {field: max((float(row[field]) for row in rows), default=math.nan) for field in fields}


def summary_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Milestone 2 D.1 CPU/GPU Infrastructure Summary",
        "",
        f"- passed: `{summary['passed']}`",
        f"- probe LR: `{summary['probe_lr']}`",
        f"- probe updates: `{summary['probe_updates']}`",
        f"- horizons: `{list(summary['horizons'])}`",
        f"- initializations: `{summary['initialization_count']}`",
        f"- training scenarios: `{summary['train_scenario_count']}`",
        f"- held-out scenarios: `{summary['held_out_scenario_count']}`",
        "",
        "## Max Differences",
        "",
        "```json",
        json.dumps(json_safe(summary["max_differences"]), indent=2, sort_keys=True),
        "```",
        "",
        "## Runtime",
        "",
        "```json",
        json.dumps(json_safe(summary["runtime"]), indent=2, sort_keys=True),
        "```",
        "",
        "## D.2 Boundary",
        "",
        "These recommendations are not frozen D.2 settings.",
        "",
        "```json",
        json.dumps(json_safe(summary["recommendations_not_frozen"]), indent=2, sort_keys=True),
        "```",
        "",
    ]
    return "\n".join(lines)
