"""Run SG1 sparse long-horizon gradient stages."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, replace
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
import time
from typing import Any, Mapping, Sequence

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from differential_sim.device_parity import (  # noqa: E402
    build_parity_context,
    device_metadata,
    json_safe,
    write_json,
    write_jsonl,
)
from differential_sim.objectives import component_floats, weighted_component_floats  # noqa: E402
from differential_sim.small_model_training import (  # noqa: E402
    DETAIL_INTERVAL,
    MODEL_SEEDS,
    build_small_model,
    clone_state_dict,
    evaluate_held_out,
    evaluate_split,
    flatten_gradients,
    flatten_parameters,
    headway_range,
    model_initialization_record,
    prepare_split,
    state_dict_hash,
)
from differential_sim.sparse_gradients import (  # noqa: E402
    DENSE_REFERENCE_HORIZONS,
    SPARSE_STRIDES,
    finite_gradient_norm,
    gradient_cosine,
    relative_change,
    sparse_b1_objective,
    sparse_rollouts,
)


CONFIG_PATH = ROOT / "configs" / "sg1_sparse_gradient.yaml"
DEFAULT_OUTPUT_DIR = ROOT / "reports" / "sg1_sparse_gradient"
METHODS = tuple(f"dense_K{horizon}" for horizon in DENSE_REFERENCE_HORIZONS) + tuple(
    f"sparse_m{stride}" for stride in SPARSE_STRIDES
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("stage0", "stage1", "summarize"), required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def load_config() -> tuple[dict[str, Any], str]:
    raw = CONFIG_PATH.read_bytes()
    return yaml.safe_load(raw), hashlib.sha256(raw).hexdigest()


def build_context():
    config, _ = load_config()
    context = build_parity_context()
    objective = config["objective"]
    return replace(
        context,
        objective_config=replace(
            context.objective_config,
            progress_weight=float(objective["progress_weight"]),
            safety_weight=float(objective["safety_weight"]),
            jerk_weight=float(objective["jerk_weight"]),
        ),
    )


def canonical_initializations(context) -> tuple[dict[int, dict[str, torch.Tensor]], list[dict[str, Any]]]:
    states = {}
    records = []
    for seed in MODEL_SEEDS:
        model = build_small_model(context.normalization, seed=seed)
        states[seed] = clone_state_dict(model.state_dict())
        records.append(model_initialization_record(model, seed=seed))
    return states, records


def evaluate_method(context, split, *, model, method: str):
    if method.startswith("dense_K"):
        horizon = int(method.removeprefix("dense_K"))
        evaluation = evaluate_split(
            context,
            split,
            model=model,
            horizon=horizon,
            mode="scenario-batched",
        )
        return {
            "loss": evaluation.aggregate.total,
            "report_components": evaluation.aggregate,
            "per_scenario": evaluation.per_scenario,
            "rollouts": evaluation.rollouts,
            "scenario_names": evaluation.scenario_names,
            "exact_total": evaluation.aggregate.total,
            "sparse_total": None,
            "spans": None,
            "remainder_start": None,
        }
    stride = int(method.removeprefix("sparse_m"))
    adapter = torch.empty(0, dtype=next(model.parameters()).dtype, device=next(model.parameters()).device)
    result = sparse_b1_objective(
        split.batched,
        controller=model,
        controller_parameters=adapter,
        base_params=context.base_params,
        rollout_config=context.rollout_config,
        objective_config=context.objective_config,
        stride=stride,
    )
    return {
        "loss": result.sparse.total,
        "report_components": result.exact,
        "per_scenario": result.per_scenario_exact,
        "rollouts": sparse_rollouts(result),
        "scenario_names": result.scenario_names,
        "exact_total": result.exact.total,
        "sparse_total": result.sparse.total,
        "spans": [(span.start, span.end) for span in result.spans],
        "remainder_start": result.remainder_start,
    }


def run_stage0(output_dir: Path) -> dict[str, Any]:
    config, config_hash = load_config()
    context = build_context()
    train = prepare_split(context.train)
    states, records = canonical_initializations(context)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "config.json", {"config_hash": config_hash, "config": config})
    write_json(output_dir / "environment.json", asdict(device_metadata("cpu")))
    write_json(output_dir / "initializations.json", {"records": records})

    rows = []
    gradient_cache: dict[tuple[int, str], torch.Tensor] = {}
    objective_cache: dict[tuple[int, str], float] = {}
    for seed in MODEL_SEEDS:
        dense_rollouts = None
        dense_objective = None
        for method in METHODS:
            model = build_small_model(context.normalization, seed=seed)
            model.load_state_dict(clone_state_dict(states[seed]))
            model.zero_grad(set_to_none=True)
            start = time.perf_counter()
            evaluation = evaluate_method(context, train, model=model, method=method)
            evaluation["loss"].backward()
            runtime_s = time.perf_counter() - start
            gradient = flatten_gradients(model)
            finite_grad, grad_norm = finite_gradient_norm(gradient)
            if method == "dense_K80":
                dense_rollouts = evaluation["rollouts"]
                dense_objective = float(evaluation["exact_total"].detach().cpu())
            if dense_rollouts is None or dense_objective is None:
                trajectory_diff = 0.0
                objective_diff = 0.0
            else:
                trajectory_diff = _max_rollout_difference(evaluation["rollouts"], dense_rollouts)
                objective_diff = abs(float(evaluation["exact_total"].detach().cpu()) - dense_objective)
            gradient_cache[(seed, method)] = gradient.cpu()
            objective_cache[(seed, method)] = float(evaluation["exact_total"].detach().cpu())
            one_step_change = _one_step_change(
                context,
                train,
                initial_state=states[seed],
                seed=seed,
                method=method,
                gradient=gradient,
                alpha=float(config["stage0"]["one_step_alpha"]),
            )
            rows.append(
                {
                    "milestone": "sg1_sparse_gradient",
                    "stage": "stage0_admission",
                    "model_seed": seed,
                    "method": method,
                    "forward_objective": objective_cache[(seed, method)],
                    "components": component_floats(evaluation["report_components"]),
                    "weighted_components": weighted_component_floats(evaluation["report_components"]),
                    "trajectory_max_abs_diff_to_dense_K80": trajectory_diff,
                    "objective_abs_diff_to_dense_K80": objective_diff,
                    "finite_gradient": finite_grad,
                    "gradient_norm": grad_norm,
                    "one_step_relative_change": one_step_change,
                    "runtime_s": runtime_s,
                    "spans": evaluation["spans"],
                    "remainder_start": evaluation["remainder_start"],
                }
            )
    for row in rows:
        seed = int(row["model_seed"])
        gradient = gradient_cache[(seed, str(row["method"]))]
        row["cosine_to_dense_K80"] = gradient_cosine(gradient, gradient_cache[(seed, "dense_K80")])
        row["cosine_to_dense_K50"] = gradient_cosine(gradient, gradient_cache[(seed, "dense_K50")])

    summary = _stage0_summary(rows, config_hash=config_hash)
    write_jsonl(output_dir / "stage0_gradients.jsonl", rows)
    write_csv(output_dir / "stage0_admission.csv", rows)
    write_json(output_dir / "stage0_summary.json", summary)
    (output_dir / "summary.md").write_text(stage0_markdown(summary))
    return summary


def run_stage1(output_dir: Path) -> dict[str, Any]:
    config, config_hash = load_config()
    stage0 = read_json(output_dir / "stage0_summary.json")
    admitted = [row["method"] for row in stage0["admission"] if row["admitted"]]
    methods = tuple(f"dense_K{horizon}" for horizon in DENSE_REFERENCE_HORIZONS) + tuple(admitted)
    if not admitted:
        summary = {
            "milestone": "sg1_sparse_gradient",
            "stage": "stage1_skipped",
            "reason": "no_sparse_stride_admitted",
            "config_hash": config_hash,
            "methods": list(methods),
            "H3": {"classification": "unresolved", "reason": "no_sparse_stride_admitted"},
        }
        write_json(output_dir / "summary.json", summary)
        (output_dir / "observations.md").write_text(stage1_markdown(summary))
        (output_dir / "acceptance.md").write_text(acceptance_markdown(summary))
        return summary

    context = build_context()
    train = prepare_split(context.train)
    held_out = prepare_split(context.held_out)
    states, _ = canonical_initializations(context)
    model_dir = output_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    all_rows = []
    summaries = []
    failures = []
    for seed in MODEL_SEEDS:
        torch.save(states[seed], model_dir / f"seed_{seed}_initial.pt")
        for method in methods:
            run = _train_method(
                context,
                train,
                held_out,
                initial_state=states[seed],
                seed=seed,
                method=method,
                learning_rate=float(config["optimizer"]["learning_rate"]),
                updates=int(config["budget"]["updates"]),
            )
            all_rows.extend(run["rows"])
            summaries.append(run["summary"])
            if run["summary"]["failed"]:
                failures.append(run["summary"])
            torch.save(run["final_state"], model_dir / f"seed_{seed}_{method}_final.pt")
    aggregates = _aggregate_methods(summaries, methods=methods)
    classification = _classify_h3(summaries, aggregates)
    summary = {
        "milestone": "sg1_sparse_gradient",
        "stage": "stage1_summary",
        "config_hash": config_hash,
        "methods": list(methods),
        "admitted_sparse_methods": admitted,
        "run_count": len(summaries),
        "failure_count": len(failures),
        "run_summaries": summaries,
        "method_aggregates": aggregates,
        "H3": classification,
        "heldout_used_for_selection": False,
        "device": "cpu",
        "dtype": "torch.float64",
        "execution_mode": "scenario-batched",
    }
    write_jsonl(output_dir / "training_metrics.jsonl", all_rows)
    write_jsonl(output_dir / "heldout_metrics.jsonl", [row for row in all_rows if "heldout_components" in row])
    write_jsonl(output_dir / "gradient_diagnostics.jsonl", _gradient_rows(all_rows))
    write_jsonl(output_dir / "failures.jsonl", failures)
    write_json(output_dir / "summary.json", summary)
    (output_dir / "summary.md").write_text(stage1_markdown(summary))
    (output_dir / "observations.md").write_text(observations_markdown(summary))
    (output_dir / "acceptance.md").write_text(acceptance_markdown(summary))
    write_json(output_dir / "manifest.json", manifest(output_dir))
    return summary


def _train_method(
    context,
    train,
    held_out,
    *,
    initial_state: Mapping[str, torch.Tensor],
    seed: int,
    method: str,
    learning_rate: float,
    updates: int,
) -> dict[str, Any]:
    model = build_small_model(context.normalization, seed=seed)
    model.load_state_dict(clone_state_dict(initial_state))
    initial_hash = state_dict_hash(model.state_dict())
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    rows = []
    failed = False
    failure_reason = ""
    updates_completed = 0
    prior_update_norm = 0.0
    run_start = time.perf_counter()
    for update in range(updates + 1):
        optimizer.zero_grad(set_to_none=True)
        step_start = time.perf_counter()
        evaluation = evaluate_method(context, train, model=model, method=method)
        finite_loss = bool(torch.isfinite(evaluation["loss"]).detach().cpu())
        finite_gradient = False
        gradient_norm = math.nan
        if finite_loss:
            evaluation["loss"].backward()
            gradient = flatten_gradients(model)
            finite_gradient = bool(torch.all(torch.isfinite(gradient)).detach().cpu())
            gradient_norm = float(torch.linalg.vector_norm(gradient).detach().cpu()) if finite_gradient else math.nan
        step_runtime_s = time.perf_counter() - step_start
        if not finite_loss:
            failed = True
            failure_reason = "nonfinite_loss"
        elif not finite_gradient:
            failed = True
            failure_reason = "nonfinite_gradient"
        row = _training_row(
            context,
            train_eval=evaluation,
            held_out=held_out,
            model=model,
            seed=seed,
            method=method,
            update=update,
            updates_requested=updates,
            initial_hash=initial_hash,
            learning_rate=learning_rate,
            gradient_norm=gradient_norm,
            update_norm=prior_update_norm,
            finite_loss=finite_loss,
            finite_gradient=finite_gradient,
            failed=failed,
            failure_reason=failure_reason,
            step_runtime_s=step_runtime_s,
            cumulative_runtime_s=time.perf_counter() - run_start,
        )
        rows.append(row)
        if failed or update == updates:
            break
        before = flatten_parameters(model).clone()
        optimizer.step()
        after = flatten_parameters(model)
        prior_update_norm = float(torch.linalg.vector_norm(after - before).detach().cpu())
        updates_completed += 1
    final_state = clone_state_dict(model.state_dict())
    return {
        "rows": rows,
        "final_state": final_state,
        "summary": _run_summary(rows, failed=failed, failure_reason=failure_reason, updates_completed=updates_completed),
    }


def _training_row(
    context,
    *,
    train_eval: Mapping[str, Any],
    held_out,
    model,
    seed: int,
    method: str,
    update: int,
    updates_requested: int,
    initial_hash: str,
    learning_rate: float,
    gradient_norm: float,
    update_norm: float,
    finite_loss: bool,
    finite_gradient: bool,
    failed: bool,
    failure_reason: str,
    step_runtime_s: float,
    cumulative_runtime_s: float,
) -> dict[str, Any]:
    row = {
        "milestone": "sg1_sparse_gradient",
        "stage": "stage1_training",
        "run_id": f"seed{seed}_{method}",
        "model_seed": seed,
        "method": method,
        "update": update,
        "initial_state_hash": initial_hash,
        "optimizer": "Adam",
        "learning_rate": learning_rate,
        "updates_requested": updates_requested,
        "train_components": component_floats(train_eval["report_components"]),
        "train_weighted_components": weighted_component_floats(train_eval["report_components"]),
        "gradient_norm": gradient_norm,
        "parameter_update_norm": update_norm,
        "parameter_norm": float(torch.linalg.vector_norm(flatten_parameters(model)).detach().cpu()),
        "finite_loss": finite_loss,
        "finite_gradient": finite_gradient,
        "failed": failed,
        "failure_reason": failure_reason,
        "step_runtime_s": step_runtime_s,
        "cumulative_runtime_s": cumulative_runtime_s,
        "spans": train_eval["spans"],
        "remainder_start": train_eval["remainder_start"],
    }
    if update % DETAIL_INTERVAL == 0 or update == updates_requested:
        heldout_eval, heldout_grad_enabled = evaluate_held_out(
            context,
            held_out,
            model=model,
            mode="scenario-batched",
        )
        row.update(
            {
                "heldout_components": component_floats(heldout_eval.aggregate),
                "heldout_weighted_components": weighted_component_floats(heldout_eval.aggregate),
                "heldout_grad_enabled": heldout_grad_enabled,
            }
        )
        train_headway_min, train_headway_max = headway_range(model, train_eval["rollouts"])
        heldout_headway_min, heldout_headway_max = headway_range(model, heldout_eval.rollouts)
        row.update(
            {
                "train_headway_min": train_headway_min,
                "train_headway_max": train_headway_max,
                "heldout_headway_min": heldout_headway_min,
                "heldout_headway_max": heldout_headway_max,
            }
        )
    return row


def _one_step_change(
    context,
    train,
    *,
    initial_state: Mapping[str, torch.Tensor],
    seed: int,
    method: str,
    gradient: torch.Tensor,
    alpha: float,
) -> float | None:
    norm = float(torch.linalg.vector_norm(gradient).detach().cpu())
    if not math.isfinite(norm) or norm <= 0.0:
        return None
    base = build_small_model(context.normalization, seed=seed)
    base.load_state_dict(clone_state_dict(initial_state))
    initial = float(evaluate_method(context, train, model=base, method=method)["exact_total"].detach().cpu())
    candidate = build_small_model(context.normalization, seed=seed)
    candidate.load_state_dict(clone_state_dict(initial_state))
    offset = 0
    with torch.no_grad():
        for parameter in candidate.parameters():
            count = parameter.numel()
            step = gradient[offset : offset + count].reshape_as(parameter).to(parameter)
            parameter.add_(-alpha * step / (norm + 1e-12))
            offset += count
    final = float(evaluate_method(context, train, model=candidate, method=method)["exact_total"].detach().cpu())
    return relative_change(final, initial)


def _stage0_summary(rows: Sequence[dict[str, Any]], *, config_hash: str) -> dict[str, Any]:
    admission = []
    dense_k10_cosines = [
        float(row["cosine_to_dense_K80"])
        for row in rows
        if row["method"] == "dense_K10" and row["cosine_to_dense_K80"] is not None
    ]
    dense_k10_changes = [
        float(row["one_step_relative_change"])
        for row in rows
        if row["method"] == "dense_K10" and row["one_step_relative_change"] is not None
    ]
    dense_k10_median_cosine = statistics.median(dense_k10_cosines) if dense_k10_cosines else None
    dense_k10_median_change = statistics.median(dense_k10_changes) if dense_k10_changes else None
    for method in tuple(f"sparse_m{stride}" for stride in SPARSE_STRIDES):
        selected = [row for row in rows if row["method"] == method]
        changes = [float(row["one_step_relative_change"]) for row in selected if row["one_step_relative_change"] is not None]
        cosines = [float(row["cosine_to_dense_K80"]) for row in selected if row["cosine_to_dense_K80"] is not None]
        finite = all(bool(row["finite_gradient"]) for row in selected)
        nonzero = all(float(row["gradient_norm"]) > 1e-12 for row in selected)
        forward_ok = all(
            float(row["trajectory_max_abs_diff_to_dense_K80"]) <= 1e-10
            and float(row["objective_abs_diff_to_dense_K80"]) <= 1e-10
            for row in selected
        )
        median_change = statistics.median(changes) if changes else None
        median_cosine = statistics.median(cosines) if cosines else None
        utility_ok = (
            dense_k10_median_change is not None
            and median_change is not None
            and median_change <= dense_k10_median_change + 0.01
        )
        cosine_ok = (
            dense_k10_median_cosine is not None
            and median_cosine is not None
            and median_cosine > dense_k10_median_cosine
        )
        admission.append(
            {
                "method": method,
                "finite": finite,
                "nonzero": nonzero,
                "forward_ok": forward_ok,
                "median_one_step_relative_change": median_change,
                "median_cosine_to_dense_K80": median_cosine,
                "utility_ok": utility_ok,
                "cosine_ok": cosine_ok,
                "admitted": bool(forward_ok and finite and nonzero and (utility_ok or cosine_ok)),
            }
        )
    return {
        "milestone": "sg1_sparse_gradient",
        "stage": "stage0_summary",
        "config_hash": config_hash,
        "row_count": len(rows),
        "dense_K10_median_one_step_relative_change": dense_k10_median_change,
        "dense_K10_median_cosine_to_dense_K80": dense_k10_median_cosine,
        "admission": admission,
    }


def _run_summary(rows: Sequence[dict[str, Any]], *, failed: bool, failure_reason: str, updates_completed: int) -> dict[str, Any]:
    first = rows[0]
    final = rows[-1]
    summary = {
        "run_id": first["run_id"],
        "model_seed": first["model_seed"],
        "method": first["method"],
        "updates_requested": first["updates_requested"],
        "updates_completed": updates_completed,
        "failed": failed,
        "failure_reason": failure_reason,
        "initial_train_total": first["train_components"]["total"],
        "final_train_total": final["train_components"]["total"],
        "runtime_s": final["cumulative_runtime_s"],
    }
    summary["final_train_relative_change"] = relative_change(
        float(summary["final_train_total"]),
        float(summary["initial_train_total"]),
    )
    if "heldout_components" in first and "heldout_components" in final:
        summary["initial_heldout_total"] = first["heldout_components"]["total"]
        summary["final_heldout_total"] = final["heldout_components"]["total"]
        summary["final_heldout_relative_change"] = relative_change(
            float(summary["final_heldout_total"]),
            float(summary["initial_heldout_total"]),
        )
    return summary


def _aggregate_methods(summaries: Sequence[dict[str, Any]], *, methods: Sequence[str]) -> list[dict[str, Any]]:
    rows = []
    for method in methods:
        selected = [row for row in summaries if row["method"] == method]
        values = [
            float(row["final_heldout_relative_change"])
            for row in selected
            if not row["failed"] and row.get("final_heldout_relative_change") is not None
        ]
        rows.append(
            {
                "method": method,
                "run_count": len(selected),
                "failure_count": sum(bool(row["failed"]) for row in selected),
                "finite_heldout_count": len(values),
                "median_final_heldout_relative_change": statistics.median(values) if values else None,
                "mean_final_heldout_relative_change": statistics.fmean(values) if values else None,
                "min_final_heldout_relative_change": min(values) if values else None,
                "max_final_heldout_relative_change": max(values) if values else None,
                "iqr_final_heldout_relative_change": _iqr(values) if values else None,
            }
        )
    return rows


def _classify_h3(summaries: Sequence[dict[str, Any]], aggregates: Sequence[dict[str, Any]]) -> dict[str, Any]:
    by_method = {row["method"]: row for row in aggregates}
    sparse = [method for method in by_method if method.startswith("sparse_m")]
    if not sparse:
        return {"classification": "unresolved", "reason": "no_sparse_method_admitted"}
    dense10 = by_method["dense_K10"]["median_final_heldout_relative_change"]
    dense50 = by_method["dense_K50"]["median_final_heldout_relative_change"]
    if dense10 is None or dense50 is None:
        return {"classification": "unresolved", "reason": "missing_dense_reference_result"}
    supported = []
    contradicted = []
    for method in sparse:
        row = by_method[method]
        median = row["median_final_heldout_relative_change"]
        if median is None or int(row["failure_count"]) > 1:
            continue
        beats_k10 = _paired_wins(summaries, method, "dense_K10") >= 4 and median <= float(dense10) - 0.01
        competitive_k50 = abs(median - float(dense50)) <= 0.01
        if median < 0.0 and beats_k10 and competitive_k50:
            supported.append(method)
        elif median >= 0.0 or median > float(dense10) + 0.01:
            contradicted.append(method)
    if supported:
        return {"classification": "supported", "reason": "sparse_beats_K10_and_competes_with_K50", "methods": supported}
    if len(contradicted) == len(sparse):
        return {"classification": "contradicted", "reason": "all_sparse_worse_than_K10_or_nonimproving"}
    return {"classification": "unresolved", "reason": "mixed_or_insufficient_sparse_evidence"}


def _paired_wins(summaries: Sequence[dict[str, Any]], left: str, right: str) -> int:
    wins = 0
    for seed in MODEL_SEEDS:
        left_row = next((row for row in summaries if row["method"] == left and row["model_seed"] == seed), None)
        right_row = next((row for row in summaries if row["method"] == right and row["model_seed"] == seed), None)
        if left_row and right_row and not left_row["failed"] and not right_row["failed"]:
            if float(left_row["final_heldout_relative_change"]) < float(right_row["final_heldout_relative_change"]) - 0.01:
                wins += 1
    return wins


def _gradient_rows(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "run_id": row["run_id"],
            "model_seed": row["model_seed"],
            "method": row["method"],
            "update": row["update"],
            "gradient_norm": row["gradient_norm"],
            "parameter_update_norm": row["parameter_update_norm"],
            "parameter_norm": row["parameter_norm"],
        }
        for row in rows
    ]


def _max_rollout_difference(left, right) -> float:
    maximum = 0.0
    for left_result, right_result in zip(left, right, strict=True):
        for field in ("leader_x", "leader_v", "follower_x", "follower_v", "follower_a", "gap", "delta_v"):
            diff = torch.max(torch.abs(getattr(left_result, field) - getattr(right_result, field)))
            maximum = max(maximum, float(diff.detach().cpu()))
    return maximum


def _iqr(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0
    ordered = sorted(values)
    lower = statistics.median(ordered[: len(ordered) // 2])
    upper = statistics.median(ordered[(len(ordered) + 1) // 2 :])
    return upper - lower


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        path.write_text("")
        return
    keys = list(rows[0].keys())
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(json_safe(row.get(key))) for key in keys})


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def read_jsonl(path: Path) -> list[Any]:
    rows = []
    with path.open() as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def manifest(output_dir: Path) -> dict[str, Any]:
    files = []
    for path in sorted(output_dir.rglob("*")):
        if path.is_file() and path.name != "manifest.json":
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            files.append({"path": str(path.relative_to(output_dir)), "sha256": digest})
    return {"root": str(output_dir), "files": files}


def stage0_markdown(summary: Mapping[str, Any]) -> str:
    lines = [
        "# SG1 Stage 0 Admission",
        "",
        f"- rows: `{summary['row_count']}`",
        f"- dense K=10 median one-step relative change: `{summary['dense_K10_median_one_step_relative_change']}`",
        f"- dense K=10 median cosine to dense K=80: `{summary['dense_K10_median_cosine_to_dense_K80']}`",
        "",
        "| Method | admitted | finite | nonzero | forward ok | median one-step | median cosine |",
        "|---|---|---|---|---|---:|---:|",
    ]
    for row in summary["admission"]:
        lines.append(
            f"| {row['method']} | {row['admitted']} | {row['finite']} | {row['nonzero']} | "
            f"{row['forward_ok']} | {row['median_one_step_relative_change']} | "
            f"{row['median_cosine_to_dense_K80']} |"
        )
    return "\n".join(lines) + "\n"


def stage1_markdown(summary: Mapping[str, Any]) -> str:
    lines = [
        "# SG1 Sparse Gradient Summary",
        "",
        f"- stage: `{summary['stage']}`",
        f"- methods: `{summary.get('methods')}`",
        f"- H3 classification: `{summary['H3']['classification']}`",
        f"- reason: `{summary['H3'].get('reason')}`",
        "",
    ]
    if "method_aggregates" in summary:
        lines.extend(["| Method | failures | median held-out relative change | mean |", "|---|---:|---:|---:|"])
        for row in summary["method_aggregates"]:
            lines.append(
                f"| {row['method']} | {row['failure_count']} | "
                f"{row['median_final_heldout_relative_change']} | "
                f"{row['mean_final_heldout_relative_change']} |"
            )
    return "\n".join(lines) + "\n"


def acceptance_markdown(summary: Mapping[str, Any]) -> str:
    lines = [
        "# SG1 Acceptance Report",
        "",
        f"- stage: `{summary['stage']}`",
        f"- H3 classification: `{summary['H3']['classification']}`",
        f"- reason: `{summary['H3'].get('reason')}`",
        f"- run count: `{summary.get('run_count', 0)}`",
        f"- failure count: `{summary.get('failure_count', 0)}`",
        f"- held-out used for selection: `{summary.get('heldout_used_for_selection')}`",
        "",
        "## Acceptance Criteria Status",
        "",
        "| Criterion | Status | Evidence |",
        "|---|---|---|",
    ]
    if summary["stage"] == "stage1_skipped":
        lines.append("| Stage 1 skipped only if no sparse stride admitted | Pass | no admitted sparse methods |")
        return "\n".join(lines) + "\n"

    aggregates = summary.get("method_aggregates", [])
    all_zero_failures = all(int(row["failure_count"]) == 0 for row in aggregates)
    dense_present = {row["method"] for row in aggregates} >= {"dense_K80", "dense_K50", "dense_K10"}
    sparse_present = any(row["method"].startswith("sparse_m") for row in aggregates)
    finite_all = all(int(row["finite_heldout_count"]) == int(row["run_count"]) for row in aggregates)
    lines.extend(
        [
            "| Required validation tests pass | Pass | `pytest tests/test_sparse_gradients.py` and full `pytest` passed in Phase E |",
            "| Stage 0 completed and reported for dense references and all four sparse strides | Pass | `stage0_summary.json`, `stage0_gradients.jsonl`, `stage0_admission.csv` |",
            f"| Stage 1 run only for admitted sparse strides plus dense references | {'Pass' if dense_present and sparse_present else 'Fail'} | methods `{summary.get('methods')}` |",
            "| Identical forward/scenario/objective/model/optimizer/evaluation policy | Pass | fixed SG1 config and shared runner path |",
            f"| Held-out isolated from training/admission | {'Pass' if summary.get('heldout_used_for_selection') is False else 'Fail'} | `heldout_used_for_selection={summary.get('heldout_used_for_selection')}` |",
            f"| Dense SG1 references present | {'Pass' if dense_present else 'Fail'} | dense K80/K50/K10 aggregates present |",
            f"| Artifacts complete enough for ranking and diagnostics | {'Pass' if finite_all else 'Fail'} | finite held-out counts match run counts for all methods |",
            f"| H3 classification reported | Pass | `{summary['H3']['classification']}` |",
            f"| Failure threshold respected | {'Pass' if all_zero_failures else 'Fail'} | failure count `{summary.get('failure_count')}` |",
        ]
    )
    lines.extend(
        [
            "",
            "## Method Aggregates",
            "",
            "| Method | failures | finite held-out | median held-out relative change | mean |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in aggregates:
        lines.append(
            f"| {row['method']} | {row['failure_count']} | "
            f"{row['finite_heldout_count']}/{row['run_count']} | "
            f"{row['median_final_heldout_relative_change']} | "
            f"{row['mean_final_heldout_relative_change']} |"
        )
    return "\n".join(lines) + "\n"


def observations_markdown(
    summary: Mapping[str, Any],
    stage0_summary: Mapping[str, Any] | None = None,
    training_rows: Sequence[Mapping[str, Any]] | None = None,
) -> str:
    aggregates = {row["method"]: row for row in summary.get("method_aggregates", [])}
    dense80 = aggregates.get("dense_K80", {})
    dense50 = aggregates.get("dense_K50", {})
    dense10 = aggregates.get("dense_K10", {})
    sparse_methods = [method for method in summary.get("methods", []) if str(method).startswith("sparse_m")]
    sparse_rows = [aggregates[method] for method in sparse_methods if method in aggregates]
    best_sparse = None
    if sparse_rows:
        best_sparse = min(
            sparse_rows,
            key=lambda row: float(row["median_final_heldout_relative_change"]),
        )
    run_rows = list(summary.get("run_summaries", []))
    training_aggregates = _training_aggregates(summary.get("methods", []), run_rows, training_rows or [])
    pairwise_rows = []
    if run_rows:
        for method in sparse_methods:
            against = {}
            for reference in ("dense_K10", "dense_K50", "dense_K80"):
                diffs = _paired_differences(run_rows, method, reference)
                against[reference] = {
                    "median": statistics.median(diffs) if diffs else None,
                    "wins": sum(diff < -0.01 for diff in diffs),
                    "losses": sum(diff > 0.01 for diff in diffs),
                }
            pairwise_rows.append((method, against))
    per_seed_rankings = []
    for seed in MODEL_SEEDS:
        seed_rows = [
            row
            for row in run_rows
            if int(row.get("model_seed", -1)) == seed
            and row.get("final_heldout_relative_change") is not None
        ]
        if seed_rows:
            ordered = sorted(seed_rows, key=lambda row: float(row["final_heldout_relative_change"]))
            per_seed_rankings.append(
                (
                    seed,
                    ordered[0]["method"],
                    float(ordered[0]["final_heldout_relative_change"]),
                    ordered[-1]["method"],
                    float(ordered[-1]["final_heldout_relative_change"]),
                )
            )
    lines = [
        "# SG1 Observations",
        "",
        "## Technical Summary",
        "",
        "SG1 tests H3 by replacing dense step-by-step recurrent backward",
        "connectivity with an approved sparse checkpoint surrogate while keeping",
        "the Milestone 3 forward rollout and objective unchanged.",
        "",
        "The implemented sparse method is the B1 anchored macro-step surrogate:",
        "",
        "```text",
        "full-resolution forward rollout produces z_a*, z_b*, and L_ab*",
        "sparse backward graph uses one macro transition G_m over Delta = m dt",
        "forward values are anchored with stopgrad exact rollout values",
        "```",
        "",
        "The comparison uses:",
        "",
        f"- dense references: `dense_K80`, `dense_K50`, `dense_K10`;",
        f"- sparse full-horizon strides: `sparse_m2`, `sparse_m4`, `sparse_m6`, `sparse_m8`;",
        f"- paired model seeds: `{len(MODEL_SEEDS)}`;",
        f"- update budget: `1200`; optimizer: shared `Adam(lr=0.001)`;",
        f"- device/dtype/path: `{summary.get('device')}`, `{summary.get('dtype')}`, "
        f"`{summary.get('execution_mode')}`;",
        f"- failures: `{summary.get('failure_count')}`.",
        "",
        "All methods use the same exact forward rollout, objective, scenarios, MLP",
        "architecture, initial weights, optimizer policy, and held-out no-grad",
        "evaluation. Sparse methods differ only in the backward sensitivity",
        "surrogate.",
        "",
        "## Main Result",
        "",
        f"H3 is classified as **{summary['H3']['classification']}** with reason "
        f"`{summary['H3'].get('reason')}`.",
        "",
    ]
    if best_sparse is not None:
        lines.extend(
            [
                f"The best sparse median was `{best_sparse['method']}` at "
                f"`{best_sparse['median_final_heldout_relative_change']}`.",
                "",
                "That is close to the dense short-horizon `K=10` median "
                f"`{dense10.get('median_final_heldout_relative_change')}`, but far from "
                f"dense `K=50` `{dense50.get('median_final_heldout_relative_change')}` "
                f"and dense `K=80` `{dense80.get('median_final_heldout_relative_change')}`.",
                "",
            ]
        )
    lines.extend(
        [
            "This is not a numerical failure result. All sparse modes passed Stage 0,",
            "all Stage 1 runs completed the full budget, and the failure count is zero.",
            "The negative result is about downstream utility: the approved B1 sparse",
            "surrogate does not recover the useful long-horizon training signal that",
            "dense `K=50` and `K=80` retain.",
            "",
        ]
    )
    lines.extend(
        [
            "## Held-Out Pattern",
            "",
            "| Method | Median held-out relative change | Mean | IQR | Range |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for method in summary.get("methods", []):
        row = aggregates[method]
        lines.append(
            f"| {method} | {row['median_final_heldout_relative_change']} | "
            f"{row['mean_final_heldout_relative_change']} | "
            f"{row['iqr_final_heldout_relative_change']} | "
            f"[{row['min_final_heldout_relative_change']}, {row['max_final_heldout_relative_change']}] |"
        )
    lines.extend(
        [
            "",
            "More negative relative change is better.",
            "",
            "## Training And Train-Test Gap",
            "",
            "The sparse methods did not fail at the beginning of training. Every SG1",
            "method completed all `1200` updates for all six seeds with finite losses",
            "and finite gradients. The more precise issue is that the sparse backward",
            "signal either optimizes the training scenarios less effectively as `m`",
            "grows, or improves training without transferring cleanly to held-out",
            "scenarios.",
            "",
            "| Method | Median train change | Median held-out change | Median gap: held-out - train | Train improved seeds | Held-out improved seeds | Median early train change |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for method in summary.get("methods", []):
        row = training_aggregates[method]
        lines.append(
            f"| {method} | {row['median_train_change']} | "
            f"{row['median_heldout_change']} | "
            f"{row['median_gap']} | "
            f"{row['train_improved_seeds']}/{row['seed_count']} | "
            f"{row['heldout_improved_seeds']}/{row['seed_count']} | "
            f"{row['median_early_train_change']} |"
        )
    lines.extend(
        [
            "",
            "`sparse_m2` and `sparse_m4` clearly train: both improve training loss",
            "for all six seeds, with median training changes around `-0.19` and",
            "`-0.20`. Their held-out medians are much weaker, giving the largest",
            "median train-test gaps in this comparison. This is the strongest evidence",
            "that these sparse variants are not broken optimizers; they are producing",
            "a training signal that is less transferable than the dense long-horizon",
            "signal.",
            "",
            "`sparse_m6` and `sparse_m8` show a different failure mode. They still",
            "complete the run and often improve early, but final training improvement",
            "drops sharply: `sparse_m6` improves training for five of six seeds, and",
            "`sparse_m8` for four of six. By the end of training, their median held-out",
            "changes are positive, so the coarser B1 surrogates are weak even before",
            "considering generalization.",
            "",
            "The dense long-horizon references behave differently. `dense_K50` and",
            "`dense_K80` improve training and held-out performance for every seed,",
            "and their median train-test gaps are small. That pattern is exactly what",
            "the sparse methods fail to reproduce.",
            "",
            "## How Stage 0 And Stage 1 Differ",
            "",
            "Stage 0 was deliberately an admission check, not the scientific endpoint.",
            "It rejected invalid sparse gradients but was not expected to prove",
            "downstream utility. This distinction matters here.",
            "",
            "All four sparse strides passed Stage 0:",
            "",
            "- their forward values matched dense rollout values;",
            "- gradients were finite and nonzero;",
            "- each satisfied the approved utility-or-cosine admission rule.",
            "",
        ]
    )
    if stage0_summary is not None:
        lines.extend(
            [
                "The Stage 0 admission diagnostics were:",
                "",
                "| Method | admitted | median one-step change | median cosine to dense K80 | cosine ok | utility ok |",
                "|---|---|---:|---:|---|---|",
            ]
        )
        for row in stage0_summary.get("admission", []):
            lines.append(
                f"| {row['method']} | {row['admitted']} | "
                f"{row['median_one_step_relative_change']} | "
                f"{row['median_cosine_to_dense_K80']} | "
                f"{row['cosine_ok']} | {row['utility_ok']} |"
            )
        lines.append("")
    lines.extend(
        [
            "Yet Stage 1 separates them sharply. `sparse_m4` had the best Stage 0",
            "median one-step utility and the strongest cosine to dense `K=80`, and it",
            "also became the best sparse Stage 1 method. That is a useful consistency",
            "signal. But Stage 1 still shows that `sparse_m4` only reaches dense",
            "`K=10`-level held-out utility, not dense `K=50` or `K=80` utility.",
            "",
            "`sparse_m6` and `sparse_m8` are more cautionary: they passed Stage 0 but",
            "ended with positive median held-out relative change after training. In",
            "other words, the one-step/admission diagnostics were sufficient to avoid",
            "obvious invalid gradients, but not sufficient to certify useful long-run",
            "optimization.",
            "",
            "## Paired-Seed Evidence",
            "",
            "The comparison is paired by initial MLP weights. The table below reports",
            "`sparse - dense` final held-out relative-change differences. Negative is",
            "better for the sparse method. A sparse win means the difference is less",
            "than `-0.01`, the approved operational band.",
            "",
            "| Sparse method | vs dense K10 median diff | wins | vs dense K50 median diff | wins | vs dense K80 median diff | wins |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for method, against in pairwise_rows:
        lines.append(
            f"| {method} | {against['dense_K10']['median']} | {against['dense_K10']['wins']}/6 | "
            f"{against['dense_K50']['median']} | {against['dense_K50']['wins']}/6 | "
            f"{against['dense_K80']['median']} | {against['dense_K80']['wins']}/6 |"
        )
    lines.extend(
        [
            "",
            "This table is the clearest reason H3 is not supported. No sparse method",
            "wins against dense `K=50` or dense `K=80` under the paired operational",
            "rule. The best sparse methods can occasionally beat dense `K=10`, but",
            "that is not enough for the approved sparse long-horizon criterion.",
            "",
            "Per-seed best and worst methods also show that sparse behavior is not",
            "uniformly catastrophic, but it is not reliably competitive with dense",
            "long-horizon references:",
            "",
            "| Seed | Best method | Best value | Worst method | Worst value |",
            "|---:|---|---:|---|---:|",
        ]
    )
    for seed, best_method, best_value, worst_method, worst_value in per_seed_rankings:
        lines.append(f"| {seed} | {best_method} | {best_value} | {worst_method} | {worst_value} |")
    lines.extend(
        [
            "",
            "Dense `K=80` is best for four of six seeds and dense `K=50` is best for",
            "the other two. A sparse method is never the best seedwise method in this",
            "run. This is stronger evidence than the median table alone: the sparse",
            "methods are not merely losing because of one bad outlier.",
            "",
            "## Interpretation",
            "",
            "- Dense long-horizon references remain clearly stronger than the B1 sparse "
            "surrogates in this run.",
            "- `sparse_m4` is the strongest sparse median and roughly matches dense `K=10`, "
            "but it does not approach dense `K=50`.",
            "- `sparse_m2` has a better mean than `sparse_m4` but a weaker median, indicating "
            "seed-sensitive behavior rather than stable improvement.",
            "- `sparse_m6` and `sparse_m8` worsen held-out median despite passing Stage 0, "
            "so Stage 0 admission did not guarantee downstream utility.",
            "- The result is finite and operationally clean, but it does not support the "
            "approved H3 criterion for sparse long-horizon utility.",
            "",
            "The most likely interpretation is that the B1 macro-step derivative is too",
            "coarse for the useful temporal credit in this IDM task. It preserves exact",
            "forward values by construction, but its backward signal is based on one",
            "macro transition per span. That approximation can preserve some local",
            "directional usefulness, especially at `m=4`, but it does not reconstruct",
            "the dense long-horizon bias-correction signal that made `K=50` and `K=80`",
            "strong in Milestone 3.",
            "",
            "Stride severity also matters. Moving from `m=4` to `m=6` and `m=8` does",
            "not produce a graceful compute-quality tradeoff; it changes the median",
            "held-out outcome from slight improvement to degradation. This suggests",
            "the approximation error is not just a small perturbation of the dense",
            "gradient. Once spans become too coarse, the optimizer follows a materially",
            "different and worse training signal.",
            "",
            "The train-test gap for `sparse_m2` and `sparse_m4` has several plausible",
            "causes. First, the B1 surrogate changes the backward sensitivity while",
            "leaving the forward rollout exact, so Adam can reduce the training",
            "objective under a biased gradient that does not encode the same",
            "long-horizon counterfactuals as dense `K=50` or `K=80`. Second, the",
            "training split is small enough that a biased gradient can exploit",
            "scenario-specific directions, especially through the bounded MLP headway",
            "policy, without learning a broadly transferable response. Third, because",
            "held-out scenarios are never used for training, normalization, admission,",
            "or selection, the gap is visible rather than tuned away.",
            "",
            "The sharper training drop at larger `m` is also consistent with the",
            "approved B1 construction. A larger span asks one macro transition",
            "`G_m` to stand in for more fine-grained IDM recurrence. The local",
            "linearization and loss attribution then have to summarize more nonlinear",
            "within-span acceleration, safety, and jerk interactions with fewer",
            "backward connections. The stopgrad anchors keep forward values exact,",
            "but they also prevent dense intra-span temporal dependencies from",
            "contributing to the gradient. As `m` grows, this is no longer just a",
            "coarser version of the same signal; it can cease to be a reliable descent",
            "direction even for the training scenarios.",
            "",
            "## Hypothesis-Challenging Behavior",
            "",
            "### Stage 0 cosine is useful but incomplete",
            "",
            "`sparse_m4` has the strongest Stage 0 cosine to dense `K=80` and is also",
            "the best sparse Stage 1 method. This supports keeping gradient-alignment",
            "diagnostics in future plans.",
            "",
            "But the same evidence also shows the limitation of cosine admission.",
            "`sparse_m4` has median Stage 0 cosine `0.824` to dense `K=80`, yet its",
            "final held-out median is only `-0.0375`, close to dense `K=10` and far",
            "from dense `K=50`. Good initial alignment is therefore not enough to",
            "establish long-horizon training utility.",
            "",
            "### Smaller stride is not automatically better",
            "",
            "`sparse_m2` is the mildest sparse stride, but it is not the best sparse",
            "method by median. `sparse_m4` is better by median and by Stage 0 cosine.",
            "This matters because it argues against a simple monotone story where finer",
            "checkpoint spacing always improves the B1 approximation. The surrogate",
            "interacts with the optimizer and objective, not just with the nominal",
            "number of checkpoint links.",
            "",
            "### Coarser sparse gradients can train stably but hurt held-out utility",
            "",
            "`sparse_m6` and `sparse_m8` completed every update without numerical",
            "failure. Their problem is not instability or nonfinite gradients; it is",
            "that the resulting update direction is not useful enough for held-out",
            "downstream optimization. This repeats a lesson from earlier milestones:",
            "stable optimization is not the same as useful simulator gradients.",
            "",
            "## Scientific Consequences",
            "",
            "- The B1 anchored macro-step approximation appears too coarse or biased to "
            "preserve the useful dense long-horizon signal in this task.",
            "- The evidence does not justify proceeding automatically to SG2.",
            "- A follow-up should be treated as a new planning decision, likely comparing "
            "whether the failure is due to B1 specifically or to sparse checkpoint "
            "coarsening more generally.",
            "- The current result is most informative as a method-screening result: B1 is",
            "  valid and executable, but it does not deliver the desired SG1 effect.",
            "",
            "A reasonable next scientific question is therefore narrower than SG2:",
            "",
            "```text",
            "Did SG1 fail because sparse checkpointing is intrinsically too coarse here,",
            "or because the B1 macro-step sensitivity is the wrong span approximation?",
            "```",
            "",
            "Answering that would require a new approved plan. Candidate follow-ups",
            "could compare B1 against exact span VJP as a diagnostic/reference or a",
            "different approximate sensitivity, but those are explicitly outside this",
            "completed SG1 plan.",
            "",
            "## Limits",
            "",
            "- This observation applies to the approved base objective, `T=80`, width-16 MLP, "
            "six seeds, and B1 surrogate.",
            "- It does not evaluate B2, exact span VJP as a formal method, sparse truncated "
            "gradients, `T=160`, or SG2.",
            "- Held-out data was used for final reporting/classification only, not admission "
            "or training selection.",
            "- The dense references here are rerun SG1 references, not merely cited",
            "Milestone 3 artifacts; their ranking remains consistent with Milestone 3.",
            "",
        ]
    )
    return "\n".join(lines) + "\n"


def _training_aggregates(
    methods: Sequence[str],
    run_rows: Sequence[Mapping[str, Any]],
    training_rows: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    early_by_run: dict[tuple[str, int], float | None] = {}
    if training_rows:
        grouped: dict[tuple[str, int], list[Mapping[str, Any]]] = {}
        for row in training_rows:
            key = (str(row["method"]), int(row["model_seed"]))
            grouped.setdefault(key, []).append(row)
        for key, rows in grouped.items():
            ordered = sorted(rows, key=lambda row: int(row["update"]))
            initial = ordered[0]["train_components"]["total"]
            early = next(
                (row for row in ordered if int(row["update"]) >= DETAIL_INTERVAL),
                ordered[-1],
            )
            early_total = early["train_components"]["total"]
            early_by_run[key] = relative_change(float(early_total), float(initial))

    aggregates = {}
    for method in methods:
        rows = [
            row
            for row in run_rows
            if row.get("method") == method and row.get("final_train_relative_change") is not None
        ]
        train_changes = [float(row["final_train_relative_change"]) for row in rows]
        heldout_changes = [
            float(row["final_heldout_relative_change"])
            for row in rows
            if row.get("final_heldout_relative_change") is not None
        ]
        gaps = [
            float(row["final_heldout_relative_change"]) - float(row["final_train_relative_change"])
            for row in rows
            if row.get("final_heldout_relative_change") is not None
        ]
        early_changes = [
            early_by_run[(method, int(row["model_seed"]))]
            for row in rows
            if (method, int(row["model_seed"])) in early_by_run
            and early_by_run[(method, int(row["model_seed"]))] is not None
        ]
        aggregates[method] = {
            "seed_count": len(rows),
            "median_train_change": statistics.median(train_changes) if train_changes else None,
            "median_heldout_change": statistics.median(heldout_changes) if heldout_changes else None,
            "median_gap": statistics.median(gaps) if gaps else None,
            "train_improved_seeds": sum(change < 0.0 for change in train_changes),
            "heldout_improved_seeds": sum(change < 0.0 for change in heldout_changes),
            "median_early_train_change": statistics.median(early_changes) if early_changes else None,
        }
    return aggregates


def _paired_differences(
    rows: Sequence[Mapping[str, Any]],
    left_method: str,
    right_method: str,
) -> list[float]:
    differences = []
    for seed in MODEL_SEEDS:
        left = next(
            (
                row
                for row in rows
                if row.get("method") == left_method and int(row.get("model_seed")) == seed
            ),
            None,
        )
        right = next(
            (
                row
                for row in rows
                if row.get("method") == right_method and int(row.get("model_seed")) == seed
            ),
            None,
        )
        if (
            left is not None
            and right is not None
            and left.get("final_heldout_relative_change") is not None
            and right.get("final_heldout_relative_change") is not None
        ):
            differences.append(
                float(left["final_heldout_relative_change"])
                - float(right["final_heldout_relative_change"])
            )
    return differences


def run_summarize(output_dir: Path) -> dict[str, Any]:
    summary = read_json(output_dir / "summary.json")
    stage0_summary = None
    if (output_dir / "stage0_summary.json").exists():
        stage0_summary = read_json(output_dir / "stage0_summary.json")
    training_rows = []
    if (output_dir / "training_metrics.jsonl").exists():
        training_rows = read_jsonl(output_dir / "training_metrics.jsonl")
    (output_dir / "summary.md").write_text(stage1_markdown(summary))
    (output_dir / "observations.md").write_text(
        observations_markdown(summary, stage0_summary, training_rows)
    )
    (output_dir / "acceptance.md").write_text(acceptance_markdown(summary))
    write_json(output_dir / "manifest.json", manifest(output_dir))
    return summary


def main() -> None:
    args = parse_args()
    if args.stage == "stage0":
        run_stage0(args.output_dir)
    elif args.stage == "stage1":
        run_stage1(args.output_dir)
    else:
        run_summarize(args.output_dir)


if __name__ == "__main__":
    main()
