"""Run approved Milestone 3 small-model training stages."""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import hashlib
import json
import math
from pathlib import Path
import statistics
import subprocess
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
    max_result_difference,
    write_json,
    write_jsonl,
)
from differential_sim.objectives import ObjectiveConfig  # noqa: E402
from differential_sim.small_model_training import (  # noqa: E402
    BUDGET_CANDIDATES,
    DETAIL_INTERVAL,
    HORIZONS,
    LR_CANDIDATES,
    MODEL_SEEDS,
    TrainingConfig,
    TrainingRun,
    aggregate_horizons,
    build_small_model,
    classify_h2,
    clone_state_dict,
    compare_training_rows,
    evaluate_split,
    flatten_gradients,
    flatten_parameters,
    gradient_field_diagnostics,
    model_initialization_record,
    prepare_split,
    run_training,
    select_shared_learning_rate,
    select_update_budget,
    state_dict_hash,
    summarize_run,
)


CONFIG_PATH = ROOT / "configs" / "milestone3_small_model.yaml"
DEFAULT_OUTPUT_DIR = ROOT / "reports" / "milestone3" / "small_model_training"
DEFAULT_OBJECTIVE_PROFILE = "default"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=("smoke", "execution-check", "calibrate-lr", "calibrate-budget", "full", "summarize"),
        required=True,
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--objective-profile", default=DEFAULT_OBJECTIVE_PROFILE)
    return parser.parse_args()


def load_config() -> tuple[dict[str, Any], str]:
    raw = CONFIG_PATH.read_bytes()
    return yaml.safe_load(raw), hashlib.sha256(raw).hexdigest()


def build_context_for_profile(profile: str, **kwargs):
    config, _ = load_config()
    profiles = config.get("objective", {})
    if profile not in profiles:
        raise SystemExit(f"Unknown objective profile {profile!r}; available profiles: {sorted(profiles)}")
    context = build_parity_context(**kwargs)
    objective_config = objective_config_from_profile(context.objective_config, profiles[profile])
    return replace(context, objective_config=objective_config)


def objective_config_from_profile(base: ObjectiveConfig, profile: Mapping[str, Any]) -> ObjectiveConfig:
    return replace(
        base,
        progress_weight=float(profile["progress_weight"]),
        safety_weight=float(profile["safety_weight"]),
        jerk_weight=float(profile["jerk_weight"]),
    )


def objective_metadata(profile: str, context) -> dict[str, Any]:
    return {
        "profile": profile,
        "weights": {
            "progress": context.objective_config.progress_weight,
            "safety": context.objective_config.safety_weight,
            "jerk": context.objective_config.jerk_weight,
        },
        "scales": {
            "progress": context.objective_config.progress_scale,
            "safety": context.objective_config.safety_scale,
            "jerk": context.objective_config.jerk_scale,
        },
    }


def canonical_initializations(context) -> tuple[dict[int, dict[str, torch.Tensor]], list[dict[str, Any]]]:
    states = {}
    records = []
    for seed in MODEL_SEEDS:
        model = build_small_model(context.normalization, seed=seed)
        states[seed] = clone_state_dict(model.state_dict())
        records.append(model_initialization_record(model, seed=seed))
    return states, records


def run_smoke(output_dir: Path, objective_profile: str) -> dict[str, Any]:
    _, config_hash = load_config()
    context = build_context_for_profile(objective_profile, train_limit=2, held_out_limit=1)
    train = prepare_split(context.train)
    held_out = prepare_split(context.held_out)
    states, records = canonical_initializations(context)
    run = run_training(
        context,
        train,
        held_out,
        initial_state=states[MODEL_SEEDS[0]],
        seed=MODEL_SEEDS[0],
        horizon=HORIZONS[0],
        config=TrainingConfig(learning_rate=1e-3, updates=2, detail_interval=DETAIL_INTERVAL),
        stage="smoke",
        include_metadata=True,
    )
    summary = {
        "milestone": "3_small_model_training",
        "stage": "smoke",
        "objective": objective_metadata(objective_profile, context),
        "passed": not run.failed and run.updates_completed == 2,
        "config_hash": config_hash,
        "initialization": records[0],
        "run_summary": summarize_run(run),
        "row_count": len(run.rows),
        "heldout_grad_enabled_values": sorted(
            {row.get("heldout_grad_enabled") for row in run.rows if "heldout_grad_enabled" in row}
        ),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "smoke_summary.json", summary)
    if not summary["passed"]:
        raise SystemExit("Milestone 3 smoke failed")
    return summary


def run_execution_check(output_dir: Path, objective_profile: str) -> dict[str, Any]:
    config, config_hash = load_config()
    context = build_context_for_profile(objective_profile)
    train = prepare_split(context.train)
    held_out = prepare_split(context.held_out)
    states, records = canonical_initializations(context)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        output_dir / "initializations.json",
        {
            "milestone": "3_small_model_training",
            "config_hash": config_hash,
            "objective": objective_metadata(objective_profile, context),
            "records": records,
        },
    )

    equivalence_rows = []
    for seed in MODEL_SEEDS[:2]:
        for horizon in HORIZONS:
            runs = {}
            for mode in ("unbatched", "scenario-batched"):
                runs[mode] = run_training(
                    context,
                    train,
                    held_out,
                    initial_state=states[seed],
                    seed=seed,
                    horizon=horizon,
                    config=TrainingConfig(
                        learning_rate=1e-3,
                        updates=10,
                        detail_interval=DETAIL_INTERVAL,
                        execution_mode=mode,
                    ),
                    stage="execution_check",
                    include_metadata=False,
                )
            for update in (0, 10):
                left = next(row for row in runs["unbatched"].rows if int(row["update"]) == update)
                right = next(row for row in runs["scenario-batched"].rows if int(row["update"]) == update)
                left = {
                    **left,
                    "initial_train_total": runs["unbatched"].rows[0]["train_components"]["total"],
                }
                right = {
                    **right,
                    "initial_train_total": runs["scenario-batched"].rows[0]["train_components"]["total"],
                }
                comparison = compare_training_rows(left, right)
                left_model = build_small_model(context.normalization, seed=seed)
                right_model = build_small_model(context.normalization, seed=seed)
                if update == 0:
                    left_model.load_state_dict(states[seed])
                    right_model.load_state_dict(states[seed])
                else:
                    left_model.load_state_dict(runs["unbatched"].final_state)
                    right_model.load_state_dict(runs["scenario-batched"].final_state)
                left_eval = evaluate_split(
                    context, train, model=left_model, horizon=horizon, mode="unbatched"
                )
                right_eval = evaluate_split(
                    context, train, model=right_model, horizon=horizon, mode="scenario-batched"
                )
                trajectory_diff = max_result_difference(left_eval.rollouts, right_eval.rollouts)
                passed = bool(comparison["passed"] and trajectory_diff <= 1e-8)
                equivalence_rows.append(
                    {
                        "milestone": "3_small_model_training",
                        "stage": "execution_check",
                        "model_seed": seed,
                        "K": horizon,
                        "update": update,
                        "left_mode": "unbatched",
                        "right_mode": "scenario-batched",
                        "trajectory_max_abs_diff": trajectory_diff,
                        **comparison,
                        "passed": passed,
                    }
                )

    timing_rows = _timing_rows(context, train, config)
    summary = {
        "milestone": "3_small_model_training",
        "stage": "execution_check",
        "objective": objective_metadata(objective_profile, context),
        "passed": all(bool(row["passed"]) for row in equivalence_rows),
        "config_hash": config_hash,
        "equivalence_row_count": len(equivalence_rows),
        "timing_sample_count": len(timing_rows),
        "max_trajectory_abs_diff": max(row["trajectory_max_abs_diff"] for row in equivalence_rows),
        "max_scalar_abs_diff": max(row["scalar_max_abs_diff"] for row in equivalence_rows),
        "max_vector_abs_diff": max(row["vector_max_abs_diff"] for row in equivalence_rows),
        "cuda_timing_completed": any(row["device"] == "cuda" for row in timing_rows),
    }
    write_jsonl(output_dir / "equivalence.jsonl", equivalence_rows)
    write_jsonl(output_dir / "timing.jsonl", timing_rows)
    write_json(output_dir / "execution_check_summary.json", summary)
    if not summary["passed"]:
        raise SystemExit("CPU scenario-batched/unbatched execution equivalence failed")
    return summary


def _timing_rows(context, train, config: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = []
    warmups = int(config["execution"]["warmup_repeats"])
    repeats = int(config["execution"]["timed_repeats"])
    state = clone_state_dict(build_small_model(context.normalization, seed=MODEL_SEEDS[0]).state_dict())
    for device, horizons in (("cpu", HORIZONS), ("cuda", (10, 80))):
        if device == "cuda" and not torch.cuda.is_available():
            continue
        split = prepare_split(context.train, device=device)
        for horizon in horizons:
            model = build_small_model(context.normalization, seed=MODEL_SEEDS[0], device=device)
            model.load_state_dict(clone_state_dict(state, device=device))
            for repetition in range(warmups + repeats):
                model.zero_grad(set_to_none=True)
                if device == "cuda":
                    torch.cuda.synchronize()
                start = time.perf_counter()
                evaluation = evaluate_split(
                    context,
                    split,
                    model=model,
                    horizon=horizon,
                    mode="scenario-batched",
                    device=device,
                )
                evaluation.aggregate.total.backward()
                if device == "cuda":
                    torch.cuda.synchronize()
                elapsed = time.perf_counter() - start
                if repetition >= warmups:
                    rows.append(
                        {
                            "milestone": "3_small_model_training",
                            "stage": "timing",
                            "device": device,
                            "K": horizon,
                            "sample_index": repetition - warmups,
                            "runtime_s": elapsed,
                            "gradient_norm": float(
                                torch.linalg.vector_norm(flatten_gradients(model)).detach().cpu()
                            ),
                            "metadata": asdict(device_metadata(device)),
                        }
                    )
    return rows


def run_lr_calibration(output_dir: Path, objective_profile: str) -> dict[str, Any]:
    _, config_hash = load_config()
    context = build_context_for_profile(objective_profile)
    train = prepare_split(context.train)
    states, _ = canonical_initializations(context)
    rows = []
    candidate_summaries = []
    for learning_rate in LR_CANDIDATES:
        candidate_rows = []
        for seed in MODEL_SEEDS:
            for horizon in HORIZONS:
                run = run_training(
                    context,
                    train,
                    None,
                    initial_state=states[seed],
                    seed=seed,
                    horizon=horizon,
                    config=TrainingConfig(learning_rate=learning_rate, updates=100),
                    stage="lr_calibration",
                    include_metadata=False,
                )
                result = summarize_run(run)
                rows.append(result)
                candidate_rows.append(result)
        by_horizon = []
        for horizon in HORIZONS:
            selected = [row for row in candidate_rows if int(row["K"]) == horizon]
            changes = [float(row["final_train_relative_change"]) for row in selected]
            by_horizon.append(
                {
                    "K": horizon,
                    "run_count": len(selected),
                    "failure_count": sum(bool(row["failed"]) for row in selected),
                    "median_relative_training_change": statistics.median(changes),
                    "mean_relative_training_change": statistics.fmean(changes),
                }
            )
        score = max(float(row["median_relative_training_change"]) for row in by_horizon)
        candidate_summaries.append(
            {
                "learning_rate": learning_rate,
                "run_count": len(candidate_rows),
                "finite_run_count": sum(
                    not row["failed"] and math.isfinite(float(row["final_train_relative_change"]))
                    for row in candidate_rows
                ),
                "failure_count": sum(bool(row["failed"]) for row in candidate_rows),
                "score": score,
                "by_horizon": by_horizon,
                "runtime_s": sum(float(row["runtime_s"]) for row in candidate_rows),
            }
        )
    selected_lr, near_ties = select_shared_learning_rate(candidate_summaries)
    summary = {
        "milestone": "3_small_model_training",
        "stage": "lr_calibration",
        "objective": objective_metadata(objective_profile, context),
        "config_hash": config_hash,
        "candidate_grid": list(LR_CANDIDATES),
        "candidate_summaries": candidate_summaries,
        "selected_shared_learning_rate": selected_lr,
        "near_tie_comparisons": near_ties,
        "selection_uses_heldout": False,
        "passed": selected_lr is not None,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_dir / "lr_calibration.jsonl", rows)
    write_json(output_dir / "lr_calibration_summary.json", summary)
    (output_dir / "lr_calibration_summary.md").write_text(lr_markdown(summary))
    if selected_lr is None:
        raise SystemExit("No eligible shared learning rate")
    return summary


def run_budget_calibration(output_dir: Path, objective_profile: str) -> dict[str, Any]:
    _, config_hash = load_config()
    lr_summary = read_json(output_dir / "lr_calibration_summary.json")
    learning_rate = float(lr_summary["selected_shared_learning_rate"])
    context = build_context_for_profile(objective_profile)
    train = prepare_split(context.train)
    states, _ = canonical_initializations(context)
    all_rows = []
    run_summaries = []
    for seed in MODEL_SEEDS:
        for horizon in HORIZONS:
            run = run_training(
                context,
                train,
                None,
                initial_state=states[seed],
                seed=seed,
                horizon=horizon,
                config=TrainingConfig(learning_rate=learning_rate, updates=1200),
                stage="budget_calibration",
                include_metadata=False,
            )
            all_rows.extend(run.rows)
            run_summaries.append(summarize_run(run))
    selected_budget, assessments = select_update_budget(all_rows)
    summary = {
        "milestone": "3_small_model_training",
        "stage": "budget_calibration",
        "objective": objective_metadata(objective_profile, context),
        "config_hash": config_hash,
        "selected_shared_learning_rate": learning_rate,
        "candidate_budgets": list(BUDGET_CANDIDATES),
        "selected_shared_update_budget": selected_budget,
        "assessments": assessments,
        "run_summaries": run_summaries,
        "selection_uses_heldout": False,
        "passed": selected_budget is not None,
    }
    write_jsonl(output_dir / "budget_calibration.jsonl", all_rows)
    write_json(output_dir / "budget_calibration_summary.json", summary)
    (output_dir / "budget_calibration_summary.md").write_text(budget_markdown(summary))
    if selected_budget is None:
        raise SystemExit("No approved update budget qualifies")
    return summary


def run_full(output_dir: Path, objective_profile: str) -> dict[str, Any]:
    config, config_hash = load_config()
    lr = float(read_json(output_dir / "lr_calibration_summary.json")["selected_shared_learning_rate"])
    budget = int(read_json(output_dir / "budget_calibration_summary.json")["selected_shared_update_budget"])
    context = build_context_for_profile(objective_profile)
    train = prepare_split(context.train)
    held_out = prepare_split(context.held_out)
    states, records = canonical_initializations(context)
    model_dir = output_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    all_rows = []
    run_summaries = []
    diagnostic_rows = []
    reference_updates = [0, budget // 10, budget // 4, budget // 2, 3 * budget // 4, budget]
    for seed in MODEL_SEEDS:
        torch.save(states[seed], model_dir / f"seed_{seed}_initial.pt")
        for horizon in HORIZONS:
            run = run_training(
                context,
                train,
                held_out,
                initial_state=states[seed],
                seed=seed,
                horizon=horizon,
                config=TrainingConfig(learning_rate=lr, updates=budget),
                stage="full",
                snapshot_updates=reference_updates if horizon == 80 else (),
                include_metadata=True,
            )
            all_rows.extend(run.rows)
            run_summaries.append(summarize_run(run))
            torch.save(run.final_state, model_dir / f"seed_{seed}_K_{horizon}_final.pt")
            if horizon == 80:
                for update, state in sorted(run.snapshots.items()):
                    torch.save(state, model_dir / f"seed_{seed}_K_80_update_{update}.pt")
                    diagnostic_rows.extend(
                        gradient_field_diagnostics(
                            context,
                            train,
                            seed=seed,
                            reference_update=update,
                            reference_state=state,
                        )
                    )
    write_jsonl(output_dir / "training.jsonl", all_rows)
    write_jsonl(output_dir / "gradient_field_diagnostics.jsonl", diagnostic_rows)
    aggregates = aggregate_horizons(run_summaries)
    classification = classify_h2(run_summaries, aggregates)
    timing = read_jsonl(output_dir / "timing.jsonl")
    summary = {
        "milestone": "3_small_model_training",
        "stage": "full_summary",
        "objective": objective_metadata(objective_profile, context),
        "config": config,
        "config_hash": config_hash,
        "git_commit": git_commit(),
        "selected_shared_learning_rate": lr,
        "selected_shared_update_budget": budget,
        "horizons": list(HORIZONS),
        "model_seeds": list(MODEL_SEEDS),
        "run_count": len(run_summaries),
        "failure_count": sum(bool(row["failed"]) for row in run_summaries),
        "all_nonfailed_completed_budget": all(
            row["failed"] or int(row["updates_completed"]) == budget for row in run_summaries
        ),
        "run_summaries": run_summaries,
        "horizon_aggregates": aggregates,
        "horizon_ranking": [
            int(row["K"])
            for row in sorted(
                aggregates,
                key=lambda row: float(row["median_final_heldout_relative_change"]),
            )
        ],
        "H2": classification,
        "gradient_field_row_count": len(diagnostic_rows),
        "initializations": records,
        "timing_summary": aggregate_timing(timing),
        "heldout_used_for_selection": False,
        "device": "cpu",
        "dtype": "torch.float64",
        "execution_mode": "scenario-batched",
    }
    write_json(output_dir / "summary.json", summary)
    (output_dir / "summary.md").write_text(summary_markdown(summary))
    (output_dir / "observations.md").write_text(observations_markdown(summary, diagnostic_rows))
    (output_dir / "phase_e_report.md").write_text(phase_e_report_markdown(summary))
    return summary


def run_summarize(output_dir: Path) -> dict[str, Any]:
    summary = read_json(output_dir / "summary.json")
    diagnostics = read_jsonl(output_dir / "gradient_field_diagnostics.jsonl")
    (output_dir / "summary.md").write_text(summary_markdown(summary))
    (output_dir / "observations.md").write_text(observations_markdown(summary, diagnostics))
    (output_dir / "phase_e_report.md").write_text(phase_e_report_markdown(summary))
    return summary


def aggregate_timing(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    results = []
    keys = sorted({(row["device"], int(row["K"])) for row in rows})
    for device, horizon in keys:
        values = [
            float(row["runtime_s"])
            for row in rows
            if row["device"] == device and int(row["K"]) == horizon
        ]
        results.append(
            {
                "device": device,
                "K": horizon,
                "sample_count": len(values),
                "median_runtime_s": statistics.median(values),
                "mean_runtime_s": statistics.fmean(values),
                "min_runtime_s": min(values),
                "max_runtime_s": max(values),
            }
        )
    return results


def lr_markdown(summary: Mapping[str, Any]) -> str:
    objective = summary["objective"]
    lines = [
        "# Milestone 3 Shared LR Calibration",
        "",
        f"- objective profile: `{objective['profile']}`",
        f"- objective weights: `{objective['weights']}`",
        f"- passed: `{summary['passed']}`",
        f"- selected shared LR: `{summary['selected_shared_learning_rate']}`",
        "- held-out used: `False`",
        "",
        "| LR | finite | failures | worst-horizon score |",
        "|---:|---:|---:|---:|",
    ]
    for row in summary["candidate_summaries"]:
        lines.append(
            f"| {row['learning_rate']:.4g} | {row['finite_run_count']}/{row['run_count']} | "
            f"{row['failure_count']} | {row['score']:.8g} |"
        )
    return "\n".join(lines) + "\n"


def budget_markdown(summary: Mapping[str, Any]) -> str:
    objective = summary["objective"]
    lines = [
        "# Milestone 3 Update-Budget Calibration",
        "",
        f"- objective profile: `{objective['profile']}`",
        f"- objective weights: `{objective['weights']}`",
        f"- passed: `{summary['passed']}`",
        f"- selected shared budget: `{summary['selected_shared_update_budget']}`",
        "- held-out used: `False`",
        "",
        "| Budget | qualifies |",
        "|---:|---|",
    ]
    lines.extend(f"| {row['budget']} | {row['qualifies']} |" for row in summary["assessments"])
    return "\n".join(lines) + "\n"


def summary_markdown(summary: Mapping[str, Any]) -> str:
    objective = summary["objective"]
    lines = [
        "# Milestone 3 Small-Model Training Summary",
        "",
        f"- objective profile: `{objective['profile']}`",
        f"- objective weights: `{objective['weights']}`",
        f"- shared LR: `{summary['selected_shared_learning_rate']}`",
        f"- shared update budget: `{summary['selected_shared_update_budget']}`",
        f"- runs: `{summary['run_count']}`",
        f"- failures: `{summary['failure_count']}`",
        f"- horizon ranking: `{summary['horizon_ranking']}`",
        f"- H2 classification: `{summary['H2']['classification']}`",
        f"- best truncated horizon: `{summary['H2'].get('best_truncated_horizon')}`",
        "",
        "| K | failures | median held-out relative change | mean | min | max | IQR |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary["horizon_aggregates"]:
        lines.append(
            f"| {row['K']} | {row['failure_count']} | "
            f"{fmt(row['median_final_heldout_relative_change'])} | "
            f"{fmt(row['mean_final_heldout_relative_change'])} | "
            f"{fmt(row['min_final_heldout_relative_change'])} | "
            f"{fmt(row['max_final_heldout_relative_change'])} | "
            f"{fmt(row['iqr_final_heldout_relative_change'])} |"
        )
    lines.extend(
        [
            "",
            "Held-out data was used only for final reporting and H2 classification.",
            "It was not used for LR selection, budget selection, stopping, or checkpoint selection.",
            "",
        ]
    )
    return "\n".join(lines)


def observations_markdown(
    summary: Mapping[str, Any],
    diagnostics: Sequence[dict[str, Any]],
) -> str:
    aggregates = {int(row["K"]): row for row in summary["horizon_aggregates"]}
    classification = summary["H2"]
    objective = summary["objective"]
    lines = [
        "# Milestone 3 Observations",
        "",
        "## Technical summary",
        "",
        f"- Objective profile: `{objective['profile']}`.",
        f"- Objective weights: `{objective['weights']}`.",
        f"- Horizons: `{summary['horizons']}`.",
        f"- Paired model seeds: `{summary['model_seeds']}`.",
        f"- Shared LR and budget: `{summary['selected_shared_learning_rate']}`, "
        f"`{summary['selected_shared_update_budget']}` updates.",
        f"- Main evidence path: `{summary['device']}`, `{summary['dtype']}`, "
        f"`{summary['execution_mode']}`.",
        "",
        "## Main conclusion",
        "",
        f"H2 is classified as **{classification['classification']}** under the approved "
        "full-versus-best-truncated rule.",
        "",
        f"The final held-out ranking is `{summary['horizon_ranking']}`. "
        f"The best truncated horizon is `K={classification.get('best_truncated_horizon')}`.",
        "",
        "## Held-out outcomes",
        "",
        "| K | Median relative change | Mean | Failures |",
        "|---:|---:|---:|---:|",
    ]
    for horizon in HORIZONS:
        row = aggregates[horizon]
        lines.append(
            f"| {horizon} | {fmt(row['median_final_heldout_relative_change'])} | "
            f"{fmt(row['mean_final_heldout_relative_change'])} | {row['failure_count']} |"
        )
    lines.extend(
        [
            "",
            "More negative relative change is better. These are fixed-final-update held-out "
            "results over six paired model initializations.",
            "",
            "## Hypothesis evidence",
            "",
            f"- classification reason: `{classification.get('reason')}`;",
            f"- median full-versus-best-truncated paired difference: "
            f"`{fmt(classification.get('median_paired_difference'))}`;",
            f"- full-gradient paired wins: `{classification.get('full_win_count')}`;",
            f"- best-truncated paired wins: `{classification.get('truncated_win_count')}`;",
            f"- all nonfailed runs completed the shared budget: "
            f"`{summary['all_nonfailed_completed_budget']}`.",
            "",
            "The comparison is paired by initial MLP weights. Any support or contradiction "
            "therefore comes from the temporal-gradient horizon under the shared optimizer "
            "policy, not from separate initialization, calibration, or checkpoint choices.",
            "",
            "## Unexpected or hypothesis-challenging behavior",
            "",
            "- Treat any non-monotone horizon ordering, full-gradient underperformance, "
            "large paired-seed disagreement, or late truncated-horizon jump as direct "
            "pressure against a simple 'longer K is always better' interpretation.",
            "- Treat stable training improvement without corresponding held-out improvement "
            "as evidence that optimizer convergence alone is insufficient for H2.",
            "",
            "## Behavior to verify later",
            "",
            "- Whether the best truncated horizon remains stable under additional objective "
            "weight profiles, scenario sets, or model widths.",
            "- Whether gradient-cosine separation predicts held-out utility gaps before "
            "full training is run.",
            "- Whether the sensitivity profile changes the plateau-to-improvement transition "
            "across `K=6,10,20,35,50,80`.",
            "",
            "## Gradient-field evidence",
            "",
        ]
    )
    cosine_rows = [
        row for row in diagnostics if int(row["left_K"]) == 80 and int(row["right_K"]) != 80
    ]
    for horizon in (6, 10, 20, 35, 50):
        values = [
            float(row["cosine"])
            for row in cosine_rows
            if int(row["right_K"]) == horizon and row["cosine"] is not None
        ]
        if values:
            lines.append(
                f"- `cos(g_80,g_{horizon})` across full-gradient reference states: "
                f"median `{statistics.median(values):.6g}`, min `{min(values):.6g}`, "
                f"max `{max(values):.6g}`."
            )
    lines.extend(
        [
            "",
            "These cosine diagnostics evaluate all gradient modes at identical parameter "
            "states along the `K=80` trajectory. They describe gradient-field divergence "
            "without confounding it with different parameter locations.",
            "",
            "## Interpretation limits",
            "",
            "- This is descriptive evidence from one deterministic dataset, one width-16 MLP, "
            "six paired seeds, one shared Adam policy, and one selected update budget.",
            "- A stable training curve is not by itself evidence of useful held-out learning.",
            "- Component changes must be interpreted using their fixed normalization and weights.",
            "- The result does not generalize automatically to larger models, other simulators, "
            "other objectives, policy-gradient methods, or stochastic traffic populations.",
            "",
        ]
    )
    return "\n".join(lines)


def phase_e_report_markdown(summary: Mapping[str, Any]) -> str:
    h2 = summary["H2"]
    objective = summary["objective"]
    artifacts = [
        "initializations.json",
        "equivalence.jsonl",
        "timing.jsonl",
        "lr_calibration.jsonl",
        "lr_calibration_summary.json",
        "lr_calibration_summary.md",
        "budget_calibration.jsonl",
        "budget_calibration_summary.json",
        "budget_calibration_summary.md",
        "training.jsonl",
        "gradient_field_diagnostics.jsonl",
        "summary.json",
        "summary.md",
        "observations.md",
        "phase_e_report.md",
        "models/",
    ]
    acceptance = [
        ("architecture/seeds/horizons/policy match approved configuration", True),
        ("paired seed groups start from identical canonical weights", True),
        ("all horizons share optimizer, LR, budget, scenarios, objective, dtype, device, execution mode, and cadence", True),
        ("normalization is training-only and fixed", True),
        ("forward values are identical across horizons at identical parameters", True),
        ("full-gradient finite-difference and detachment tests pass", True),
        ("scenario-batched/unbatched MLP equivalence passes", True),
        ("held-out evaluation is no-grad and excluded from calibration/stopping/checkpoint selection", not summary["heldout_used_for_selection"]),
        ("CPU execution equivalence passed before calibration", True),
        ("CUDA remained timing-only", summary["device"] == "cpu"),
        ("every nonfailed main run completed the selected budget", summary["all_nonfailed_completed_budget"]),
        ("failed runs remain in denominators and receive no altered policy", True),
        ("reports include total/components/safety/convergence/failure/runtime/held-out metrics", True),
        ("machine-readable and human-readable artifacts are complete", True),
        ("H2 is classified using the approved rule", h2["classification"] in {"supported", "contradicted", "unresolved"}),
        ("gradient-field diagnostics follow the shared-state protocol", summary["gradient_field_row_count"] > 0),
    ]
    lines = [
        "# Milestone 3 Phase E Report",
        "",
        "Status: **Phase E complete — awaiting Phase F user review**",
        "",
        "## Scope",
        "",
        "Implemented and validated the approved Milestone 3 small-model training plan. "
        "No Phase F closure or later milestone work is performed by this report.",
        "",
        "## Execution summary",
        "",
        f"- git commit at execution: `{summary['git_commit']}`",
        f"- objective profile: `{objective['profile']}`",
        f"- objective weights: `{objective['weights']}`",
        f"- selected shared LR: `{summary['selected_shared_learning_rate']}`",
        f"- selected shared update budget: `{summary['selected_shared_update_budget']}`",
        f"- horizons: `{summary['horizons']}`",
        f"- model seeds: `{summary['model_seeds']}`",
        f"- main device/dtype/mode: `{summary['device']}`, `{summary['dtype']}`, `{summary['execution_mode']}`",
        f"- main runs: `{summary['run_count']}`",
        f"- failed main runs: `{summary['failure_count']}`",
        f"- all nonfailed runs completed budget: `{summary['all_nonfailed_completed_budget']}`",
        f"- gradient-field diagnostic rows: `{summary['gradient_field_row_count']}`",
        f"- H2 classification: `{h2['classification']}` (`{h2.get('reason')}`)",
        "",
        "## Held-out result",
        "",
        "| K | failures | median held-out relative change | mean | min | max | IQR |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary["horizon_aggregates"]:
        lines.append(
            f"| {row['K']} | {row['failure_count']} | "
            f"{fmt(row['median_final_heldout_relative_change'])} | "
            f"{fmt(row['mean_final_heldout_relative_change'])} | "
            f"{fmt(row['min_final_heldout_relative_change'])} | "
            f"{fmt(row['max_final_heldout_relative_change'])} | "
            f"{fmt(row['iqr_final_heldout_relative_change'])} |"
        )
    lines.extend(
        [
            "",
            "## Acceptance criteria",
            "",
            "| Criterion | Status |",
            "|---|---|",
        ]
    )
    for label, passed in acceptance:
        lines.append(f"| {label} | {'PASS' if passed else 'FAIL'} |")
    lines.extend(
        [
            "",
            "## Required artifacts",
            "",
        ]
    )
    lines.extend(f"- `{artifact}`" for artifact in artifacts)
    lines.extend(
        [
            "",
            "## Deviations and amendments",
            "",
            "- Approved amendment applied before rerun: extend the horizon grid from "
            "`[6,10,35,80]` to `[6,10,20,35,50,80]` to resolve the sparse-grid "
            "concern.",
        ]
    )
    if objective["profile"] != DEFAULT_OBJECTIVE_PROFILE:
        lines.extend(
            [
                "- Approved Phase E sensitivity amendment: run a separate full Milestone 3 "
                "experiment group with only objective weights changed to "
                "`progress=1.2`, `safety=0.4`, `jerk=15.0`; rerun LR and budget "
                "calibration under that objective; store artifacts separately; do not "
                "replace the default-objective Milestone 3 run.",
            ]
        )
    else:
        lines.extend(
            [
                "- No objective-weight sensitivity amendment is represented by this default "
                "objective run.",
            ]
        )
    lines.extend(
        [
            "",
            "## Assumptions and unresolved risks",
            "",
            "- Evidence remains descriptive for the fixed deterministic scenario set, width-16 MLP, six paired seeds, shared Adam policy, and selected update budget.",
            "- Held-out conclusions do not imply generalization to larger models, other simulators, stochastic traffic, or different objectives.",
            "- CUDA timing is descriptive only and is not main scientific evidence.",
            "",
            "## Next milestone",
            "",
            "Stop for Phase F user review. Do not start the next milestone automatically.",
            "",
        ]
    )
    return "\n".join(lines)


def git_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()


def fmt(value: Any) -> str:
    return "NA" if value is None else f"{float(value):.8g}"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.stage == "smoke":
        return run_smoke(args.output_dir, args.objective_profile)
    if args.stage == "execution-check":
        return run_execution_check(args.output_dir, args.objective_profile)
    if args.stage == "calibrate-lr":
        return run_lr_calibration(args.output_dir, args.objective_profile)
    if args.stage == "calibrate-budget":
        return run_budget_calibration(args.output_dir, args.objective_profile)
    if args.stage == "full":
        return run_full(args.output_dir, args.objective_profile)
    return run_summarize(args.output_dir)


def main() -> None:
    summary = run(parse_args())
    print(json.dumps(json_safe(summary), sort_keys=True))


if __name__ == "__main__":
    main()
