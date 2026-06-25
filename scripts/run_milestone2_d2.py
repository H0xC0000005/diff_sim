"""Run Milestone 2 D.2 iterative structured optimization stages."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import statistics
import sys
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from differential_sim.device_parity import build_parity_context, json_safe, write_json, write_jsonl
from differential_sim.structured_optimization import (
    LR_CANDIDATES,
    OptimizationConfig,
    OptimizationRun,
    aggregate_horizons,
    classify_h1,
    compare_rows,
    operational_ties,
    prepare_split,
    rank_horizons,
    run_optimization,
    select_shared_learning_rate,
    spearman_against_milestone1,
    summarize_run,
)


DEFAULT_OUTPUT_DIR = ROOT / "reports" / "milestone2" / "structured_optimization"
T_INIT_VALUES = (0.9, 1.2, 1.4, 1.6, 1.9, 2.2)
SEEDS = (0, 1, 2, 3, 4, 5)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("smoke", "calibrate", "full", "summarize"), required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def run_smoke(output_dir: Path) -> dict[str, Any]:
    context = build_parity_context()
    train = prepare_split(context.train)
    held_out = prepare_split(context.held_out)
    initialization_id = 2
    beta_initial = context.init_betas_cpu[initialization_id]
    comparison_rows = []
    for horizon in context.horizons:
        runs = {}
        for mode in ("unbatched", "scenario-batched"):
            runs[mode] = run_optimization(
                context,
                train,
                held_out,
                beta_initial=beta_initial,
                horizon=horizon,
                initialization_id=initialization_id,
                t_init=T_INIT_VALUES[initialization_id],
                seed=SEEDS[initialization_id],
                config=OptimizationConfig(
                    learning_rate=0.03,
                    updates=50,
                    execution_mode=mode,
                ),
                stage="equivalence_smoke",
                include_metadata=True,
            )
        for update in (0, 10, 50):
            left = _row_at(runs["unbatched"].rows, update)
            right = _row_at(runs["scenario-batched"].rows, update)
            left = {**left, "initial_train_total": runs["unbatched"].rows[0]["train_components"]["total"]}
            right = {**right, "initial_train_total": runs["scenario-batched"].rows[0]["train_components"]["total"]}
            comparison = compare_rows(left, right)
            comparison_rows.append(
                {
                    "milestone": "2_d2_iterative_structured_optimization",
                    "stage": "equivalence_smoke",
                    "K": horizon,
                    "horizon_label": "T" if horizon == 80 else str(horizon),
                    "initialization_id": initialization_id,
                    "T_init": T_INIT_VALUES[initialization_id],
                    "learning_rate": 0.03,
                    "update": update,
                    "left_execution_mode": "unbatched",
                    "right_execution_mode": "scenario-batched",
                    **comparison,
                }
            )
    summary = {
        "milestone": "2_d2_iterative_structured_optimization",
        "stage": "equivalence_smoke",
        "passed": all(bool(row["passed"]) for row in comparison_rows),
        "row_count": len(comparison_rows),
        "horizons": list(context.horizons),
        "initialization_id": initialization_id,
        "T_init": T_INIT_VALUES[initialization_id],
        "learning_rate": 0.03,
        "updates": 50,
        "comparison_updates": [0, 10, 50],
        "max_scalar_abs_diff": max(float(row["scalar_max_abs_diff"]) for row in comparison_rows),
        "max_scalar_rel_diff": max(float(row["scalar_max_rel_diff"]) for row in comparison_rows),
        "max_beta_abs_diff": max(float(row["beta_max_abs_diff"]) for row in comparison_rows),
        "max_beta_rel_diff": max(float(row["beta_max_rel_diff"]) for row in comparison_rows),
        "all_flags_match": all(bool(row["flags_match"]) for row in comparison_rows),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_dir / "equivalence_smoke.jsonl", comparison_rows)
    write_json(output_dir / "equivalence_smoke_summary.json", summary)
    if not summary["passed"]:
        raise SystemExit("D.2 equivalence smoke failed; stop before calibration")
    return summary


def run_calibration(output_dir: Path) -> dict[str, Any]:
    context = build_parity_context()
    train = prepare_split(context.train)
    rows = []
    candidate_summaries = []
    for learning_rate in LR_CANDIDATES:
        candidate_rows = []
        for initialization_id, beta_initial in enumerate(context.init_betas_cpu):
            for horizon in context.horizons:
                run = run_optimization(
                    context,
                    train,
                    None,
                    beta_initial=beta_initial,
                    horizon=horizon,
                    initialization_id=initialization_id,
                    t_init=T_INIT_VALUES[initialization_id],
                    seed=SEEDS[initialization_id],
                    config=OptimizationConfig(learning_rate=learning_rate, updates=40),
                    stage="calibration",
                    include_metadata=True,
                )
                summary = summarize_run(run)
                rows.append(summary)
                candidate_rows.append(summary)
        relative_changes = [float(row["final_train_relative_change"]) for row in candidate_rows]
        finite_count = sum(
            not row["failed"] and math.isfinite(float(row["final_train_relative_change"]))
            for row in candidate_rows
        )
        candidate_summaries.append(
            {
                "learning_rate": learning_rate,
                "run_count": len(candidate_rows),
                "finite_run_count": finite_count,
                "failure_count": sum(bool(row["failed"]) for row in candidate_rows),
                "failure_reasons": sorted(
                    {str(row["failure_reason"]) for row in candidate_rows if row["failed"]}
                ),
                "median_relative_training_change": statistics.median(relative_changes),
                "mean_relative_training_change": statistics.fmean(relative_changes),
                "min_relative_training_change": min(relative_changes),
                "max_relative_training_change": max(relative_changes),
                "runtime_s": sum(float(row["runtime_s"]) for row in candidate_rows),
                "by_horizon": _group_calibration(candidate_rows, "K"),
                "by_initialization": _group_calibration(candidate_rows, "initialization_id"),
            }
        )
    selected_lr, near_tie_comparisons = select_shared_learning_rate(candidate_summaries)
    summary = {
        "milestone": "2_d2_iterative_structured_optimization",
        "stage": "calibration",
        "candidate_grid": list(LR_CANDIDATES),
        "candidate_summaries": candidate_summaries,
        "selected_shared_learning_rate": selected_lr,
        "near_tie_comparisons": near_tie_comparisons,
        "selection_uses_heldout": False,
        "passed": selected_lr is not None,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_dir / "calibration.jsonl", rows)
    write_json(output_dir / "calibration_summary.json", summary)
    (output_dir / "calibration_summary.md").write_text(calibration_markdown(summary))
    if selected_lr is None:
        raise SystemExit("No LR candidate is eligible; stop for a plan amendment")
    return summary


def run_full(output_dir: Path) -> dict[str, Any]:
    calibration = _read_json(output_dir / "calibration_summary.json")
    learning_rate = calibration.get("selected_shared_learning_rate")
    if learning_rate is None:
        raise SystemExit("Calibration summary has no selected shared LR")
    context = build_parity_context()
    train = prepare_split(context.train)
    held_out = prepare_split(context.held_out)
    all_rows = []
    run_summaries = []
    for initialization_id, beta_initial in enumerate(context.init_betas_cpu):
        for horizon in context.horizons:
            run = run_optimization(
                context,
                train,
                held_out,
                beta_initial=beta_initial,
                horizon=horizon,
                initialization_id=initialization_id,
                t_init=T_INIT_VALUES[initialization_id],
                seed=SEEDS[initialization_id],
                config=OptimizationConfig(learning_rate=float(learning_rate), updates=500),
                stage="full",
                include_metadata=True,
            )
            all_rows.extend(run.rows)
            run_summaries.append(summarize_run(run))
    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_dir / "optimization.jsonl", all_rows)
    summary = build_full_summary(run_summaries, float(learning_rate))
    write_json(output_dir / "summary.json", summary)
    (output_dir / "summary.md").write_text(summary_markdown(summary))
    return summary


def run_summarize(output_dir: Path) -> dict[str, Any]:
    rows = _read_jsonl(output_dir / "optimization.jsonl")
    calibration = _read_json(output_dir / "calibration_summary.json")
    run_summaries = []
    for run_id in sorted({str(row["run_id"]) for row in rows}):
        run_rows = sorted(
            [row for row in rows if row["run_id"] == run_id],
            key=lambda row: int(row["update"]),
        )
        run = OptimizationRun(
            rows=run_rows,
            failed=bool(run_rows[-1]["failed"]),
            failure_reason=str(run_rows[-1]["failure_reason"]),
            updates_completed=int(run_rows[-1]["update"]),
            runtime_s=sum(float(row["step_runtime_s"]) for row in run_rows),
        )
        run_summaries.append(summarize_run(run))
    summary = build_full_summary(
        run_summaries,
        float(calibration["selected_shared_learning_rate"]),
    )
    write_json(output_dir / "summary.json", summary)
    (output_dir / "summary.md").write_text(summary_markdown(summary))
    return summary


def build_full_summary(run_summaries: Sequence[dict[str, Any]], learning_rate: float) -> dict[str, Any]:
    aggregates = aggregate_horizons(run_summaries)
    ranking = rank_horizons(aggregates)
    ties = operational_ties(aggregates)
    return {
        "milestone": "2_d2_iterative_structured_optimization",
        "stage": "full_summary",
        "selected_shared_learning_rate": learning_rate,
        "horizons": [1, 3, 6, 10, 80],
        "initializations": list(T_INIT_VALUES),
        "run_count": len(run_summaries),
        "failure_count": sum(bool(row["failed"]) for row in run_summaries),
        "all_nonfailed_completed_500": all(
            bool(row["failed"]) or int(row["updates_completed"]) == 500 for row in run_summaries
        ),
        "run_summaries": list(run_summaries),
        "horizon_aggregates": aggregates,
        "milestone1_ranking": [80, 10, 6, 3, 1],
        "d2_ranking": ranking,
        "operational_ties": ties,
        "spearman_rank_correlation": spearman_against_milestone1(ranking),
        "top_group_overlap_count": len(set(ranking[:3]) & {80, 10, 6}),
        "K80_status": _k80_status(ranking, ties),
        "H1_classification": classify_h1(ranking, ties),
        "primary_outcome": "update-500 held-out relative objective change",
        "heldout_used_for_selection": False,
        "device": "cpu",
        "dtype": "torch.float64",
        "execution_mode": "scenario-batched",
        "updates": 500,
    }


def calibration_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Milestone 2 D.2 Shared LR Calibration",
        "",
        f"- passed: `{summary['passed']}`",
        f"- selected shared LR: `{summary['selected_shared_learning_rate']}`",
        "- held-out data used: `False`",
        "",
        "| LR | finite | failures | median relative change | mean relative change |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in summary["candidate_summaries"]:
        lines.append(
            f"| {row['learning_rate']:.3g} | {row['finite_run_count']}/{row['run_count']} | "
            f"{row['failure_count']} | {row['median_relative_training_change']:.8g} | "
            f"{row['mean_relative_training_change']:.8g} |"
        )
    return "\n".join(lines) + "\n"


def summary_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Milestone 2 D.2 Structured Optimization Summary",
        "",
        f"- selected shared LR: `{summary['selected_shared_learning_rate']}`",
        f"- runs: `{summary['run_count']}`",
        f"- failures: `{summary['failure_count']}`",
        f"- D.2 ranking: `{summary['d2_ranking']}`",
        f"- Milestone 1 ranking: `{summary['milestone1_ranking']}`",
        f"- Spearman correlation: `{summary['spearman_rank_correlation']}`",
        f"- H1 classification: `{summary['H1_classification']}`",
        f"- K=80 status: `{summary['K80_status']}`",
        "",
        "| K | failures | median held-out relative change | mean | min | max | IQR |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in sorted(summary["horizon_aggregates"], key=lambda item: int(item["K"])):
        lines.append(
            f"| {row['K']} | {row['failure_count']} | "
            f"{_fmt(row['median_final_heldout_relative_change'])} | "
            f"{_fmt(row['mean_final_heldout_relative_change'])} | "
            f"{_fmt(row['min_final_heldout_relative_change'])} | "
            f"{_fmt(row['max_final_heldout_relative_change'])} | "
            f"{_fmt(row['iqr_final_heldout_relative_change'])} |"
        )
    lines.extend(
        [
            "",
            "Held-out evaluation was no-grad and was not used for LR selection, stopping, or checkpoint selection.",
            "",
        ]
    )
    return "\n".join(lines)


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.stage == "smoke":
        return run_smoke(args.output_dir)
    if args.stage == "calibrate":
        return run_calibration(args.output_dir)
    if args.stage == "full":
        return run_full(args.output_dir)
    return run_summarize(args.output_dir)


def main() -> None:
    summary = run(parse_args())
    print(json.dumps(json_safe(summary), sort_keys=True))


def _row_at(rows: Sequence[dict[str, Any]], update: int) -> dict[str, Any]:
    return next(row for row in rows if int(row["update"]) == update)


def _group_calibration(rows: Sequence[dict[str, Any]], field: str) -> list[dict[str, Any]]:
    groups = []
    for value in sorted({row[field] for row in rows}):
        selected = [row for row in rows if row[field] == value]
        changes = [float(row["final_train_relative_change"]) for row in selected]
        groups.append(
            {
                field: value,
                "count": len(selected),
                "failure_count": sum(bool(row["failed"]) for row in selected),
                "median_relative_training_change": statistics.median(changes),
                "mean_relative_training_change": statistics.fmean(changes),
                "min_relative_training_change": min(changes),
                "max_relative_training_change": max(changes),
            }
        )
    return groups


def _k80_status(ranking: Sequence[int], ties: Sequence[Sequence[int]]) -> str:
    if not ranking:
        return "unresolved"
    if ranking[0] == 80:
        return "best"
    if any(80 in pair and ranking[0] in pair for pair in ties):
        return "operationally_tied_with_best"
    return "surpassed"


def _fmt(value: Any) -> str:
    return "NA" if value is None else f"{float(value):.8g}"


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


if __name__ == "__main__":
    main()
