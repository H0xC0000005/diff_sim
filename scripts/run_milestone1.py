"""Run Milestone 1 one-step gradient utility diagnostics."""

from __future__ import annotations

import json
from importlib import metadata
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from differential_sim.milestone1_diagnostics import run_one_step_diagnostics, write_jsonl


REPORT_DIR = ROOT / "reports" / "milestone1"


def main() -> None:
    torch.manual_seed(0)
    normalized = run_and_write("normalized")
    si_units = run_and_write("si_units")
    top = [
        "# Milestone 1 Local Gradient Utility",
        "",
        "Primary result: `normalized/summary.md`.",
        "",
        "Secondary SI-unit validation: `si_units/summary.md`.",
        "",
        "## Environment",
        "",
        f"- python: `{sys.version.split()[0]}`",
        f"- torch: `{torch.__version__}`",
        f"- diffidm: `{metadata.version('diffidm')}`",
        f"- dtype: `torch.float64`",
        f"- device: `cpu`",
        "",
        "## Descriptive Beats-Random Rule",
        "",
        f"- normalized any mode beats random: `{any(v['beats_random_aggregated'] for v in normalized['by_mode'].values())}`",
        f"- si_units any mode beats random: `{any(v['beats_random_aggregated'] for v in si_units['by_mode'].values())}`",
        "",
    ]
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "summary.md").write_text("\n".join(top))


def run_and_write(parameterization: str) -> dict:
    artifacts = run_one_step_diagnostics(input_parameterization=parameterization, random_direction_count=32)
    out_dir = REPORT_DIR / parameterization
    out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(out_dir / "diagnostics.jsonl", artifacts.rows)
    summary = {
        "environment": {
            "python": sys.version,
            "torch": torch.__version__,
            "diffidm": metadata.version("diffidm"),
            "dtype": "torch.float64",
            "device": "cpu",
        },
        **artifacts.summary,
        "held_out": artifacts.held_out,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    (out_dir / "summary.md").write_text(summary_markdown(summary))
    return summary


def summary_markdown(summary: dict) -> str:
    lines = [
        f"# Milestone 1 Summary: {summary['input_parameterization']}",
        "",
        f"- primary alpha: `{summary['primary_alpha']}`",
        f"- random directions per initialization: `{summary['random_direction_count']}`",
        f"- aggregated random improvement probability: `{summary['random_aggregated_primary_probability']:.6g}`",
        f"- zero-gradient count: `{summary['zero_gradient_count']}`",
        f"- unsafe/non-finite update count: `{summary['unsafe_or_nonfinite_count']}`",
        f"- diagnostic scenarios: `{len(summary['scenarios']['diagnostic'])}`",
        f"- held-out scenarios: `{len(summary['scenarios']['held_out'])}`",
        f"- objective weights: `{summary['objective_weights']}`",
        "",
        "## Interpretation",
        "",
        "- This report addresses H1 Stage 1 only: one-step local utility of simulator-gradient directions.",
        "- Normalized inputs are the primary result; SI-unit inputs are secondary parameterization validation.",
        "- Compare the richer-scenario result to the earlier elementary-scenario observation by checking whether gradient modes still beat random directions and whether horizon rankings change.",
        "",
        "## Primary Alpha Results",
        "",
        "| mode | gradient improved | gradient probability | random probability | beats random | mean relative change | mean grad norm | mean cosine to full |",
        "|---|---:|---:|---:|---|---:|---:|---:|",
    ]
    for mode, values in summary["by_mode"].items():
        grad = values["gradient"]
        rand = values["random"]
        lines.append(
            "| {mode} | {improved}/{count} | {gprob:.6g} | {rprob:.6g} | `{beats}` | {rel:.6g} | {norm:.6g} | {cosine:.6g} |".format(
                mode=mode,
                improved=grad["improved_count"],
                count=grad["count"],
                gprob=grad["improvement_probability"],
                rprob=rand["improvement_probability"],
                beats=values["beats_random_aggregated"],
                rel=grad["mean_relative_objective_change"],
                norm=grad["mean_gradient_norm"],
                cosine=grad["mean_gradient_cosine_to_full"],
            )
        )
    lines.extend(
        [
            "",
            "## Per-Alpha Results",
            "",
            "| mode | alpha | gradient improved | gradient probability | random probability | mean relative change |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for mode, values in summary["by_mode"].items():
        for alpha in summary["alpha_grid"]:
            alpha_values = values["by_alpha"][str(alpha)]
            grad = alpha_values["gradient"]
            rand = alpha_values["random"]
            lines.append(
                "| {mode} | {alpha:.6g} | {improved}/{count} | {gprob:.6g} | {rprob:.6g} | {rel:.6g} |".format(
                    mode=mode,
                    alpha=alpha,
                    improved=grad["improved_count"],
                    count=grad["count"],
                    gprob=grad["improvement_probability"],
                    rprob=rand["improvement_probability"],
                    rel=grad["mean_relative_objective_change"],
                )
            )
    lines.extend(
        [
            "",
            "Machine-readable per-alpha summaries, including runtime and cosine fields, are stored in `summary.json` under each mode's `by_alpha` key.",
            "",
            "## Semantic Weights",
            "",
            "Diagnostic-only inverse-mean-magnitude weights were computed but not used to retune this amended objective.",
            "",
            "```json",
            json.dumps(summary["semantic_inverse_mean_weights_diagnostic_only"], indent=2, sort_keys=True),
            "```",
            "",
            "## Component Reporting",
            "",
            "Rows contain unweighted components and weighted contributions under the fixed amended objective weights.",
            "",
            "## Held-Out Isolation",
            "",
            f"- held-out used for selection: `{summary['held_out']['used_for_selection']}`",
            f"- held-out grad_enabled flags: `{[row['grad_enabled'] for row in summary['held_out']['rows']]}`",
            "",
        ]
    )
    return "\n".join(lines)


if __name__ == "__main__":
    main()
