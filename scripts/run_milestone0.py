"""Run the approved Milestone 0 smoke reproduction.

This script writes:
- reports/milestone0/results.json
- reports/milestone0/summary.md
"""

from __future__ import annotations

import json
from importlib import metadata
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from differential_sim.fit import (
    FitConfig,
    fit_synthetic_parameters,
    generate_synthetic_trajectory,
    normalized_parameter_distance,
)
from differential_sim.idm import IDMParameters
from differential_sim.rollout import RolloutConfig
from differential_sim.scenarios import ScenarioConfig, leader_profile


REPORT_DIR = ROOT / "reports" / "milestone0"


def main() -> None:
    dtype = torch.float64
    device = torch.device("cpu")
    seed = 0
    torch.manual_seed(seed)

    scenario_config = ScenarioConfig(kind="braking_recovery", steps=80, dt=0.2)
    rollout_config = RolloutConfig(dt=scenario_config.dt, acceleration_mode="diffidm")
    truth = IDMParameters(
        a_max=1.4,
        b_comfort=2.0,
        v0=28.0,
        s0=2.0,
        time_headway=1.4,
        a_min=-8.0,
    )
    initial_values = {"time_headway": 2.2, "v0": 20.0}
    truth_values = {"time_headway": 1.4, "v0": 28.0}
    base = IDMParameters(
        a_max=truth.a_max,
        b_comfort=truth.b_comfort,
        v0=truth.v0,
        s0=truth.s0,
        time_headway=truth.time_headway,
        a_min=truth.a_min,
    )

    leader = leader_profile(scenario_config, dtype=dtype, device=device)
    reference = generate_synthetic_trajectory(
        leader,
        truth,
        rollout_config,
        dtype=dtype,
        device=device,
    )
    fit_config = FitConfig(fitted_names=("time_headway", "v0"), steps=300, learning_rate=0.03, seed=seed)
    fit_result = fit_synthetic_parameters(
        leader=leader,
        reference=reference,
        base_params=base,
        truth_values=truth_values,
        initial_values=initial_values,
        rollout_config=rollout_config,
        fit_config=fit_config,
        dtype=dtype,
        device=device,
    )

    initial_distance = normalized_parameter_distance(
        fit_result.initial_parameters,
        fit_result.truth_parameters,
        fit_config.fitted_names,
    )
    final_distance = normalized_parameter_distance(
        fit_result.final_parameters,
        fit_result.truth_parameters,
        fit_config.fitted_names,
    )

    results = {
        "environment": {
            "python": sys.version,
            "torch": torch.__version__,
            "diffidm": metadata.version("diffidm"),
            "dtype": str(dtype),
            "device": str(device),
        },
        "seed": seed,
        "scenario": scenario_config.__dict__,
        "rollout": {
            "dt": rollout_config.dt,
            "leader_length": rollout_config.leader_length,
            "initial_follower": rollout_config.initial_follower.__dict__,
            "acceleration_mode": rollout_config.acceleration_mode,
            "prevent_negative_speed": rollout_config.prevent_negative_speed,
        },
        "fit": {
            "fitted_names": fit_config.fitted_names,
            "steps": fit_config.steps,
            "learning_rate": fit_config.learning_rate,
            "initial_loss": fit_result.initial_loss,
            "final_loss": fit_result.final_loss,
            "loss_reduction_fraction": 1.0 - fit_result.final_loss / fit_result.initial_loss,
            "initial_parameters": fit_result.initial_parameters,
            "final_parameters": fit_result.final_parameters,
            "truth_parameters": fit_result.truth_parameters,
            "initial_normalized_distance": initial_distance,
            "final_normalized_distance": final_distance,
            "gradients_finite": fit_result.gradients_finite,
        },
    }

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "results.json").write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")
    (REPORT_DIR / "summary.md").write_text(_summary(results))


def _summary(results: dict) -> str:
    fit = results["fit"]
    return "\n".join(
        [
            "# Milestone 0 Reproduction Summary",
            "",
            "## Environment",
            "",
            f"- torch: `{results['environment']['torch']}`",
            f"- diffidm: `{results['environment']['diffidm']}`",
            f"- dtype: `{results['environment']['dtype']}`",
            f"- device: `{results['environment']['device']}`",
            "",
            "## Primary Smoke Fit",
            "",
            "- synthetic trajectory path: `diffidm` smooth-clamped rollout",
            "- transparency path: textbook IDM helpers and smooth-clamped reference helpers",
            f"- fitted parameters: `{', '.join(fit['fitted_names'])}`",
            f"- optimization steps: `{fit['steps']}`",
            f"- initial loss: `{fit['initial_loss']:.12g}`",
            f"- final loss: `{fit['final_loss']:.12g}`",
            f"- loss reduction fraction: `{fit['loss_reduction_fraction']:.6f}`",
            f"- initial normalized distance: `{fit['initial_normalized_distance']:.12g}`",
            f"- final normalized distance: `{fit['final_normalized_distance']:.12g}`",
            f"- gradients finite: `{fit['gradients_finite']}`",
            "",
            "## Reproduce",
            "",
            "```bash",
            "/home/zpz/miniconda3/envs/differential_sim/bin/python -m pytest",
            "/home/zpz/miniconda3/envs/differential_sim/bin/python scripts/run_milestone0.py",
            "```",
            "",
        ]
    )


if __name__ == "__main__":
    main()
