"""One-step descent diagnostics for Milestone 1."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
import time
from typing import Iterable, Literal, Sequence

import torch

from differential_sim.controllers import (
    HeadwayBounds,
    InputNormalization,
    InputParameterization,
    StructuredHeadwayController,
    input_normalization_from_scenarios,
    noisy_center_betas,
    tensor_to_list,
)
from differential_sim.temporal_gradients import default_rollout_config, objective_for_scenarios
from differential_sim.objectives import (
    ObjectiveConfig,
    component_floats,
    default_idm_params,
    semantic_inverse_mean_weights,
    weighted_component_floats,
)
from differential_sim.rollout import RolloutConfig
from differential_sim.scenarios import ScenarioConfig, random_braking_cycle_events


DirectionType = Literal["gradient", "random"]


ALPHA_GRID = (0.001, 0.002, 0.004, 0.01, 0.02, 0.04, 0.1, 0.3, 1.0)
PRIMARY_ALPHA = 0.1
HORIZONS = (1, 3, 6, 10, 80)


@dataclass(frozen=True)
class DiagnosticConfig:
    random_direction_count: int = 32
    random_direction_seed: int = 10000
    alpha_grid: tuple[float, ...] = ALPHA_GRID
    primary_alpha: float = PRIMARY_ALPHA
    horizons: tuple[int, ...] = HORIZONS
    dtype_name: str = "torch.float64"
    device: str = "cpu"


@dataclass(frozen=True)
class DiagnosticArtifacts:
    rows: list[dict]
    summary: dict
    held_out: dict


def diagnostic_scenarios() -> list[tuple[str, ScenarioConfig]]:
    return [
        ("constant_16", ScenarioConfig(kind="constant", steps=80, dt=0.2, initial_speed=16.0)),
        ("constant_20", ScenarioConfig(kind="constant", steps=80, dt=0.2, initial_speed=20.0)),
        (
            "brake_mild",
            ScenarioConfig(
                kind="braking_recovery",
                steps=80,
                dt=0.2,
                initial_speed=18.0,
                brake_delta_v=3.0,
                brake_start=4.0,
                brake_duration=2.0,
                recovery_duration=3.0,
            ),
        ),
        (
            "brake_stronger",
            ScenarioConfig(
                kind="braking_recovery",
                steps=80,
                dt=0.2,
                initial_speed=18.0,
                brake_delta_v=5.0,
                brake_start=4.0,
                brake_duration=2.0,
                recovery_duration=3.0,
            ),
        ),
        (
            "sinusoidal_low",
            ScenarioConfig(
                kind="sinusoidal",
                steps=80,
                dt=0.2,
                initial_speed=18.0,
                sinusoid_amplitude=1.5,
                sinusoid_period=8.0,
            ),
        ),
        (
            "sinusoidal_high",
            ScenarioConfig(
                kind="sinusoidal",
                steps=80,
                dt=0.2,
                initial_speed=18.0,
                sinusoid_amplitude=2.5,
                sinusoid_period=10.0,
            ),
        ),
        (
            "random_brake_0",
            ScenarioConfig(
                kind="random_braking_cycles",
                steps=80,
                dt=0.2,
                seed=101,
                initial_speed_range=(16.0, 20.0),
                upper_target_speed_range=(18.0, 24.0),
                post_brake_speed_range=(10.0, 18.0),
                minimum_speed_drop=3.0,
                acceleration_magnitude_range=(0.4, 1.2),
                braking_magnitude_range=(0.8, 2.2),
                hold_duration_range=(1.0, 3.0),
                post_brake_hold_duration_range=(0.6, 1.6),
                speed_floor=6.0,
                speed_ceiling=26.0,
            ),
        ),
        (
            "random_brake_1",
            ScenarioConfig(
                kind="random_braking_cycles",
                steps=80,
                dt=0.2,
                seed=102,
                initial_speed_range=(16.0, 20.0),
                upper_target_speed_range=(18.0, 24.0),
                post_brake_speed_range=(10.0, 18.0),
                minimum_speed_drop=3.0,
                acceleration_magnitude_range=(0.4, 1.2),
                braking_magnitude_range=(0.8, 2.2),
                hold_duration_range=(1.0, 3.0),
                post_brake_hold_duration_range=(0.6, 1.6),
                speed_floor=6.0,
                speed_ceiling=26.0,
            ),
        ),
        (
            "random_brake_2",
            ScenarioConfig(
                kind="random_braking_cycles",
                steps=80,
                dt=0.2,
                seed=103,
                initial_speed_range=(16.0, 20.0),
                upper_target_speed_range=(18.0, 24.0),
                post_brake_speed_range=(10.0, 18.0),
                minimum_speed_drop=3.0,
                acceleration_magnitude_range=(0.4, 1.2),
                braking_magnitude_range=(0.8, 2.2),
                hold_duration_range=(1.0, 3.0),
                post_brake_hold_duration_range=(0.6, 1.6),
                speed_floor=6.0,
                speed_ceiling=26.0,
            ),
        ),
        (
            "multipulse_mild",
            ScenarioConfig(
                kind="multi_pulse_braking",
                steps=80,
                dt=0.2,
                initial_speed=18.0,
                pulse_starts=(2.5, 6.5, 10.5),
                pulse_brake_delta_v=(2.0, 3.0, 2.5),
                pulse_brake_durations=(1.0, 1.2, 1.4),
                pulse_recovery_durations=(1.5, 1.95, 2.4),
                speed_floor=6.0,
                speed_ceiling=26.0,
            ),
        ),
        (
            "multipulse_varied",
            ScenarioConfig(
                kind="multi_pulse_braking",
                steps=80,
                dt=0.2,
                initial_speed=18.0,
                pulse_starts=(2.0, 5.8, 11.0),
                pulse_brake_delta_v=(2.5, 4.0, 3.0),
                pulse_brake_durations=(1.0, 1.2, 1.4),
                pulse_recovery_durations=(1.5, 1.95, 2.4),
                speed_floor=6.0,
                speed_ceiling=26.0,
            ),
        ),
        (
            "chirp_low",
            ScenarioConfig(
                kind="chirp_sinusoidal",
                steps=80,
                dt=0.2,
                base_speed=18.0,
                sinusoid_amplitude=1.5,
                period_start=10.0,
                period_end=4.0,
                speed_floor=8.0,
                speed_ceiling=26.0,
            ),
        ),
        (
            "chirp_high",
            ScenarioConfig(
                kind="chirp_sinusoidal",
                steps=80,
                dt=0.2,
                base_speed=18.0,
                sinusoid_amplitude=2.5,
                period_start=10.0,
                period_end=4.0,
                speed_floor=8.0,
                speed_ceiling=26.0,
            ),
        ),
        (
            "mixed_regime",
            ScenarioConfig(
                kind="mixed_regime",
                steps=80,
                dt=0.2,
                base_speed=18.0,
                first_brake_delta_v=2.5,
                first_brake_duration=1.4,
                first_recovery_duration=1.6,
                sinusoid_amplitude=1.2,
                sinusoid_period=3.5,
                second_brake_delta_v=4.0,
                second_brake_duration=1.2,
                second_recovery_duration=2.0,
                speed_floor=8.0,
                speed_ceiling=24.0,
            ),
        ),
    ]


def held_out_scenarios() -> list[tuple[str, ScenarioConfig]]:
    return [
        ("constant_18", ScenarioConfig(kind="constant", steps=80, dt=0.2, initial_speed=18.0)),
        (
            "brake_shifted",
            ScenarioConfig(
                kind="braking_recovery",
                steps=80,
                dt=0.2,
                initial_speed=18.0,
                brake_delta_v=4.0,
                brake_start=5.0,
                brake_duration=2.0,
                recovery_duration=3.0,
            ),
        ),
        (
            "sinusoidal_period6",
            ScenarioConfig(
                kind="sinusoidal",
                steps=80,
                dt=0.2,
                initial_speed=18.0,
                sinusoid_amplitude=2.0,
                sinusoid_period=6.0,
            ),
        ),
        (
            "heldout_random_brake_0",
            ScenarioConfig(
                kind="random_braking_cycles",
                steps=80,
                dt=0.2,
                seed=201,
                initial_speed_range=(15.0, 21.0),
                upper_target_speed_range=(19.0, 25.0),
                post_brake_speed_range=(9.0, 17.0),
                minimum_speed_drop=4.0,
                acceleration_magnitude_range=(0.5, 1.4),
                braking_magnitude_range=(1.0, 2.5),
                hold_duration_range=(0.8, 2.5),
                post_brake_hold_duration_range=(0.5, 1.4),
                speed_floor=6.0,
                speed_ceiling=27.0,
            ),
        ),
        (
            "heldout_random_brake_1",
            ScenarioConfig(
                kind="random_braking_cycles",
                steps=80,
                dt=0.2,
                seed=202,
                initial_speed_range=(15.0, 21.0),
                upper_target_speed_range=(19.0, 25.0),
                post_brake_speed_range=(9.0, 17.0),
                minimum_speed_drop=4.0,
                acceleration_magnitude_range=(0.5, 1.4),
                braking_magnitude_range=(1.0, 2.5),
                hold_duration_range=(0.8, 2.5),
                post_brake_hold_duration_range=(0.5, 1.4),
                speed_floor=6.0,
                speed_ceiling=27.0,
            ),
        ),
        (
            "heldout_multipulse_shifted",
            ScenarioConfig(
                kind="multi_pulse_braking",
                steps=80,
                dt=0.2,
                initial_speed=18.0,
                pulse_starts=(3.0, 7.2, 12.0),
                pulse_brake_delta_v=(3.0, 3.5, 4.0),
                pulse_brake_durations=(1.2, 1.2, 1.5),
                pulse_recovery_durations=(1.8, 2.0, 2.2),
                speed_floor=6.0,
                speed_ceiling=26.0,
            ),
        ),
        (
            "heldout_chirp_shifted",
            ScenarioConfig(
                kind="chirp_sinusoidal",
                steps=80,
                dt=0.2,
                base_speed=18.0,
                sinusoid_amplitude=2.0,
                period_start=8.0,
                period_end=3.5,
                speed_floor=8.0,
                speed_ceiling=26.0,
            ),
        ),
        (
            "heldout_mixed_regime_shifted",
            ScenarioConfig(
                kind="mixed_regime",
                steps=80,
                dt=0.2,
                base_speed=19.0,
                first_brake_delta_v=3.0,
                first_brake_duration=1.2,
                first_recovery_duration=1.8,
                sinusoid_amplitude=1.5,
                sinusoid_period=3.0,
                second_brake_delta_v=3.5,
                second_brake_duration=1.4,
                second_recovery_duration=2.2,
                speed_floor=8.0,
                speed_ceiling=25.0,
            ),
        ),
    ]


def build_controller(
    input_parameterization: InputParameterization,
    normalization: InputNormalization | None,
) -> StructuredHeadwayController:
    return StructuredHeadwayController(
        bounds=HeadwayBounds(minimum=0.5, maximum=3.0),
        normalization=normalization if input_parameterization == "normalized" else None,
        input_parameterization=input_parameterization,
    )


def run_one_step_diagnostics(
    *,
    input_parameterization: InputParameterization,
    random_direction_count: int = 32,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str = "cpu",
) -> DiagnosticArtifacts:
    rollout_config = default_rollout_config()
    base_params = default_idm_params(time_headway=1.4)
    train = diagnostic_scenarios()
    held_out = held_out_scenarios()
    normalization = None
    if input_parameterization == "normalized":
        normalization = input_normalization_from_scenarios(
            [config for _, config in train],
            base_params,
            rollout_config,
            dtype=dtype,
            device=device,
        )
    controller = build_controller(input_parameterization, normalization)
    init_betas = noisy_center_betas(dtype=dtype, device=device)
    objective_config = ObjectiveConfig()

    semantic_rows = []
    for beta in init_betas:
        with torch.no_grad():
            components, _ = objective_for_scenarios(
                train,
                beta=beta,
                controller=controller,
                base_params=base_params,
                rollout_config=rollout_config,
                horizon_k=80,
                objective_config=objective_config,
                dtype=dtype,
                device=device,
            )
            semantic_rows.append(component_floats(components))

    rows: list[dict] = []
    random_vectors = {}
    candidate_cache: dict[tuple[int, str, str, float], dict] = {}
    for init_index, beta_init in enumerate(init_betas):
        random_dirs = generate_random_directions(
            count=random_direction_count,
            init_index=init_index,
            input_parameterization=input_parameterization,
            dtype=dtype,
            device=device,
        )
        random_vectors[str(init_index)] = [tensor_to_list(direction) for direction in random_dirs]
        gradient_info = {}
        for horizon in HORIZONS:
            beta = beta_init.detach().clone().requires_grad_(True)
            start = time.perf_counter()
            components_before, rollouts_before = objective_for_scenarios(
                train,
                beta=beta,
                controller=controller,
                base_params=base_params,
                rollout_config=rollout_config,
                horizon_k=horizon,
                objective_config=objective_config,
                dtype=dtype,
                device=device,
            )
            grad = torch.autograd.grad(components_before.total, beta, allow_unused=False)[0]
            runtime_s = time.perf_counter() - start
            grad_norm = torch.linalg.vector_norm(grad)
            zero_gradient = bool((grad_norm <= 1e-12).detach().cpu().item())
            if zero_gradient:
                direction = torch.zeros_like(grad)
            else:
                direction = -grad / (grad_norm + 1e-12)
            gradient_info[horizon] = {
                "components_before": components_before,
                "rollouts_before": rollouts_before,
                "grad": grad.detach(),
                "grad_norm": float(grad_norm.detach().cpu().item()),
                "zero_gradient": zero_gradient,
                "direction": direction.detach(),
                "runtime_s": runtime_s,
            }
        full_grad = gradient_info[80]["grad"]
        for horizon in HORIZONS:
            info = gradient_info[horizon]
            cosine = gradient_cosine_similarity(info["grad"], full_grad)
            rows.extend(
                evaluate_direction_grid(
                    direction=info["direction"],
                    direction_type="gradient",
                    direction_id=f"K{horizon}",
                    beta_init=beta_init,
                    init_index=init_index,
                    horizon=horizon,
                    components_before=component_floats(info["components_before"]),
                    weighted_components_before=weighted_component_floats(info["components_before"]),
                    rollouts_before=info["rollouts_before"],
                    grad_norm=info["grad_norm"],
                    gradient_cosine_to_full=cosine,
                    zero_gradient=info["zero_gradient"],
                    runtime_s=info["runtime_s"],
                    train=train,
                    controller=controller,
                    base_params=base_params,
                    rollout_config=rollout_config,
                    objective_config=objective_config,
                    input_parameterization=input_parameterization,
                    candidate_cache=candidate_cache,
                    dtype=dtype,
                    device=device,
                )
            )
            for random_index, random_direction in enumerate(random_dirs):
                rows.extend(
                    evaluate_direction_grid(
                        direction=random_direction,
                        direction_type="random",
                        direction_id=f"random_{random_index}",
                        beta_init=beta_init,
                        init_index=init_index,
                        horizon=horizon,
                        components_before=component_floats(info["components_before"]),
                        weighted_components_before=weighted_component_floats(info["components_before"]),
                        rollouts_before=info["rollouts_before"],
                        grad_norm=info["grad_norm"],
                        gradient_cosine_to_full=math.nan,
                        zero_gradient=False,
                        runtime_s=info["runtime_s"],
                        train=train,
                        controller=controller,
                        base_params=base_params,
                        rollout_config=rollout_config,
                        objective_config=objective_config,
                        input_parameterization=input_parameterization,
                        candidate_cache=candidate_cache,
                        dtype=dtype,
                        device=device,
                    )
                )

    held_out_report = evaluate_held_out(
        held_out,
        init_betas=init_betas,
        controller=controller,
        base_params=base_params,
        rollout_config=rollout_config,
        objective_config=objective_config,
        input_parameterization=input_parameterization,
        dtype=dtype,
        device=device,
    )
    summary = summarize_rows(
        rows,
        input_parameterization=input_parameterization,
        normalization=normalization,
        init_betas=init_betas,
        random_vectors=random_vectors,
        semantic_rows=semantic_rows,
        random_direction_count=random_direction_count,
        rollout_config=rollout_config,
        base_params=base_params,
        objective_config=objective_config,
    )
    return DiagnosticArtifacts(rows=rows, summary=summary, held_out=held_out_report)


def generate_random_directions(
    *,
    count: int,
    init_index: int,
    input_parameterization: InputParameterization,
    dtype: torch.dtype,
    device: torch.device | str,
) -> list[torch.Tensor]:
    seed_offset = 0 if input_parameterization == "normalized" else 100000
    generator = torch.Generator(device=str(device) if str(device) != "cpu" else "cpu")
    generator.manual_seed(10000 + seed_offset + 1000 * init_index)
    directions = []
    for _ in range(count):
        raw = torch.randn(4, generator=generator, dtype=dtype, device=device)
        directions.append(raw / (torch.linalg.vector_norm(raw) + 1e-12))
    return directions


def evaluate_direction_grid(
    *,
    direction: torch.Tensor,
    direction_type: DirectionType,
    direction_id: str,
    beta_init: torch.Tensor,
    init_index: int,
    horizon: int,
    components_before: dict[str, float],
    weighted_components_before: dict[str, float],
    rollouts_before,
    grad_norm: float,
    gradient_cosine_to_full: float,
    zero_gradient: bool,
    runtime_s: float,
    train: Sequence[tuple[str, ScenarioConfig]],
    controller: StructuredHeadwayController,
    base_params,
    rollout_config: RolloutConfig,
    objective_config: ObjectiveConfig,
    input_parameterization: InputParameterization,
    candidate_cache: dict[tuple[int, str, str, float], dict] | None,
    dtype: torch.dtype,
    device: torch.device | str,
) -> list[dict]:
    rows: list[dict] = []
    before_total = components_before["total"]
    min_gap_before, min_speed_before = min_gap_and_speed([r.result for r in rollouts_before])
    for alpha in ALPHA_GRID:
        unsafe = False
        improved = False
        rel_change = math.nan
        after_components = {"total": math.nan, "progress": math.nan, "safety": math.nan, "jerk": math.nan}
        weighted_after_components = {"progress": math.nan, "safety": math.nan, "jerk": math.nan}
        min_gap_after = math.nan
        min_speed_after = math.nan
        if not zero_gradient or direction_type == "random":
            cache_key = (init_index, direction_type, direction_id, alpha)
            cached = candidate_cache.get(cache_key) if candidate_cache is not None else None
            if cached is not None:
                unsafe = cached["unsafe"]
                improved = cached["improved"]
                rel_change = cached["rel_change"]
                after_components = cached["after_components"]
                weighted_after_components = cached["weighted_after_components"]
                min_gap_after = cached["min_gap_after"]
                min_speed_after = cached["min_speed_after"]
            else:
                with torch.no_grad():
                    candidate = beta_init + alpha * direction
                    try:
                        components_after, rollouts_after = objective_for_scenarios(
                            train,
                            beta=candidate,
                            controller=controller,
                            base_params=base_params,
                            rollout_config=rollout_config,
                            horizon_k=80,
                            objective_config=objective_config,
                            dtype=dtype,
                            device=device,
                        )
                        after_components = component_floats(components_after)
                        weighted_after_components = weighted_component_floats(components_after)
                        min_gap_after, min_speed_after = min_gap_and_speed([r.result for r in rollouts_after])
                        finite = all(math.isfinite(value) for value in after_components.values())
                        unsafe = (not finite) or (not math.isfinite(min_gap_after)) or (not math.isfinite(min_speed_after))
                        if finite and math.isfinite(before_total) and before_total != 0.0:
                            rel_change = (after_components["total"] - before_total) / abs(before_total)
                            improved = after_components["total"] < before_total
                    except (RuntimeError, ValueError, FloatingPointError):
                        unsafe = True
                if candidate_cache is not None:
                    candidate_cache[cache_key] = {
                        "unsafe": unsafe,
                        "improved": improved,
                        "rel_change": rel_change,
                        "after_components": after_components,
                        "weighted_after_components": weighted_after_components,
                        "min_gap_after": min_gap_after,
                        "min_speed_after": min_speed_after,
                    }
        rows.append(
            {
                "input_parameterization": input_parameterization,
                "mode": f"K={horizon}",
                "K": horizon,
                "initialization_id": init_index,
                "direction_type": direction_type,
                "direction_id": direction_id,
                "alpha": alpha,
                "primary_alpha": alpha == PRIMARY_ALPHA,
                "objective_before": before_total,
                "objective_after": after_components["total"],
                "relative_objective_change": rel_change,
                "improved": improved,
                "components_before": components_before,
                "components_after": after_components,
                "weighted_components_before": weighted_components_before,
                "weighted_components_after": weighted_after_components,
                "gradient_norm": grad_norm,
                "gradient_cosine_to_full": gradient_cosine_to_full,
                "finite_gradient": math.isfinite(grad_norm),
                "zero_gradient": zero_gradient if direction_type == "gradient" else False,
                "unsafe_or_nonfinite_update": unsafe,
                "runtime_s": runtime_s,
                "min_gap_before": min_gap_before,
                "min_speed_before": min_speed_before,
                "min_gap_after": min_gap_after,
                "min_speed_after": min_speed_after,
                "scenario_count": len(train),
            }
        )
    return rows


def evaluate_held_out(
    scenarios: Sequence[tuple[str, ScenarioConfig]],
    *,
    init_betas: Sequence[torch.Tensor],
    controller: StructuredHeadwayController,
    base_params,
    rollout_config: RolloutConfig,
    objective_config: ObjectiveConfig,
    input_parameterization: InputParameterization,
    dtype: torch.dtype,
    device: torch.device | str,
) -> dict:
    rows = []
    with torch.no_grad():
        for init_index, beta in enumerate(init_betas):
            components, rollouts = objective_for_scenarios(
                scenarios,
                beta=beta,
                controller=controller,
                base_params=base_params,
                rollout_config=rollout_config,
                horizon_k=80,
                objective_config=objective_config,
                dtype=dtype,
                device=device,
            )
            min_gap, min_speed = min_gap_and_speed([r.result for r in rollouts])
            rows.append(
                {
                    "input_parameterization": input_parameterization,
                    "initialization_id": init_index,
                    "components": component_floats(components),
                    "weighted_components": weighted_component_floats(components),
                    "min_gap": min_gap,
                    "min_speed": min_speed,
                    "grad_enabled": torch.is_grad_enabled(),
                    "scenario_count": len(scenarios),
                }
            )
    return {"rows": rows, "scenarios": scenario_configs_to_dicts(scenarios), "used_for_selection": False}


def summarize_rows(
    rows: Sequence[dict],
    *,
    input_parameterization: InputParameterization,
    normalization: InputNormalization | None,
    init_betas: Sequence[torch.Tensor],
    random_vectors: dict,
    semantic_rows: Sequence[dict],
    random_direction_count: int,
    rollout_config: RolloutConfig,
    base_params,
    objective_config: ObjectiveConfig,
) -> dict:
    primary = [row for row in rows if row["primary_alpha"]]
    by_mode = {}
    random_primary = [row for row in primary if row["direction_type"] == "random"]
    random_prob = _improvement_probability(random_primary)
    for horizon in HORIZONS:
        grad_rows = [row for row in primary if row["direction_type"] == "gradient" and row["K"] == horizon]
        rand_rows = [row for row in primary if row["direction_type"] == "random" and row["K"] == horizon]
        by_mode[f"K={horizon}"] = {
            "gradient": _row_summary(grad_rows),
            "random": _row_summary(rand_rows),
            "by_alpha": {
                str(alpha): {
                    "gradient": _row_summary(
                        [
                            row
                            for row in rows
                            if row["direction_type"] == "gradient" and row["K"] == horizon and row["alpha"] == alpha
                        ]
                    ),
                    "random": _row_summary(
                        [
                            row
                            for row in rows
                            if row["direction_type"] == "random" and row["K"] == horizon and row["alpha"] == alpha
                        ]
                    ),
                }
                for alpha in ALPHA_GRID
            },
            "beats_random_aggregated": _improvement_probability(grad_rows) > random_prob,
        }
    return {
        "input_parameterization": input_parameterization,
        "primary_alpha": PRIMARY_ALPHA,
        "alpha_grid": list(ALPHA_GRID),
        "horizons": list(HORIZONS),
        "random_direction_count": random_direction_count,
        "random_direction_seed": 10000,
        "initial_betas": [tensor_to_list(beta) for beta in init_betas],
        "random_direction_vectors": random_vectors,
        "normalization": None
        if normalization is None
        else {
            "mean": tensor_to_list(normalization.mean),
            "sigma": tensor_to_list(normalization.sigma),
            "sigma_floor_used": normalization.sigma_floor_used,
        },
        "semantic_inverse_mean_weights_diagnostic_only": semantic_inverse_mean_weights(semantic_rows),
        "semantic_component_rows": list(semantic_rows),
        "objective_weights": {
            "progress": objective_config.progress_weight,
            "safety": objective_config.safety_weight,
            "jerk": objective_config.jerk_weight,
        },
        "by_mode": by_mode,
        "random_aggregated_primary_probability": random_prob,
        "zero_gradient_count": sum(1 for row in primary if row["direction_type"] == "gradient" and row["zero_gradient"]),
        "unsafe_or_nonfinite_count": sum(1 for row in rows if row["unsafe_or_nonfinite_update"]),
        "scenarios": {
            "diagnostic": scenario_configs_to_dicts(diagnostic_scenarios()),
            "held_out": scenario_configs_to_dicts(held_out_scenarios()),
        },
        "rollout": {
            "dt": rollout_config.dt,
            "leader_length": rollout_config.leader_length,
            "initial_follower": asdict(rollout_config.initial_follower),
            "prevent_negative_speed": rollout_config.prevent_negative_speed,
        },
        "idm_parameters": asdict(base_params),
    }


def _row_summary(rows: Sequence[dict]) -> dict:
    rel = [row["relative_objective_change"] for row in rows if math.isfinite(row["relative_objective_change"])]
    norms = [row["gradient_norm"] for row in rows if math.isfinite(row["gradient_norm"])]
    cosines = [
        row["gradient_cosine_to_full"]
        for row in rows
        if row["direction_type"] == "gradient" and math.isfinite(row["gradient_cosine_to_full"])
    ]
    return {
        "count": len(rows),
        "improved_count": sum(1 for row in rows if row["improved"]),
        "improvement_probability": _improvement_probability(rows),
        "mean_relative_objective_change": float(sum(rel) / len(rel)) if rel else math.nan,
        "mean_gradient_norm": float(sum(norms) / len(norms)) if norms else math.nan,
        "mean_gradient_cosine_to_full": float(sum(cosines) / len(cosines)) if cosines else math.nan,
        "zero_gradient_count": sum(1 for row in rows if row["zero_gradient"]),
        "unsafe_or_nonfinite_count": sum(1 for row in rows if row["unsafe_or_nonfinite_update"]),
        "mean_runtime_s": float(sum(row["runtime_s"] for row in rows) / len(rows)) if rows else math.nan,
    }


def _improvement_probability(rows: Sequence[dict]) -> float:
    if not rows:
        return math.nan
    return sum(1 for row in rows if row["improved"]) / len(rows)


def min_gap_and_speed(results: Iterable) -> tuple[float, float]:
    min_gap = math.inf
    min_speed = math.inf
    for result in results:
        min_gap = min(min_gap, float(torch.min(result.gap.detach()).cpu().item()))
        min_speed = min(min_speed, float(torch.min(result.follower_v.detach()).cpu().item()))
    return min_gap, min_speed


def gradient_cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    a_norm = torch.linalg.vector_norm(a)
    b_norm = torch.linalg.vector_norm(b)
    if bool((a_norm <= 1e-12).detach().cpu().item()) or bool((b_norm <= 1e-12).detach().cpu().item()):
        return math.nan
    cosine = torch.dot(a.reshape(-1), b.reshape(-1)) / (a_norm * b_norm)
    return float(cosine.detach().cpu().item())


def scenario_configs_to_dicts(scenarios: Sequence[tuple[str, ScenarioConfig]]) -> list[dict]:
    rows = []
    for name, config in scenarios:
        row = {"name": name, **asdict(config)}
        events = random_braking_cycle_events(config)
        if events:
            row["sampled_random_braking_events"] = events
        rows.append(row)
    return rows


def write_jsonl(path, rows: Sequence[dict]) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
