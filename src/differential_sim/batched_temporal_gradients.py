"""Scenario-batched controller rollout with temporal gradient truncation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch

from differential_sim.controllers import HeadwayController
from differential_sim.idm import (
    IDMParameters,
    diffidm_acceleration,
    parameters_to_tensors,
    smooth_min_clamp,
)
from differential_sim.objectives import ObjectiveComponents, ObjectiveConfig
from differential_sim.rollout import RolloutConfig, RolloutResult
from differential_sim.scenarios import ScenarioConfig, leader_profile


@dataclass(frozen=True)
class BatchedScenarioData:
    scenario_names: tuple[str, ...]
    leader_x: torch.Tensor
    leader_v: torch.Tensor


@dataclass(frozen=True)
class BatchedObjectiveResult:
    aggregate: ObjectiveComponents
    per_scenario: ObjectiveComponents
    rollout: RolloutResult
    scenario_names: tuple[str, ...]


def build_batched_scenarios(
    scenario_configs: Sequence[tuple[str, ScenarioConfig]],
    *,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str = "cpu",
) -> BatchedScenarioData:
    if not scenario_configs:
        raise ValueError("scenario_configs must not be empty")
    profiles = [leader_profile(config, dtype=dtype, device=device) for _, config in scenario_configs]
    lengths = {int(profile.position.shape[0]) for profile in profiles}
    if len(lengths) != 1:
        raise ValueError("all scenarios must have the same rollout length")
    return BatchedScenarioData(
        scenario_names=tuple(name for name, _ in scenario_configs),
        leader_x=torch.stack([profile.position for profile in profiles], dim=0),
        leader_v=torch.stack([profile.speed for profile in profiles], dim=0),
    )


def rollout_batched_scenarios_with_controller(
    scenarios: BatchedScenarioData,
    *,
    beta: torch.Tensor,
    controller: HeadwayController,
    base_params: IDMParameters,
    rollout_config: RolloutConfig,
    horizon_k: int,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str = "cpu",
) -> RolloutResult:
    leader_x = scenarios.leader_x.to(dtype=dtype, device=device)
    leader_v = scenarios.leader_v.to(dtype=dtype, device=device)
    scenario_count, time_count = leader_x.shape
    steps = time_count - 1
    if horizon_k < 1 or horizon_k > steps:
        raise ValueError(f"horizon_k must be in [1, {steps}], got {horizon_k}")
    retain_start = max(0, steps - horizon_k)

    params_base = parameters_to_tensors(base_params, dtype=dtype, device=device)
    dt = torch.tensor(rollout_config.dt, dtype=dtype, device=device)
    leader_length = torch.tensor(rollout_config.leader_length, dtype=dtype, device=device)
    recurrent_x = torch.full(
        (scenario_count,),
        rollout_config.initial_follower.position,
        dtype=dtype,
        device=device,
    )
    recurrent_v = torch.full(
        (scenario_count,),
        rollout_config.initial_follower.speed,
        dtype=dtype,
        device=device,
    )
    follower_x = [recurrent_x]
    follower_v = [recurrent_v]
    follower_a: list[torch.Tensor] = []

    for t in range(steps):
        x_t = recurrent_x
        v_t = recurrent_v
        gap_t = leader_x[:, t] - x_t - leader_length
        delta_v_t = v_t - leader_v[:, t]
        inputs = torch.stack([v_t, delta_v_t, gap_t], dim=-1)
        time_headway = controller(beta, inputs)
        params = params_base.with_updates({"time_headway": time_headway})
        acc_t = diffidm_acceleration(
            v_follower=v_t,
            v_leader=leader_v[:, t],
            gap=gap_t,
            params=params,
            dt=dt,
            prevent_negative_speed=rollout_config.prevent_negative_speed,
        )
        v_next = v_t + dt * acc_t
        x_next = x_t + dt * v_next
        follower_a.append(acc_t)
        follower_v.append(v_next)
        follower_x.append(x_next)
        if t < retain_start:
            recurrent_x = x_next.detach()
            recurrent_v = v_next.detach()
        else:
            recurrent_x = x_next
            recurrent_v = v_next

    follower_x_t = torch.stack(follower_x, dim=1)
    follower_v_t = torch.stack(follower_v, dim=1)
    follower_a_t = torch.stack(follower_a, dim=1)
    gap = leader_x - follower_x_t - leader_length
    delta_v = follower_v_t - leader_v
    return RolloutResult(
        leader_x=leader_x,
        leader_v=leader_v,
        follower_x=follower_x_t,
        follower_v=follower_v_t,
        follower_a=follower_a_t,
        gap=gap,
        delta_v=delta_v,
    )


def batched_rollout_objective(
    result: RolloutResult,
    config: ObjectiveConfig = ObjectiveConfig(),
) -> tuple[ObjectiveComponents, ObjectiveComponents]:
    if result.follower_v.ndim != 2:
        raise ValueError("batched rollout tensors must have shape [S, T+1]")
    dtype = result.follower_v.dtype
    device = result.follower_v.device
    progress_scale = torch.tensor(config.progress_scale, dtype=dtype, device=device)
    safety_scale = torch.tensor(config.safety_scale, dtype=dtype, device=device)
    jerk_scale = torch.tensor(config.jerk_scale, dtype=dtype, device=device)

    v = result.follower_v[:, 1:]
    leader_v = result.leader_v[:, 1:]
    gap = result.gap[:, 1:]
    delta_v = result.delta_v[:, 1:]
    progress = torch.mean(torch.square((v - leader_v) / progress_scale), dim=1)

    s0 = torch.tensor(config.safety_s0, dtype=dtype, device=device)
    t_safe = torch.tensor(config.safety_time_headway, dtype=dtype, device=device)
    a_max = torch.tensor(config.safety_a_max, dtype=dtype, device=device)
    b_comfort = torch.tensor(config.safety_b_comfort, dtype=dtype, device=device)
    spacing = s0 + v * t_safe + v * delta_v / (2.0 * torch.sqrt(a_max * b_comfort))
    s_safe = smooth_min_clamp(spacing, 0.0)
    safety = torch.mean(torch.square(torch.nn.functional.softplus((s_safe - gap) / safety_scale)), dim=1)
    if result.follower_a.shape[1] < 2:
        jerk = torch.zeros(result.follower_a.shape[0], dtype=dtype, device=device)
    else:
        jerk = torch.mean(
            torch.square((result.follower_a[:, 1:] - result.follower_a[:, :-1]) / jerk_scale),
            dim=1,
        )

    weighted_progress = torch.tensor(config.progress_weight, dtype=dtype, device=device) * progress
    weighted_safety = torch.tensor(config.safety_weight, dtype=dtype, device=device) * safety
    weighted_jerk = torch.tensor(config.jerk_weight, dtype=dtype, device=device) * jerk
    total = weighted_progress + weighted_safety + weighted_jerk
    per_scenario = ObjectiveComponents(
        total=total,
        progress=progress,
        safety=safety,
        jerk=jerk,
        weighted_progress=weighted_progress,
        weighted_safety=weighted_safety,
        weighted_jerk=weighted_jerk,
    )
    aggregate = ObjectiveComponents(
        total=total.mean(),
        progress=progress.mean(),
        safety=safety.mean(),
        jerk=jerk.mean(),
        weighted_progress=weighted_progress.mean(),
        weighted_safety=weighted_safety.mean(),
        weighted_jerk=weighted_jerk.mean(),
    )
    return aggregate, per_scenario


def objective_for_batched_scenarios(
    scenario_configs: Sequence[tuple[str, ScenarioConfig]],
    *,
    beta: torch.Tensor,
    controller: HeadwayController,
    base_params: IDMParameters,
    rollout_config: RolloutConfig,
    horizon_k: int,
    objective_config: ObjectiveConfig = ObjectiveConfig(),
    dtype: torch.dtype = torch.float64,
    device: torch.device | str = "cpu",
) -> BatchedObjectiveResult:
    scenarios = build_batched_scenarios(scenario_configs, dtype=dtype, device=device)
    rollout = rollout_batched_scenarios_with_controller(
        scenarios,
        beta=beta,
        controller=controller,
        base_params=base_params,
        rollout_config=rollout_config,
        horizon_k=horizon_k,
        dtype=dtype,
        device=device,
    )
    aggregate, per_scenario = batched_rollout_objective(rollout, objective_config)
    return BatchedObjectiveResult(
        aggregate=aggregate,
        per_scenario=per_scenario,
        rollout=rollout,
        scenario_names=scenarios.scenario_names,
    )


def split_batched_rollout(result: RolloutResult) -> list[RolloutResult]:
    if result.follower_v.ndim != 2:
        raise ValueError("batched rollout tensors must have shape [S, T+1]")
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
