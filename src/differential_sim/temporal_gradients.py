"""Controller rollout with full and truncated temporal gradients."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch

from differential_sim.controllers import StructuredHeadwayController
from differential_sim.idm import IDMParameters, diffidm_acceleration, parameters_to_tensors
from differential_sim.objectives import ObjectiveComponents, ObjectiveConfig, mean_scenario_objective
from differential_sim.rollout import InitialFollowerState, RolloutConfig, RolloutResult
from differential_sim.scenarios import ScenarioConfig, leader_profile


@dataclass(frozen=True)
class ScenarioRollout:
    scenario_name: str
    result: RolloutResult


def rollout_with_controller(
    *,
    leader,
    beta: torch.Tensor,
    controller: StructuredHeadwayController,
    base_params: IDMParameters,
    rollout_config: RolloutConfig,
    horizon_k: int,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str = "cpu",
) -> RolloutResult:
    """Roll out with temporal detachment horizon ``horizon_k``.

    Forward values are identical for all horizons. For truncated horizons,
    connected same-step outputs are stored before recurrent state is detached.
    """

    leader_x = leader.position.to(dtype=dtype, device=device)
    leader_v = leader.speed.to(dtype=dtype, device=device)
    steps = leader_x.shape[0] - 1
    if horizon_k < 1 or horizon_k > steps:
        raise ValueError(f"horizon_k must be in [1, {steps}], got {horizon_k}")
    retain_start = max(0, steps - horizon_k)
    params_base = parameters_to_tensors(base_params, dtype=dtype, device=device)
    dt = torch.tensor(rollout_config.dt, dtype=dtype, device=device)
    leader_length = torch.tensor(rollout_config.leader_length, dtype=dtype, device=device)

    recurrent_x = torch.tensor(rollout_config.initial_follower.position, dtype=dtype, device=device)
    recurrent_v = torch.tensor(rollout_config.initial_follower.speed, dtype=dtype, device=device)
    follower_x = [recurrent_x]
    follower_v = [recurrent_v]
    follower_a: list[torch.Tensor] = []

    for t in range(steps):
        x_t = recurrent_x
        v_t = recurrent_v
        gap_t = leader_x[t] - x_t - leader_length
        delta_v_t = v_t - leader_v[t]
        inputs = torch.stack([v_t, delta_v_t, gap_t])
        time_headway = controller(beta, inputs)
        params = params_base.with_updates({"time_headway": time_headway})
        acc_t = diffidm_acceleration(
            v_follower=v_t,
            v_leader=leader_v[t],
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

    follower_x_t = torch.stack(follower_x)
    follower_v_t = torch.stack(follower_v)
    follower_a_t = torch.stack(follower_a)
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


def rollout_scenarios_with_controller(
    scenario_configs: Sequence[tuple[str, ScenarioConfig]],
    *,
    beta: torch.Tensor,
    controller: StructuredHeadwayController,
    base_params: IDMParameters,
    rollout_config: RolloutConfig,
    horizon_k: int,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str = "cpu",
) -> list[ScenarioRollout]:
    rollouts: list[ScenarioRollout] = []
    for name, config in scenario_configs:
        leader = leader_profile(config, dtype=dtype, device=device)
        result = rollout_with_controller(
            leader=leader,
            beta=beta,
            controller=controller,
            base_params=base_params,
            rollout_config=rollout_config,
            horizon_k=horizon_k,
            dtype=dtype,
            device=device,
        )
        rollouts.append(ScenarioRollout(scenario_name=name, result=result))
    return rollouts


def objective_for_scenarios(
    scenario_configs: Sequence[tuple[str, ScenarioConfig]],
    *,
    beta: torch.Tensor,
    controller: StructuredHeadwayController,
    base_params: IDMParameters,
    rollout_config: RolloutConfig,
    horizon_k: int,
    objective_config: ObjectiveConfig = ObjectiveConfig(),
    dtype: torch.dtype = torch.float64,
    device: torch.device | str = "cpu",
) -> tuple[ObjectiveComponents, list[ScenarioRollout]]:
    rollouts = rollout_scenarios_with_controller(
        scenario_configs,
        beta=beta,
        controller=controller,
        base_params=base_params,
        rollout_config=rollout_config,
        horizon_k=horizon_k,
        dtype=dtype,
        device=device,
    )
    components = mean_scenario_objective([rollout.result for rollout in rollouts], objective_config)
    return components, rollouts


def max_forward_difference(a: RolloutResult, b: RolloutResult) -> float:
    max_diff = 0.0
    for field in ("follower_x", "follower_v", "follower_a", "gap", "delta_v"):
        diff = torch.max(torch.abs(getattr(a, field).detach() - getattr(b, field).detach()))
        max_diff = max(max_diff, float(diff.cpu().item()))
    return max_diff


def default_rollout_config() -> RolloutConfig:
    return RolloutConfig(
        dt=0.2,
        leader_length=5.0,
        initial_follower=InitialFollowerState(position=-22.0, speed=16.0),
        acceleration_mode="diffidm",
        prevent_negative_speed=True,
    )
