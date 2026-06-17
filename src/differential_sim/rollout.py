"""One-leader, one-follower deterministic IDM rollout."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch

from differential_sim.idm import (
    IDMParameters,
    diffidm_acceleration,
    parameters_to_tensors,
    smooth_clamped_idm_acceleration_reference,
    textbook_idm_acceleration,
)
from differential_sim.scenarios import LeaderProfile


AccelerationMode = Literal["diffidm", "smooth_reference", "textbook"]


@dataclass(frozen=True)
class InitialFollowerState:
    position: float
    speed: float


@dataclass(frozen=True)
class RolloutConfig:
    dt: float = 0.2
    leader_length: float = 5.0
    initial_follower: InitialFollowerState = InitialFollowerState(position=-22.0, speed=16.0)
    acceleration_mode: AccelerationMode = "diffidm"
    prevent_negative_speed: bool = True


@dataclass(frozen=True)
class RolloutResult:
    leader_x: torch.Tensor
    leader_v: torch.Tensor
    follower_x: torch.Tensor
    follower_v: torch.Tensor
    follower_a: torch.Tensor
    gap: torch.Tensor
    delta_v: torch.Tensor


def rollout_follower(
    leader: LeaderProfile,
    params: IDMParameters,
    config: RolloutConfig,
    *,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str = "cpu",
) -> RolloutResult:
    """Roll out one follower behind an exogenous leader.

    Public outputs are time-major. Acceleration has shape ``[T]`` and all other
    arrays have shape ``[T + 1]``.
    """

    params_t = parameters_to_tensors(params, dtype=dtype, device=device)
    leader_x = leader.position.to(dtype=dtype, device=device)
    leader_v = leader.speed.to(dtype=dtype, device=device)
    dt = torch.tensor(config.dt, dtype=dtype, device=device)
    leader_length = torch.tensor(config.leader_length, dtype=dtype, device=device)
    steps = leader_x.shape[0] - 1

    follower_x = [torch.tensor(config.initial_follower.position, dtype=dtype, device=device)]
    follower_v = [torch.tensor(config.initial_follower.speed, dtype=dtype, device=device)]
    follower_a: list[torch.Tensor] = []

    for t in range(steps):
        x_t = follower_x[-1]
        v_t = follower_v[-1]
        gap_t = leader_x[t] - x_t - leader_length
        acc_t = _acceleration(
            mode=config.acceleration_mode,
            v_follower=v_t,
            v_leader=leader_v[t],
            gap=gap_t,
            params=params_t,
            dt=dt,
            prevent_negative_speed=config.prevent_negative_speed,
        )
        v_next = v_t + dt * acc_t
        x_next = x_t + dt * v_next
        follower_a.append(acc_t)
        follower_v.append(v_next)
        follower_x.append(x_next)

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


def _acceleration(
    *,
    mode: AccelerationMode,
    v_follower: torch.Tensor,
    v_leader: torch.Tensor,
    gap: torch.Tensor,
    params: IDMParameters,
    dt: torch.Tensor,
    prevent_negative_speed: bool,
) -> torch.Tensor:
    if mode == "diffidm":
        return diffidm_acceleration(
            v_follower=v_follower,
            v_leader=v_leader,
            gap=gap,
            params=params,
            dt=dt,
            prevent_negative_speed=prevent_negative_speed,
        )
    if mode == "smooth_reference":
        return smooth_clamped_idm_acceleration_reference(
            v_follower=v_follower,
            v_leader=v_leader,
            gap=gap,
            params=params,
            dt=dt,
            prevent_negative_speed=prevent_negative_speed,
        )
    if mode == "textbook":
        return textbook_idm_acceleration(
            v_follower=v_follower,
            v_leader=v_leader,
            gap=gap,
            params=params,
        )
    raise ValueError(f"unknown acceleration mode: {mode}")
