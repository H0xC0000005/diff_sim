"""Deterministic exogenous leader profiles."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch


LeaderProfileKind = Literal["constant", "braking_recovery", "sinusoidal"]


@dataclass(frozen=True)
class ScenarioConfig:
    kind: LeaderProfileKind = "braking_recovery"
    steps: int = 80
    dt: float = 0.2
    initial_position: float = 0.0
    initial_speed: float = 18.0
    brake_start: float = 4.0
    brake_duration: float = 2.0
    recovery_duration: float = 3.0
    brake_delta_v: float = 5.0
    sinusoid_amplitude: float = 2.0
    sinusoid_period: float = 8.0


@dataclass(frozen=True)
class LeaderProfile:
    time: torch.Tensor
    position: torch.Tensor
    speed: torch.Tensor


def leader_profile(
    config: ScenarioConfig,
    *,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str = "cpu",
) -> LeaderProfile:
    """Generate deterministic leader position and speed arrays of shape ``[T + 1]``."""

    dt = torch.tensor(config.dt, dtype=dtype, device=device)
    time = torch.arange(config.steps + 1, dtype=dtype, device=device) * dt

    if config.kind == "constant":
        speed = torch.full_like(time, config.initial_speed)
    elif config.kind == "braking_recovery":
        speed = _braking_recovery_speed(time, config)
    elif config.kind == "sinusoidal":
        speed = config.initial_speed + config.sinusoid_amplitude * torch.sin(
            2.0 * torch.pi * time / config.sinusoid_period
        )
    else:
        raise ValueError(f"unknown leader profile kind: {config.kind}")

    position = torch.empty_like(speed)
    position[0] = config.initial_position
    position[1:] = config.initial_position + torch.cumsum(speed[1:] * dt, dim=0)
    return LeaderProfile(time=time, position=position, speed=speed)


def _braking_recovery_speed(time: torch.Tensor, config: ScenarioConfig) -> torch.Tensor:
    speed = torch.full_like(time, config.initial_speed)
    brake_start = config.brake_start
    brake_end = config.brake_start + config.brake_duration
    recovery_end = brake_end + config.recovery_duration
    low_speed = config.initial_speed - config.brake_delta_v

    braking = (time >= brake_start) & (time < brake_end)
    if torch.any(braking):
        frac = (time[braking] - brake_start) / config.brake_duration
        speed[braking] = config.initial_speed - config.brake_delta_v * frac

    recovering = (time >= brake_end) & (time < recovery_end)
    if torch.any(recovering):
        frac = (time[recovering] - brake_end) / config.recovery_duration
        speed[recovering] = low_speed + config.brake_delta_v * frac

    speed[time >= recovery_end] = config.initial_speed
    return speed
