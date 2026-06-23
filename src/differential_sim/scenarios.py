"""Deterministic exogenous leader profiles."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch


LeaderProfileKind = Literal[
    "constant",
    "braking_recovery",
    "sinusoidal",
    "random_braking_cycles",
    "multi_pulse_braking",
    "chirp_sinusoidal",
    "mixed_regime",
]


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
    seed: int = 0
    initial_speed_range: tuple[float, float] = (16.0, 20.0)
    upper_target_speed_range: tuple[float, float] = (18.0, 24.0)
    post_brake_speed_range: tuple[float, float] = (10.0, 18.0)
    minimum_speed_drop: float = 3.0
    acceleration_magnitude_range: tuple[float, float] = (0.4, 1.2)
    braking_magnitude_range: tuple[float, float] = (0.8, 2.2)
    hold_duration_range: tuple[float, float] = (1.0, 3.0)
    post_brake_hold_duration_range: tuple[float, float] = (0.6, 1.6)
    speed_floor: float = 6.0
    speed_ceiling: float = 26.0
    pulse_starts: tuple[float, ...] = ()
    pulse_brake_delta_v: tuple[float, ...] = ()
    pulse_brake_durations: tuple[float, ...] = ()
    pulse_recovery_durations: tuple[float, ...] = ()
    period_start: float = 10.0
    period_end: float = 4.0
    base_speed: float = 18.0
    first_brake_delta_v: float = 2.5
    first_brake_duration: float = 1.4
    first_recovery_duration: float = 1.6
    second_brake_delta_v: float = 4.0
    second_brake_duration: float = 1.2
    second_recovery_duration: float = 2.0


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
    elif config.kind == "random_braking_cycles":
        speed = _random_braking_cycles_speed(config, dtype=dtype, device=device)
    elif config.kind == "multi_pulse_braking":
        speed = _multi_pulse_braking_speed(time, config)
    elif config.kind == "chirp_sinusoidal":
        speed = _chirp_sinusoidal_speed(time, config)
    elif config.kind == "mixed_regime":
        speed = _mixed_regime_speed(time, config)
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


def random_braking_cycle_events(config: ScenarioConfig) -> list[dict[str, float]]:
    """Return deterministic sampled cycle parameters for reporting."""

    if config.kind != "random_braking_cycles":
        return []
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(config.seed))
    current_speed = _uniform(config.initial_speed_range, generator)
    elapsed = 0.0
    events: list[dict[str, float]] = []
    while elapsed < config.steps * config.dt:
        upper = _uniform(config.upper_target_speed_range, generator)
        accel = _uniform(config.acceleration_magnitude_range, generator)
        hold = _uniform(config.hold_duration_range, generator)
        lower_max = min(config.post_brake_speed_range[1], upper - config.minimum_speed_drop)
        lower_min = min(config.post_brake_speed_range[0], lower_max)
        lower = _uniform((lower_min, lower_max), generator)
        brake = _uniform(config.braking_magnitude_range, generator)
        post_hold = _uniform(config.post_brake_hold_duration_range, generator)
        accel_duration = abs(upper - current_speed) / max(accel, 1e-12)
        brake_duration = abs(upper - lower) / max(brake, 1e-12)
        events.append(
            {
                "start_time": elapsed,
                "start_speed": current_speed,
                "upper_target_speed": upper,
                "acceleration_magnitude": accel,
                "acceleration_duration": accel_duration,
                "hold_duration": hold,
                "post_brake_speed": lower,
                "braking_magnitude": brake,
                "braking_duration": brake_duration,
                "post_brake_hold_duration": post_hold,
            }
        )
        elapsed += accel_duration + hold + brake_duration + post_hold
        current_speed = lower
    return events


def _random_braking_cycles_speed(
    config: ScenarioConfig,
    *,
    dtype: torch.dtype,
    device: torch.device | str,
) -> torch.Tensor:
    dt = config.dt
    speed_values: list[float] = []
    current_speed = random_braking_cycle_events(config)[0]["start_speed"]
    for event in random_braking_cycle_events(config):
        current_speed = _append_linear_segment(
            speed_values,
            current_speed,
            event["upper_target_speed"],
            event["acceleration_magnitude"],
            dt,
            config.steps + 1,
        )
        current_speed = _append_hold_segment(
            speed_values,
            current_speed,
            event["hold_duration"],
            dt,
            config.steps + 1,
        )
        current_speed = _append_linear_segment(
            speed_values,
            current_speed,
            event["post_brake_speed"],
            event["braking_magnitude"],
            dt,
            config.steps + 1,
        )
        current_speed = _append_hold_segment(
            speed_values,
            current_speed,
            event["post_brake_hold_duration"],
            dt,
            config.steps + 1,
        )
        if len(speed_values) >= config.steps + 1:
            break
    while len(speed_values) < config.steps + 1:
        speed_values.append(current_speed)
    speed = torch.tensor(speed_values[: config.steps + 1], dtype=dtype, device=device)
    return torch.clamp(speed, min=config.speed_floor, max=config.speed_ceiling)


def _multi_pulse_braking_speed(time: torch.Tensor, config: ScenarioConfig) -> torch.Tensor:
    speed = torch.full_like(time, config.initial_speed)
    starts = config.pulse_starts
    deltas = config.pulse_brake_delta_v
    brake_durations = config.pulse_brake_durations
    recovery_durations = config.pulse_recovery_durations
    if not (len(starts) == len(deltas) == len(brake_durations) == len(recovery_durations)):
        raise ValueError("multi_pulse_braking pulse tuples must have equal length")
    for start, delta, brake_duration, recovery_duration in zip(
        starts, deltas, brake_durations, recovery_durations, strict=True
    ):
        pulse = _braking_recovery_delta(
            time,
            start=float(start),
            brake_duration=float(brake_duration),
            recovery_duration=float(recovery_duration),
            brake_delta_v=float(delta),
        )
        speed = speed - pulse
    return torch.clamp(speed, min=config.speed_floor, max=config.speed_ceiling)


def _chirp_sinusoidal_speed(time: torch.Tensor, config: ScenarioConfig) -> torch.Tensor:
    duration = torch.clamp(time[-1], min=torch.tensor(config.dt, dtype=time.dtype, device=time.device))
    f0 = torch.tensor(1.0 / config.period_start, dtype=time.dtype, device=time.device)
    f1 = torch.tensor(1.0 / config.period_end, dtype=time.dtype, device=time.device)
    phase = 2.0 * torch.pi * (f0 * time + 0.5 * (f1 - f0) * torch.square(time) / duration)
    speed = config.base_speed + config.sinusoid_amplitude * torch.sin(phase)
    return torch.clamp(speed, min=config.speed_floor, max=config.speed_ceiling)


def _mixed_regime_speed(time: torch.Tensor, config: ScenarioConfig) -> torch.Tensor:
    speed = torch.full_like(time, config.base_speed)
    first = _braking_recovery_delta(
        time,
        start=3.0,
        brake_duration=config.first_brake_duration,
        recovery_duration=config.first_recovery_duration,
        brake_delta_v=config.first_brake_delta_v,
    )
    second = _braking_recovery_delta(
        time,
        start=11.0,
        brake_duration=config.second_brake_duration,
        recovery_duration=config.second_recovery_duration,
        brake_delta_v=config.second_brake_delta_v,
    )
    sinusoid_mask = (time >= 7.0) & (time < 11.0)
    if torch.any(sinusoid_mask):
        local_t = time[sinusoid_mask] - 7.0
        speed[sinusoid_mask] = speed[sinusoid_mask] + config.sinusoid_amplitude * torch.sin(
            2.0 * torch.pi * local_t / config.sinusoid_period
        )
    speed = speed - first - second
    return torch.clamp(speed, min=config.speed_floor, max=config.speed_ceiling)


def _braking_recovery_delta(
    time: torch.Tensor,
    *,
    start: float,
    brake_duration: float,
    recovery_duration: float,
    brake_delta_v: float,
) -> torch.Tensor:
    delta = torch.zeros_like(time)
    brake_end = start + brake_duration
    recovery_end = brake_end + recovery_duration
    braking = (time >= start) & (time < brake_end)
    if torch.any(braking):
        frac = (time[braking] - start) / brake_duration
        delta[braking] = brake_delta_v * frac
    recovering = (time >= brake_end) & (time < recovery_end)
    if torch.any(recovering):
        frac = (time[recovering] - brake_end) / recovery_duration
        delta[recovering] = brake_delta_v * (1.0 - frac)
    return delta


def _uniform(bounds: tuple[float, float], generator: torch.Generator) -> float:
    low, high = bounds
    if high < low:
        raise ValueError(f"invalid bounds: {bounds}")
    if high == low:
        return float(low)
    return float(low + (high - low) * torch.rand((), generator=generator).item())


def _append_linear_segment(
    values: list[float],
    current_speed: float,
    target_speed: float,
    magnitude: float,
    dt: float,
    max_len: int,
) -> float:
    if len(values) == 0:
        values.append(current_speed)
    if current_speed == target_speed:
        return current_speed
    sign = 1.0 if target_speed > current_speed else -1.0
    while len(values) < max_len and sign * (target_speed - current_speed) > 1e-12:
        step = sign * min(abs(target_speed - current_speed), magnitude * dt)
        current_speed += step
        values.append(current_speed)
    return current_speed


def _append_hold_segment(
    values: list[float],
    current_speed: float,
    duration: float,
    dt: float,
    max_len: int,
) -> float:
    if len(values) == 0:
        values.append(current_speed)
    count = max(0, int(round(duration / dt)))
    for _ in range(count):
        if len(values) >= max_len:
            break
        values.append(current_speed)
    return current_speed
