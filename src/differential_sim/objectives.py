"""Milestone 1 rollout objective and component reporting."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import torch

from differential_sim.idm import IDMParameters, smooth_min_clamp
from differential_sim.rollout import RolloutResult


@dataclass(frozen=True)
class ObjectiveComponents:
    total: torch.Tensor
    progress: torch.Tensor
    safety: torch.Tensor
    jerk: torch.Tensor
    weighted_progress: torch.Tensor
    weighted_safety: torch.Tensor
    weighted_jerk: torch.Tensor


@dataclass(frozen=True)
class ObjectiveConfig:
    progress_weight: float = 1.0
    safety_weight: float = 0.7
    jerk_weight: float = 5.0
    progress_scale: float = 5.0
    safety_scale: float = 5.0
    jerk_scale: float = 2.0
    safety_s0: float = 2.0
    safety_time_headway: float = 1.0
    safety_a_max: float = 1.4
    safety_b_comfort: float = 2.0


def rollout_objective(
    result: RolloutResult,
    config: ObjectiveConfig = ObjectiveConfig(),
) -> ObjectiveComponents:
    """Compute weighted total objective from one rollout."""

    dtype = result.follower_v.dtype
    device = result.follower_v.device
    progress_scale = torch.tensor(config.progress_scale, dtype=dtype, device=device)
    safety_scale = torch.tensor(config.safety_scale, dtype=dtype, device=device)
    jerk_scale = torch.tensor(config.jerk_scale, dtype=dtype, device=device)

    v = result.follower_v[1:]
    leader_v = result.leader_v[1:]
    gap = result.gap[1:]
    delta_v = result.delta_v[1:]

    progress = torch.mean(torch.square((v - leader_v) / progress_scale))
    s_safe = safety_spacing(v=v, delta_v=delta_v, dtype=dtype, device=device, config=config)
    safety = torch.mean(torch.square(torch.nn.functional.softplus((s_safe - gap) / safety_scale)))
    if result.follower_a.shape[0] < 2:
        jerk = torch.zeros((), dtype=dtype, device=device)
    else:
        jerk = torch.mean(torch.square((result.follower_a[1:] - result.follower_a[:-1]) / jerk_scale))
    weighted_progress = torch.tensor(config.progress_weight, dtype=dtype, device=device) * progress
    weighted_safety = torch.tensor(config.safety_weight, dtype=dtype, device=device) * safety
    weighted_jerk = torch.tensor(config.jerk_weight, dtype=dtype, device=device) * jerk
    total = weighted_progress + weighted_safety + weighted_jerk
    return ObjectiveComponents(
        total=total,
        progress=progress,
        safety=safety,
        jerk=jerk,
        weighted_progress=weighted_progress,
        weighted_safety=weighted_safety,
        weighted_jerk=weighted_jerk,
    )


def safety_spacing(
    *,
    v: torch.Tensor,
    delta_v: torch.Tensor,
    dtype: torch.dtype,
    device: torch.device,
    config: ObjectiveConfig = ObjectiveConfig(),
) -> torch.Tensor:
    s0 = torch.tensor(config.safety_s0, dtype=dtype, device=device)
    t_safe = torch.tensor(config.safety_time_headway, dtype=dtype, device=device)
    a_max = torch.tensor(config.safety_a_max, dtype=dtype, device=device)
    b_comfort = torch.tensor(config.safety_b_comfort, dtype=dtype, device=device)
    spacing = s0 + v * t_safe + v * delta_v / (2.0 * torch.sqrt(a_max * b_comfort))
    return smooth_min_clamp(spacing, 0.0)


def mean_scenario_objective(
    results: Sequence[RolloutResult],
    config: ObjectiveConfig = ObjectiveConfig(),
) -> ObjectiveComponents:
    components = [rollout_objective(result, config) for result in results]
    return ObjectiveComponents(
        total=torch.stack([c.total for c in components]).mean(),
        progress=torch.stack([c.progress for c in components]).mean(),
        safety=torch.stack([c.safety for c in components]).mean(),
        jerk=torch.stack([c.jerk for c in components]).mean(),
        weighted_progress=torch.stack([c.weighted_progress for c in components]).mean(),
        weighted_safety=torch.stack([c.weighted_safety for c in components]).mean(),
        weighted_jerk=torch.stack([c.weighted_jerk for c in components]).mean(),
    )


def component_floats(components: ObjectiveComponents) -> dict[str, float]:
    return {
        "total": float(components.total.detach().cpu().item()),
        "progress": float(components.progress.detach().cpu().item()),
        "safety": float(components.safety.detach().cpu().item()),
        "jerk": float(components.jerk.detach().cpu().item()),
    }


def weighted_component_floats(components: ObjectiveComponents) -> dict[str, float]:
    return {
        "progress": float(components.weighted_progress.detach().cpu().item()),
        "safety": float(components.weighted_safety.detach().cpu().item()),
        "jerk": float(components.weighted_jerk.detach().cpu().item()),
    }


def semantic_inverse_mean_weights(component_rows: Sequence[Mapping[str, float]]) -> dict[str, float]:
    """Diagnostic-only inverse mean magnitude weights for unweighted components."""

    weights: dict[str, float] = {}
    for name in ("progress", "safety", "jerk"):
        values = torch.tensor([float(row[name]) for row in component_rows], dtype=torch.float64)
        mean = float(values.mean().item())
        weights[name] = float("inf") if mean == 0.0 else 1.0 / mean
    return weights


def default_idm_params(time_headway: float = 1.4) -> IDMParameters:
    return IDMParameters(
        a_max=1.4,
        b_comfort=2.0,
        v0=28.0,
        s0=2.0,
        time_headway=time_headway,
        a_min=-8.0,
    )
