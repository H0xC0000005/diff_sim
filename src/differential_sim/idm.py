"""IDM equations and the approved Milestone 0 diffidm wrapper."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Mapping, Sequence

import torch
from diffidm import IDMLayer


IDM_DELTA = 4.0


@dataclass(frozen=True)
class IDMParameters:
    """Physical IDM parameters.

    Units: acceleration m/s^2, speed m/s, gap m, headway s.
    """

    a_max: float | torch.Tensor
    b_comfort: float | torch.Tensor
    v0: float | torch.Tensor
    s0: float | torch.Tensor
    time_headway: float | torch.Tensor
    a_min: float | torch.Tensor = -8.0

    def with_updates(self, updates: Mapping[str, torch.Tensor]) -> "IDMParameters":
        values = {name: getattr(self, name) for name in self.__dataclass_fields__}
        values.update(updates)
        return replace(self, **values)


@dataclass(frozen=True)
class ParameterBounds:
    lower: float
    upper: float


DEFAULT_FIT_BOUNDS: dict[str, ParameterBounds] = {
    "time_headway": ParameterBounds(0.5, 3.0),
    "v0": ParameterBounds(10.0, 40.0),
    "s0": ParameterBounds(0.5, 10.0),
}


def as_tensor(
    value: float | torch.Tensor,
    *,
    dtype: torch.dtype,
    device: torch.device | str,
) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.to(dtype=dtype, device=device)
    return torch.tensor(value, dtype=dtype, device=device)


def parameters_to_tensors(
    params: IDMParameters,
    *,
    dtype: torch.dtype,
    device: torch.device | str,
) -> IDMParameters:
    return IDMParameters(
        a_max=as_tensor(params.a_max, dtype=dtype, device=device),
        b_comfort=as_tensor(params.b_comfort, dtype=dtype, device=device),
        v0=as_tensor(params.v0, dtype=dtype, device=device),
        s0=as_tensor(params.s0, dtype=dtype, device=device),
        time_headway=as_tensor(params.time_headway, dtype=dtype, device=device),
        a_min=as_tensor(params.a_min, dtype=dtype, device=device),
    )


def idm_optimal_spacing(
    *,
    v_follower: torch.Tensor,
    delta_v: torch.Tensor,
    params: IDMParameters,
) -> torch.Tensor:
    """Textbook desired dynamic spacing, before any diffidm smooth bounds."""

    return (
        params.s0
        + v_follower * params.time_headway
        + v_follower * delta_v / (2.0 * torch.sqrt(params.a_max * params.b_comfort))
    )


def idm_acceleration_from_spacing(
    *,
    v_follower: torch.Tensor,
    gap: torch.Tensor,
    optimal_spacing: torch.Tensor,
    params: IDMParameters,
) -> torch.Tensor:
    """Textbook IDM acceleration from a supplied optimal spacing."""

    return params.a_max * (
        1.0
        - torch.pow(v_follower / params.v0, IDM_DELTA)
        - torch.pow(optimal_spacing / gap, 2.0)
    )


def textbook_idm_acceleration(
    *,
    v_follower: torch.Tensor,
    v_leader: torch.Tensor,
    gap: torch.Tensor,
    params: IDMParameters,
) -> torch.Tensor:
    """Unclamped textbook IDM acceleration.

    ``delta_v`` follows the project convention ``v_follower - v_leader``.
    """

    delta_v = v_follower - v_leader
    optimal_spacing = idm_optimal_spacing(
        v_follower=v_follower,
        delta_v=delta_v,
        params=params,
    )
    return idm_acceleration_from_spacing(
        v_follower=v_follower,
        gap=gap,
        optimal_spacing=optimal_spacing,
        params=params,
    )


def smooth_min_clamp(x: torch.Tensor, min_value: torch.Tensor | float) -> torch.Tensor:
    """Replicate ``diffidm.IDMLayer.soft_min_clamp``."""

    return min_value + torch.nn.functional.softplus(x - min_value)


def smooth_clamped_idm_acceleration_reference(
    *,
    v_follower: torch.Tensor,
    v_leader: torch.Tensor,
    gap: torch.Tensor,
    params: IDMParameters,
    dt: torch.Tensor,
    prevent_negative_speed: bool = True,
) -> torch.Tensor:
    """Reference implementation of the approved diffidm smooth-clamped step."""

    delta_v = v_follower - v_leader
    optimal_spacing = idm_optimal_spacing(
        v_follower=v_follower,
        delta_v=delta_v,
        params=params,
    )
    optimal_spacing = smooth_min_clamp(optimal_spacing, 0.0)
    acc = idm_acceleration_from_spacing(
        v_follower=v_follower,
        gap=gap,
        optimal_spacing=optimal_spacing,
        params=params,
    )
    if prevent_negative_speed:
        acc_lb = torch.maximum(-v_follower / dt, params.a_min)
    else:
        acc_lb = params.a_min
    return smooth_min_clamp(acc, acc_lb)


def diffidm_acceleration(
    *,
    v_follower: torch.Tensor,
    v_leader: torch.Tensor,
    gap: torch.Tensor,
    params: IDMParameters,
    dt: torch.Tensor,
    prevent_negative_speed: bool = True,
) -> torch.Tensor:
    """Call ``diffidm.IDMLayer.apply`` with project naming and units."""

    delta_v = v_follower - v_leader
    return IDMLayer.apply(
        params.a_max,
        params.a_min,
        params.b_comfort,
        v_follower,
        params.v0,
        gap,
        delta_v,
        params.s0,
        params.time_headway,
        dt,
        prevent_negative_speed,
    )


def values_to_unconstrained(
    values: Mapping[str, float],
    bounds: Mapping[str, ParameterBounds] = DEFAULT_FIT_BOUNDS,
    *,
    dtype: torch.dtype,
    device: torch.device | str,
) -> dict[str, torch.Tensor]:
    """Map physical values inside bounds to unconstrained logits."""

    raw: dict[str, torch.Tensor] = {}
    eps = torch.finfo(dtype).eps
    for name, value in values.items():
        bound = bounds[name]
        y = (float(value) - bound.lower) / (bound.upper - bound.lower)
        y_tensor = torch.tensor(y, dtype=dtype, device=device).clamp(eps, 1.0 - eps)
        raw[name] = torch.logit(y_tensor)
    return raw


def unconstrained_to_values(
    raw_values: Mapping[str, torch.Tensor],
    bounds: Mapping[str, ParameterBounds] = DEFAULT_FIT_BOUNDS,
) -> dict[str, torch.Tensor]:
    """Map unconstrained fit variables into documented physical ranges."""

    values: dict[str, torch.Tensor] = {}
    for name, raw in raw_values.items():
        bound = bounds[name]
        values[name] = bound.lower + (bound.upper - bound.lower) * torch.sigmoid(raw)
    return values


def build_parameter_tensors(
    base: IDMParameters,
    fitted_raw: Mapping[str, torch.Tensor],
    fitted_names: Sequence[str],
    *,
    dtype: torch.dtype,
    device: torch.device | str,
    bounds: Mapping[str, ParameterBounds] = DEFAULT_FIT_BOUNDS,
) -> IDMParameters:
    base_tensors = parameters_to_tensors(base, dtype=dtype, device=device)
    bounded = unconstrained_to_values(fitted_raw, bounds)
    updates = {name: bounded[name] for name in fitted_names}
    return base_tensors.with_updates(updates)
