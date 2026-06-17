"""Synthetic trajectory generation and Milestone 0 fitting helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import torch

from differential_sim.idm import (
    DEFAULT_FIT_BOUNDS,
    IDMParameters,
    ParameterBounds,
    build_parameter_tensors,
    values_to_unconstrained,
)
from differential_sim.rollout import RolloutConfig, RolloutResult, rollout_follower
from differential_sim.scenarios import LeaderProfile


@dataclass(frozen=True)
class LossScales:
    x_scale: float = 10.0
    v_scale: float = 5.0


@dataclass(frozen=True)
class FitConfig:
    fitted_names: tuple[str, ...] = ("time_headway", "v0")
    steps: int = 300
    learning_rate: float = 0.03
    seed: int = 0
    loss_scales: LossScales = LossScales()


@dataclass(frozen=True)
class FitResult:
    initial_loss: float
    final_loss: float
    initial_parameters: dict[str, float]
    final_parameters: dict[str, float]
    truth_parameters: dict[str, float]
    loss_history: list[float]
    gradients_finite: bool


def generate_synthetic_trajectory(
    leader: LeaderProfile,
    params: IDMParameters,
    rollout_config: RolloutConfig,
    *,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str = "cpu",
) -> RolloutResult:
    """Generate the reference trajectory with the approved rollout code."""

    return rollout_follower(
        leader,
        params,
        rollout_config,
        dtype=dtype,
        device=device,
    )


def trajectory_loss(
    pred: RolloutResult,
    ref: RolloutResult,
    scales: LossScales = LossScales(),
) -> torch.Tensor:
    x_scale = torch.as_tensor(scales.x_scale, dtype=pred.follower_x.dtype, device=pred.follower_x.device)
    v_scale = torch.as_tensor(scales.v_scale, dtype=pred.follower_v.dtype, device=pred.follower_v.device)
    x_loss = torch.mean(torch.square((pred.follower_x - ref.follower_x) / x_scale))
    v_loss = torch.mean(torch.square((pred.follower_v - ref.follower_v) / v_scale))
    return x_loss + v_loss


def fit_synthetic_parameters(
    *,
    leader: LeaderProfile,
    reference: RolloutResult,
    base_params: IDMParameters,
    truth_values: Mapping[str, float],
    initial_values: Mapping[str, float],
    rollout_config: RolloutConfig,
    fit_config: FitConfig = FitConfig(),
    bounds: Mapping[str, ParameterBounds] = DEFAULT_FIT_BOUNDS,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str = "cpu",
) -> FitResult:
    """Fit a configurable IDM parameter subset through full autograd rollout."""

    torch.manual_seed(fit_config.seed)
    fitted_names = fit_config.fitted_names
    raw_init = values_to_unconstrained(
        {name: initial_values[name] for name in fitted_names},
        bounds,
        dtype=dtype,
        device=device,
    )
    raw_params = {
        name: value.detach().clone().requires_grad_(True) for name, value in raw_init.items()
    }
    optimizer = torch.optim.Adam(list(raw_params.values()), lr=fit_config.learning_rate)
    loss_history: list[float] = []
    gradients_finite = True

    with torch.no_grad():
        initial_params = build_parameter_tensors(
            base_params,
            raw_params,
            fitted_names,
            dtype=dtype,
            device=device,
            bounds=bounds,
        )
        initial_rollout = rollout_follower(
            leader,
            initial_params,
            rollout_config,
            dtype=dtype,
            device=device,
        )
        initial_loss = float(trajectory_loss(initial_rollout, reference, fit_config.loss_scales).item())

    for _ in range(fit_config.steps):
        optimizer.zero_grad(set_to_none=True)
        current_params = build_parameter_tensors(
            base_params,
            raw_params,
            fitted_names,
            dtype=dtype,
            device=device,
            bounds=bounds,
        )
        pred = rollout_follower(
            leader,
            current_params,
            rollout_config,
            dtype=dtype,
            device=device,
        )
        loss = trajectory_loss(pred, reference, fit_config.loss_scales)
        loss.backward()
        gradients_finite = gradients_finite and all(
            raw.grad is not None and bool(torch.isfinite(raw.grad).all().item())
            for raw in raw_params.values()
        )
        optimizer.step()
        loss_history.append(float(loss.detach().item()))

    with torch.no_grad():
        final_params = build_parameter_tensors(
            base_params,
            raw_params,
            fitted_names,
            dtype=dtype,
            device=device,
            bounds=bounds,
        )
        final_rollout = rollout_follower(
            leader,
            final_params,
            rollout_config,
            dtype=dtype,
            device=device,
        )
        final_loss = float(trajectory_loss(final_rollout, reference, fit_config.loss_scales).item())
        final_values = {
            name: float(getattr(final_params, name).detach().cpu().item()) for name in fitted_names
        }

    return FitResult(
        initial_loss=initial_loss,
        final_loss=final_loss,
        initial_parameters={name: float(initial_values[name]) for name in fitted_names},
        final_parameters=final_values,
        truth_parameters={name: float(truth_values[name]) for name in fitted_names},
        loss_history=loss_history,
        gradients_finite=gradients_finite,
    )


def normalized_parameter_distance(
    values: Mapping[str, float],
    truth: Mapping[str, float],
    fitted_names: Sequence[str],
    bounds: Mapping[str, ParameterBounds] = DEFAULT_FIT_BOUNDS,
) -> float:
    terms = []
    for name in fitted_names:
        span = bounds[name].upper - bounds[name].lower
        terms.append(((values[name] - truth[name]) / span) ** 2)
    return float(sum(terms) ** 0.5)


def central_finite_difference_directional_derivative(
    loss_fn,
    raw_values: Mapping[str, torch.Tensor],
    direction: Mapping[str, torch.Tensor],
    epsilon: float,
) -> torch.Tensor:
    """Central finite-difference directional derivative for raw fit variables."""

    plus = {name: value + epsilon * direction[name] for name, value in raw_values.items()}
    minus = {name: value - epsilon * direction[name] for name, value in raw_values.items()}
    return (loss_fn(plus) - loss_fn(minus)) / (2.0 * epsilon)


def autograd_directional_derivative(
    loss: torch.Tensor,
    raw_values: Mapping[str, torch.Tensor],
    direction: Mapping[str, torch.Tensor],
) -> torch.Tensor:
    grads = torch.autograd.grad(loss, tuple(raw_values.values()), create_graph=False)
    total = torch.zeros((), dtype=loss.dtype, device=loss.device)
    for grad, name in zip(grads, raw_values.keys(), strict=True):
        total = total + grad * direction[name]
    return total
