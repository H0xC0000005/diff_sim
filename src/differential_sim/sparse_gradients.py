"""SG1 sparse checkpoint-gradient utilities."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from differential_sim.batched_temporal_gradients import (
    BatchedScenarioData,
    batched_rollout_objective,
    rollout_batched_scenarios_with_controller,
    split_batched_rollout,
)
from differential_sim.controllers import HeadwayController
from differential_sim.idm import IDMParameters, diffidm_acceleration, parameters_to_tensors
from differential_sim.objectives import ObjectiveComponents, ObjectiveConfig, safety_spacing
from differential_sim.rollout import RolloutConfig, RolloutResult


SPARSE_STRIDES = (2, 4, 6, 8)
DENSE_REFERENCE_HORIZONS = (80, 50, 10)


@dataclass(frozen=True)
class SparseSpan:
    start: int
    end: int

    @property
    def length(self) -> int:
        return self.end - self.start


@dataclass(frozen=True)
class SparseObjectiveResult:
    sparse: ObjectiveComponents
    exact: ObjectiveComponents
    per_scenario_exact: ObjectiveComponents
    rollout: RolloutResult
    spans: tuple[SparseSpan, ...]
    remainder_start: int | None
    scenario_names: tuple[str, ...]


def complete_spans(steps: int, stride: int) -> tuple[SparseSpan, ...]:
    """Return complete SG1 checkpoint spans for ``steps`` and ``stride``."""

    if stride < 1:
        raise ValueError("stride must be positive")
    return tuple(SparseSpan(start, start + stride) for start in range(0, steps - stride + 1, stride))


def sparse_b1_objective(
    scenarios: BatchedScenarioData,
    *,
    controller: HeadwayController,
    controller_parameters: torch.Tensor,
    base_params: IDMParameters,
    rollout_config: RolloutConfig,
    objective_config: ObjectiveConfig,
    stride: int,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str = "cpu",
) -> SparseObjectiveResult:
    """Build the approved SG1 B1 anchored sparse objective.

    The exact full-resolution rollout is evaluated under ``torch.no_grad()`` and
    used only as stop-gradient anchor values. Gradients flow through the
    checkpoint-level B1 macro-step surrogate.
    """

    if stride not in SPARSE_STRIDES:
        raise ValueError(f"SG1 stride must be one of {SPARSE_STRIDES}")
    leader_x = scenarios.leader_x.to(dtype=dtype, device=device)
    leader_v = scenarios.leader_v.to(dtype=dtype, device=device)
    scenario_count, time_count = leader_x.shape
    steps = time_count - 1
    if steps != 80:
        raise ValueError(f"SG1 requires T=80, got {steps}")

    with torch.no_grad():
        exact_rollout = rollout_batched_scenarios_with_controller(
            scenarios,
            beta=controller_parameters.detach(),
            controller=controller,
            base_params=base_params,
            rollout_config=rollout_config,
            horizon_k=steps,
            dtype=dtype,
            device=device,
        )
        exact_aggregate, exact_per_scenario = batched_rollout_objective(
            exact_rollout,
            objective_config,
        )

    spans = complete_spans(steps, stride)
    if not spans:
        raise ValueError("stride produced no complete spans")
    remainder_start = spans[-1].end if spans[-1].end < steps else None

    params_base = parameters_to_tensors(base_params, dtype=dtype, device=device)
    dt = torch.tensor(rollout_config.dt, dtype=dtype, device=device)
    leader_length = torch.tensor(rollout_config.leader_length, dtype=dtype, device=device)
    z_sparse = torch.stack(
        [exact_rollout.follower_x[:, 0], exact_rollout.follower_v[:, 0]],
        dim=-1,
    ).detach()
    total_sparse = _zero_components(dtype=dtype, device=device)

    for span in spans:
        z_start_star = torch.stack(
            [
                exact_rollout.follower_x[:, span.start],
                exact_rollout.follower_v[:, span.start],
            ],
            dim=-1,
        ).detach()
        z_end_star = torch.stack(
            [
                exact_rollout.follower_x[:, span.end],
                exact_rollout.follower_v[:, span.end],
            ],
            dim=-1,
        ).detach()
        exact_span = _exact_span_contribution(
            exact_rollout,
            span=span,
            objective_config=objective_config,
        )
        macro = _macro_step_components(
            z_sparse,
            leader_x=leader_x,
            leader_v=leader_v,
            leader_length=leader_length,
            span=span,
            controller=controller,
            controller_parameters=controller_parameters,
            params_base=params_base,
            rollout_config=rollout_config,
            objective_config=objective_config,
            dt=dt,
            scenario_count=scenario_count,
            steps=steps,
            previous_accel=(
                None
                if span.start == 0
                else exact_rollout.follower_a[:, span.start - 1].detach()
            ),
        )
        macro_anchor = _macro_step_components(
            z_start_star,
            leader_x=leader_x,
            leader_v=leader_v,
            leader_length=leader_length,
            span=span,
            controller=controller,
            controller_parameters=controller_parameters,
            params_base=params_base,
            rollout_config=rollout_config,
            objective_config=objective_config,
            dt=dt,
            scenario_count=scenario_count,
            steps=steps,
            previous_accel=(
                None
                if span.start == 0
                else exact_rollout.follower_a[:, span.start - 1].detach()
            ),
        )
        z_sparse = z_end_star + macro.z_end - macro_anchor.z_end.detach()
        sparse_span = _components_add(
            _components_detach(exact_span.components),
            _components_sub(macro.components, _components_detach(macro_anchor.components)),
        )
        total_sparse = _components_add(total_sparse, sparse_span)

    if remainder_start is not None:
        remainder = _exact_span_contribution(
            exact_rollout,
            span=SparseSpan(remainder_start, steps),
            objective_config=objective_config,
        )
        total_sparse = _components_add(total_sparse, _components_detach(remainder.components))

    return SparseObjectiveResult(
        sparse=total_sparse,
        exact=exact_aggregate,
        per_scenario_exact=exact_per_scenario,
        rollout=exact_rollout,
        spans=spans,
        remainder_start=remainder_start,
        scenario_names=scenarios.scenario_names,
    )


def dense_equivalent_vjp_objective(
    scenarios: BatchedScenarioData,
    *,
    controller: HeadwayController,
    controller_parameters: torch.Tensor,
    base_params: IDMParameters,
    rollout_config: RolloutConfig,
    objective_config: ObjectiveConfig,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str = "cpu",
) -> SparseObjectiveResult:
    """Return a dense-equivalent objective through the SG1 validation surface."""

    rollout = rollout_batched_scenarios_with_controller(
        scenarios,
        beta=controller_parameters,
        controller=controller,
        base_params=base_params,
        rollout_config=rollout_config,
        horizon_k=scenarios.leader_x.shape[1] - 1,
        dtype=dtype,
        device=device,
    )
    aggregate, per_scenario = batched_rollout_objective(rollout, objective_config)
    return SparseObjectiveResult(
        sparse=aggregate,
        exact=aggregate,
        per_scenario_exact=per_scenario,
        rollout=rollout,
        spans=complete_spans(scenarios.leader_x.shape[1] - 1, 1),
        remainder_start=None,
        scenario_names=scenarios.scenario_names,
    )


@dataclass(frozen=True)
class _SpanContribution:
    components: ObjectiveComponents


@dataclass(frozen=True)
class _MacroStep:
    z_end: torch.Tensor
    accel: torch.Tensor
    components: ObjectiveComponents


def _macro_step_components(
    z_start: torch.Tensor,
    *,
    leader_x: torch.Tensor,
    leader_v: torch.Tensor,
    leader_length: torch.Tensor,
    span: SparseSpan,
    controller: HeadwayController,
    controller_parameters: torch.Tensor,
    params_base: IDMParameters,
    rollout_config: RolloutConfig,
    objective_config: ObjectiveConfig,
    dt: torch.Tensor,
    scenario_count: int,
    steps: int,
    previous_accel: torch.Tensor | None,
) -> _MacroStep:
    dtype = z_start.dtype
    device = z_start.device
    delta = dt * span.length
    x_start = z_start[:, 0]
    v_start = z_start[:, 1]
    gap_start = leader_x[:, span.start] - x_start - leader_length
    delta_v_start = v_start - leader_v[:, span.start]
    inputs = torch.stack([v_start, delta_v_start, gap_start], dim=-1)
    headway = controller(controller_parameters, inputs)
    params = params_base.with_updates({"time_headway": headway})
    accel = diffidm_acceleration(
        v_follower=v_start,
        v_leader=leader_v[:, span.start],
        gap=gap_start,
        params=params,
        dt=delta,
        prevent_negative_speed=rollout_config.prevent_negative_speed,
    )
    v_end = v_start + delta * accel
    x_end = x_start + delta * v_end
    gap_end = leader_x[:, span.end] - x_end - leader_length
    delta_v_end = v_end - leader_v[:, span.end]

    progress_scale = torch.tensor(objective_config.progress_scale, dtype=dtype, device=device)
    safety_scale = torch.tensor(objective_config.safety_scale, dtype=dtype, device=device)
    jerk_scale = torch.tensor(objective_config.jerk_scale, dtype=dtype, device=device)
    progress = torch.sum(torch.square((v_end - leader_v[:, span.end]) / progress_scale))
    progress = progress * (span.length / (scenario_count * steps))
    safe = safety_spacing(
        v=v_end,
        delta_v=delta_v_end,
        dtype=dtype,
        device=device,
        config=objective_config,
    )
    safety = torch.sum(torch.square(torch.nn.functional.softplus((safe - gap_end) / safety_scale)))
    safety = safety * (span.length / (scenario_count * steps))
    jerk_count = _assigned_jerk_count(span)
    if jerk_count == 0:
        jerk = torch.zeros((), dtype=dtype, device=device)
    else:
        if previous_accel is None:
            previous = accel.detach()
        else:
            previous = previous_accel.to(dtype=dtype, device=device).detach()
        jerk = torch.sum(torch.square((accel - previous) / jerk_scale))
        jerk = jerk * (jerk_count / (scenario_count * max(steps - 1, 1)))
    weighted_progress = torch.tensor(objective_config.progress_weight, dtype=dtype, device=device) * progress
    weighted_safety = torch.tensor(objective_config.safety_weight, dtype=dtype, device=device) * safety
    weighted_jerk = torch.tensor(objective_config.jerk_weight, dtype=dtype, device=device) * jerk
    total = weighted_progress + weighted_safety + weighted_jerk
    return _MacroStep(
        z_end=torch.stack([x_end, v_end], dim=-1),
        accel=accel,
        components=ObjectiveComponents(
            total=total,
            progress=progress,
            safety=safety,
            jerk=jerk,
            weighted_progress=weighted_progress,
            weighted_safety=weighted_safety,
            weighted_jerk=weighted_jerk,
        ),
    )


def _assigned_jerk_count(span: SparseSpan) -> int:
    return max(0, span.length - (1 if span.start == 0 else 0))


def _exact_span_contribution(
    result: RolloutResult,
    *,
    span: SparseSpan,
    objective_config: ObjectiveConfig,
) -> _SpanContribution:
    dtype = result.follower_v.dtype
    device = result.follower_v.device
    scenario_count = int(result.follower_v.shape[0])
    steps = int(result.follower_a.shape[1])
    if not (0 <= span.start <= span.end <= steps):
        raise ValueError(f"invalid span [{span.start}, {span.end}] for T={steps}")
    if span.start == span.end:
        return _SpanContribution(_zero_components(dtype=dtype, device=device))

    progress_scale = torch.tensor(objective_config.progress_scale, dtype=dtype, device=device)
    safety_scale = torch.tensor(objective_config.safety_scale, dtype=dtype, device=device)
    jerk_scale = torch.tensor(objective_config.jerk_scale, dtype=dtype, device=device)

    value_slice = slice(span.start + 1, span.end + 1)
    v = result.follower_v[:, value_slice]
    leader_v = result.leader_v[:, value_slice]
    gap = result.gap[:, value_slice]
    delta_v = result.delta_v[:, value_slice]
    progress = torch.sum(torch.square((v - leader_v) / progress_scale)) / (scenario_count * steps)
    safe = safety_spacing(v=v, delta_v=delta_v, dtype=dtype, device=device, config=objective_config)
    safety = torch.sum(torch.square(torch.nn.functional.softplus((safe - gap) / safety_scale)))
    safety = safety / (scenario_count * steps)

    if steps < 2:
        jerk = torch.zeros((), dtype=dtype, device=device)
    else:
        jerk_start = max(1, span.start)
        jerk_end = span.end
        if jerk_start >= jerk_end:
            jerk = torch.zeros((), dtype=dtype, device=device)
        else:
            diffs = result.follower_a[:, jerk_start:jerk_end] - result.follower_a[:, jerk_start - 1 : jerk_end - 1]
            jerk = torch.sum(torch.square(diffs / jerk_scale)) / (scenario_count * (steps - 1))
    weighted_progress = torch.tensor(objective_config.progress_weight, dtype=dtype, device=device) * progress
    weighted_safety = torch.tensor(objective_config.safety_weight, dtype=dtype, device=device) * safety
    weighted_jerk = torch.tensor(objective_config.jerk_weight, dtype=dtype, device=device) * jerk
    total = weighted_progress + weighted_safety + weighted_jerk
    return _SpanContribution(
        ObjectiveComponents(
            total=total,
            progress=progress,
            safety=safety,
            jerk=jerk,
            weighted_progress=weighted_progress,
            weighted_safety=weighted_safety,
            weighted_jerk=weighted_jerk,
        )
    )


def flatten_model_gradient(model: torch.nn.Module) -> torch.Tensor:
    values = []
    for parameter in model.parameters():
        if parameter.grad is None:
            values.append(torch.zeros_like(parameter).reshape(-1))
        else:
            values.append(parameter.grad.detach().reshape(-1))
    if not values:
        return torch.empty(0, dtype=torch.float64)
    return torch.cat(values)


def gradient_cosine(left: torch.Tensor, right: torch.Tensor) -> float | None:
    left_norm = torch.linalg.vector_norm(left)
    right_norm = torch.linalg.vector_norm(right)
    if float(left_norm.detach().cpu()) == 0.0 or float(right_norm.detach().cpu()) == 0.0:
        return None
    value = torch.dot(left.reshape(-1), right.reshape(-1)) / (left_norm * right_norm)
    return float(value.detach().cpu())


def relative_change(final: float, initial: float, *, epsilon: float = 1e-12) -> float:
    return (final - initial) / max(abs(initial), epsilon)


def sparse_rollouts(result: SparseObjectiveResult) -> tuple[RolloutResult, ...]:
    return tuple(split_batched_rollout(result.rollout))


def _zero_components(*, dtype: torch.dtype, device: torch.device | str) -> ObjectiveComponents:
    zero = torch.zeros((), dtype=dtype, device=device)
    return ObjectiveComponents(
        total=zero,
        progress=zero,
        safety=zero,
        jerk=zero,
        weighted_progress=zero,
        weighted_safety=zero,
        weighted_jerk=zero,
    )


def _components_detach(components: ObjectiveComponents) -> ObjectiveComponents:
    return ObjectiveComponents(**{name: getattr(components, name).detach() for name in components.__dataclass_fields__})


def _components_add(left: ObjectiveComponents, right: ObjectiveComponents) -> ObjectiveComponents:
    return ObjectiveComponents(
        **{name: getattr(left, name) + getattr(right, name) for name in left.__dataclass_fields__}
    )


def _components_sub(left: ObjectiveComponents, right: ObjectiveComponents) -> ObjectiveComponents:
    return ObjectiveComponents(
        **{name: getattr(left, name) - getattr(right, name) for name in left.__dataclass_fields__}
    )


def finite_gradient_norm(gradient: torch.Tensor) -> tuple[bool, float]:
    finite = bool(torch.all(torch.isfinite(gradient)).detach().cpu())
    if not finite:
        return False, math.nan
    return True, float(torch.linalg.vector_norm(gradient).detach().cpu())
