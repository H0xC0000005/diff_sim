import torch

from differential_sim.fit import (
    FitConfig,
    autograd_directional_derivative,
    central_finite_difference_directional_derivative,
    fit_synthetic_parameters,
    generate_synthetic_trajectory,
    normalized_parameter_distance,
    trajectory_loss,
)
from differential_sim.idm import (
    IDMParameters,
    build_parameter_tensors,
    values_to_unconstrained,
)
from differential_sim.rollout import RolloutConfig, rollout_follower
from differential_sim.scenarios import ScenarioConfig, leader_profile


def _case(steps=40):
    dtype = torch.float64
    scenario = ScenarioConfig(kind="braking_recovery", steps=steps, dt=0.2)
    leader = leader_profile(scenario, dtype=dtype)
    rollout_config = RolloutConfig(dt=scenario.dt, acceleration_mode="diffidm")
    truth = IDMParameters(
        a_max=1.4,
        b_comfort=2.0,
        v0=28.0,
        s0=2.0,
        time_headway=1.4,
        a_min=-8.0,
    )
    reference = generate_synthetic_trajectory(leader, truth, rollout_config, dtype=dtype)
    return dtype, leader, rollout_config, truth, reference


def test_fitting_reduces_loss_and_moves_toward_truth():
    dtype, leader, rollout_config, truth, reference = _case(steps=50)
    fitted_names = ("time_headway", "v0")
    initial_values = {"time_headway": 2.2, "v0": 20.0}
    truth_values = {"time_headway": 1.4, "v0": 28.0}
    fit_config = FitConfig(fitted_names=fitted_names, steps=120, learning_rate=0.04, seed=0)

    result = fit_synthetic_parameters(
        leader=leader,
        reference=reference,
        base_params=truth,
        truth_values=truth_values,
        initial_values=initial_values,
        rollout_config=rollout_config,
        fit_config=fit_config,
        dtype=dtype,
    )

    initial_distance = normalized_parameter_distance(result.initial_parameters, result.truth_parameters, fitted_names)
    final_distance = normalized_parameter_distance(result.final_parameters, result.truth_parameters, fitted_names)

    assert result.final_loss < 0.5 * result.initial_loss
    assert final_distance < initial_distance
    assert result.gradients_finite


def test_three_parameter_fit_configuration_runs_with_finite_gradients():
    dtype, leader, rollout_config, truth, reference = _case(steps=20)
    fitted_names = ("time_headway", "v0", "s0")
    result = fit_synthetic_parameters(
        leader=leader,
        reference=reference,
        base_params=truth,
        truth_values={"time_headway": 1.4, "v0": 28.0, "s0": 2.0},
        initial_values={"time_headway": 2.2, "v0": 20.0, "s0": 5.0},
        rollout_config=rollout_config,
        fit_config=FitConfig(fitted_names=fitted_names, steps=5, learning_rate=0.02, seed=0),
        dtype=dtype,
    )
    assert result.gradients_finite
    assert torch.isfinite(torch.tensor(result.final_loss, dtype=dtype))


def test_central_finite_difference_matches_autograd_directional_derivative():
    dtype, leader, rollout_config, truth, reference = _case(steps=12)
    fitted_names = ("time_headway", "v0")
    raw_values = values_to_unconstrained(
        {"time_headway": 1.9, "v0": 22.0},
        dtype=dtype,
        device="cpu",
    )
    raw_values = {name: raw.detach().clone().requires_grad_(True) for name, raw in raw_values.items()}
    direction = {
        "time_headway": torch.tensor(0.6, dtype=dtype),
        "v0": torch.tensor(-0.8, dtype=dtype),
    }

    def loss_fn(raw):
        params = build_parameter_tensors(
            truth,
            raw,
            fitted_names,
            dtype=dtype,
            device="cpu",
        )
        pred = rollout_follower(leader, params, rollout_config, dtype=dtype)
        return trajectory_loss(pred, reference)

    loss = loss_fn(raw_values)
    autograd_dd = autograd_directional_derivative(loss, raw_values, direction)
    errors = []
    for epsilon in (1e-4, 3e-5, 1e-5):
        fd_dd = central_finite_difference_directional_derivative(
            loss_fn,
            raw_values,
            direction,
            epsilon,
        )
        abs_err = torch.abs(fd_dd - autograd_dd)
        rel_err = abs_err / torch.clamp(torch.abs(autograd_dd), min=torch.tensor(1e-12, dtype=dtype))
        errors.append((float(abs_err.item()), float(rel_err.item())))

    assert any(rel < 1e-3 or abs_err < 1e-6 for abs_err, rel in errors)
