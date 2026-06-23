import torch

from differential_sim.controllers import StructuredHeadwayController, center_beta
from differential_sim.milestone1_diagnostics import diagnostic_scenarios
from differential_sim.temporal_gradients import (
    default_rollout_config,
    max_forward_difference,
    objective_for_scenarios,
    rollout_scenarios_with_controller,
)
from differential_sim.objectives import default_idm_params


def _case(steps=12):
    dtype = torch.float64
    scenarios = [(name, config.__class__(**{**config.__dict__, "steps": steps})) for name, config in diagnostic_scenarios()[:2]]
    controller = StructuredHeadwayController(input_parameterization="si_units")
    beta = center_beta(1.4, dtype=dtype) + torch.tensor([0.01, -0.02, 0.005, 0.003], dtype=dtype)
    return dtype, scenarios, controller, beta, default_idm_params(), default_rollout_config()


def test_forward_trajectories_and_objectives_identical_across_horizons():
    dtype, scenarios, controller, beta, params, rollout_config = _case(steps=12)
    full = rollout_scenarios_with_controller(
        scenarios,
        beta=beta,
        controller=controller,
        base_params=params,
        rollout_config=rollout_config,
        horizon_k=12,
        dtype=dtype,
    )
    full_obj, _ = objective_for_scenarios(
        scenarios,
        beta=beta,
        controller=controller,
        base_params=params,
        rollout_config=rollout_config,
        horizon_k=12,
        dtype=dtype,
    )

    for horizon in (1, 3, 6, 10, 12):
        current = rollout_scenarios_with_controller(
            scenarios,
            beta=beta,
            controller=controller,
            base_params=params,
            rollout_config=rollout_config,
            horizon_k=horizon,
            dtype=dtype,
        )
        current_obj, _ = objective_for_scenarios(
            scenarios,
            beta=beta,
            controller=controller,
            base_params=params,
            rollout_config=rollout_config,
            horizon_k=horizon,
            dtype=dtype,
        )
        for a, b in zip(full, current, strict=True):
            assert max_forward_difference(a.result, b.result) == 0.0
        assert torch.equal(full_obj.total.detach(), current_obj.total.detach())


def test_k_equals_t_gradient_matches_no_detach_full_autograd_repeated_call():
    dtype, scenarios, controller, beta, params, rollout_config = _case(steps=10)
    beta_a = beta.detach().clone().requires_grad_(True)
    obj_a, _ = objective_for_scenarios(
        scenarios,
        beta=beta_a,
        controller=controller,
        base_params=params,
        rollout_config=rollout_config,
        horizon_k=10,
        dtype=dtype,
    )
    grad_a = torch.autograd.grad(obj_a.total, beta_a)[0]
    beta_b = beta.detach().clone().requires_grad_(True)
    obj_b, _ = objective_for_scenarios(
        scenarios,
        beta=beta_b,
        controller=controller,
        base_params=params,
        rollout_config=rollout_config,
        horizon_k=10,
        dtype=dtype,
    )
    grad_b = torch.autograd.grad(obj_b.total, beta_b)[0]

    assert torch.allclose(grad_a, grad_b, rtol=0.0, atol=1e-12)


def test_full_gradient_matches_central_finite_difference_directional_derivative():
    dtype, scenarios, controller, beta, params, rollout_config = _case(steps=8)
    direction = torch.tensor([0.4, -0.3, 0.2, -0.1], dtype=dtype)
    direction = direction / torch.linalg.vector_norm(direction)
    beta_grad = beta.detach().clone().requires_grad_(True)
    obj, _ = objective_for_scenarios(
        scenarios,
        beta=beta_grad,
        controller=controller,
        base_params=params,
        rollout_config=rollout_config,
        horizon_k=8,
        dtype=dtype,
    )
    grad = torch.autograd.grad(obj.total, beta_grad)[0]
    autograd_dd = torch.dot(grad, direction)

    errors = []
    for epsilon in (1e-4, 3e-5, 1e-5):
        plus, _ = objective_for_scenarios(
            scenarios,
            beta=beta + epsilon * direction,
            controller=controller,
            base_params=params,
            rollout_config=rollout_config,
            horizon_k=8,
            dtype=dtype,
        )
        minus, _ = objective_for_scenarios(
            scenarios,
            beta=beta - epsilon * direction,
            controller=controller,
            base_params=params,
            rollout_config=rollout_config,
            horizon_k=8,
            dtype=dtype,
        )
        fd_dd = (plus.total - minus.total) / (2.0 * epsilon)
        abs_err = torch.abs(fd_dd - autograd_dd)
        rel_err = abs_err / torch.clamp(torch.abs(autograd_dd), min=torch.tensor(1e-12, dtype=dtype))
        errors.append((float(abs_err.item()), float(rel_err.item())))

    assert any(rel < 1e-3 or abs_err < 1e-6 for abs_err, rel in errors)


def test_truncation_cuts_future_state_credit_but_keeps_local_dependence():
    dtype, scenarios, controller, beta, params, rollout_config = _case(steps=8)
    beta_full = beta.detach().clone().requires_grad_(True)
    full_obj, _ = objective_for_scenarios(
        scenarios,
        beta=beta_full,
        controller=controller,
        base_params=params,
        rollout_config=rollout_config,
        horizon_k=8,
        dtype=dtype,
    )
    full_grad = torch.autograd.grad(full_obj.total, beta_full)[0]
    beta_local = beta.detach().clone().requires_grad_(True)
    local_obj, _ = objective_for_scenarios(
        scenarios,
        beta=beta_local,
        controller=controller,
        base_params=params,
        rollout_config=rollout_config,
        horizon_k=1,
        dtype=dtype,
    )
    local_grad = torch.autograd.grad(local_obj.total, beta_local)[0]

    assert torch.linalg.vector_norm(local_grad) > 0.0
    assert not torch.allclose(full_grad, local_grad, rtol=1e-8, atol=1e-10)
