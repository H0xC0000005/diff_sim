import torch

from differential_sim.batched_temporal_gradients import batched_rollout_objective
from differential_sim.device_parity import build_parity_context, max_result_difference
from differential_sim.small_model_training import (
    build_small_model,
    evaluate_held_out,
    evaluate_split,
    flatten_gradients,
    prepare_split,
)
from differential_sim.sparse_gradients import complete_spans, sparse_b1_objective, sparse_rollouts
from differential_sim.sparse_gradients import dense_equivalent_vjp_objective


def small_context():
    return build_parity_context(train_limit=2, held_out_limit=1, init_limit=1)


def test_complete_spans_ignore_m6_remainder():
    spans = complete_spans(80, 6)

    assert [(span.start, span.end) for span in spans] == [
        (0, 6),
        (6, 12),
        (12, 18),
        (18, 24),
        (24, 30),
        (30, 36),
        (36, 42),
        (42, 48),
        (48, 54),
        (54, 60),
        (60, 66),
        (66, 72),
        (72, 78),
    ]


def test_sparse_b1_preserves_forward_rollout_and_objective_values():
    context = small_context()
    split = prepare_split(context.train)
    model = build_small_model(context.normalization, seed=1000)
    dense = evaluate_split(context, split, model=model, horizon=80, mode="scenario-batched")
    result = sparse_b1_objective(
        split.batched,
        controller=model,
        controller_parameters=torch.empty(0, dtype=torch.float64),
        base_params=context.base_params,
        rollout_config=context.rollout_config,
        objective_config=context.objective_config,
        stride=4,
    )

    assert max_result_difference(dense.rollouts, sparse_rollouts(result)) <= 1e-10
    assert torch.allclose(result.exact.total, dense.aggregate.total, atol=1e-10, rtol=1e-8)
    assert torch.allclose(result.sparse.total.detach(), dense.aggregate.total, atol=1e-10, rtol=1e-8)


def test_sparse_b1_m6_keeps_full_forward_objective_with_no_short_span():
    context = small_context()
    split = prepare_split(context.train)
    model = build_small_model(context.normalization, seed=1000)
    result = sparse_b1_objective(
        split.batched,
        controller=model,
        controller_parameters=torch.empty(0, dtype=torch.float64),
        base_params=context.base_params,
        rollout_config=context.rollout_config,
        objective_config=context.objective_config,
        stride=6,
    )
    dense_components, _ = batched_rollout_objective(result.rollout, context.objective_config)

    assert result.remainder_start == 78
    assert (78, 80) not in [(span.start, span.end) for span in result.spans]
    assert torch.allclose(result.exact.total, dense_components.total, atol=1e-10, rtol=1e-8)
    assert torch.allclose(result.sparse.total.detach(), dense_components.total, atol=1e-10, rtol=1e-8)


def test_sparse_b1_gradient_is_finite_nonzero_and_differs_from_dense():
    context = small_context()
    split = prepare_split(context.train)
    dense_model = build_small_model(context.normalization, seed=1000)
    sparse_model = build_small_model(context.normalization, seed=1000)

    dense = evaluate_split(context, split, model=dense_model, horizon=80, mode="scenario-batched")
    dense.aggregate.total.backward()
    dense_gradient = flatten_gradients(dense_model)

    sparse = sparse_b1_objective(
        split.batched,
        controller=sparse_model,
        controller_parameters=torch.empty(0, dtype=torch.float64),
        base_params=context.base_params,
        rollout_config=context.rollout_config,
        objective_config=context.objective_config,
        stride=4,
    )
    sparse.sparse.total.backward()
    sparse_gradient = flatten_gradients(sparse_model)

    assert torch.all(torch.isfinite(sparse_gradient))
    assert torch.linalg.vector_norm(sparse_gradient) > 1e-12
    assert not torch.allclose(sparse_gradient, dense_gradient, atol=1e-10, rtol=1e-8)


def test_sparse_b1_does_not_retain_dense_microstep_connectivity():
    context = small_context()
    split = prepare_split(context.train)
    model = build_small_model(context.normalization, seed=1000)
    result = sparse_b1_objective(
        split.batched,
        controller=model,
        controller_parameters=torch.empty(0, dtype=torch.float64),
        base_params=context.base_params,
        rollout_config=context.rollout_config,
        objective_config=context.objective_config,
        stride=4,
    )

    assert result.rollout.follower_x[:, 1].requires_grad is False
    assert result.rollout.follower_v[:, 1].requires_grad is False
    assert result.sparse.total.requires_grad is True


def test_dense_equivalent_vjp_matches_dense_autograd_on_required_scenarios():
    context = build_parity_context(train_limit=None, held_out_limit=1, init_limit=1)
    selected = [item for item in context.train if item[0] in {"mixed_regime", "brake_stronger"}]
    split = prepare_split(selected)
    dense_model = build_small_model(context.normalization, seed=1000)
    vjp_model = build_small_model(context.normalization, seed=1000)

    dense = evaluate_split(context, split, model=dense_model, horizon=80, mode="scenario-batched")
    dense.aggregate.total.backward()
    dense_gradient = flatten_gradients(dense_model)

    vjp = dense_equivalent_vjp_objective(
        split.batched,
        controller=vjp_model,
        controller_parameters=torch.empty(0, dtype=torch.float64),
        base_params=context.base_params,
        rollout_config=context.rollout_config,
        objective_config=context.objective_config,
    )
    vjp.sparse.total.backward()
    vjp_gradient = flatten_gradients(vjp_model)
    diff = torch.linalg.vector_norm(vjp_gradient - dense_gradient)
    norm = torch.clamp(torch.linalg.vector_norm(dense_gradient), min=1e-12)

    assert diff / norm <= 1e-4


def test_heldout_helper_remains_no_grad_for_sg1_context():
    context = small_context()
    held_out = prepare_split(context.held_out)
    model = build_small_model(context.normalization, seed=1000)

    evaluation, grad_enabled = evaluate_held_out(
        context,
        held_out,
        model=model,
        mode="scenario-batched",
    )

    assert grad_enabled is False
    assert evaluation.aggregate.total.requires_grad is False
