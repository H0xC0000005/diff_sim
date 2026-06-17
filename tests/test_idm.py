import torch
from diffidm import IDMLayer

from differential_sim.idm import (
    IDMParameters,
    diffidm_acceleration,
    idm_acceleration_from_spacing,
    idm_optimal_spacing,
    parameters_to_tensors,
    smooth_clamped_idm_acceleration_reference,
    textbook_idm_acceleration,
)


def _params(dtype=torch.float64):
    return parameters_to_tensors(
        IDMParameters(a_max=1.4, b_comfort=2.0, v0=30.0, s0=2.0, time_headway=1.5, a_min=-8.0),
        dtype=dtype,
        device="cpu",
    )


def test_textbook_helpers_match_diffidm_unclamped_helpers():
    dtype = torch.float64
    params = _params(dtype)
    v_follower = torch.tensor([12.0, 18.0], dtype=dtype)
    v_leader = torch.tensor([15.0, 17.0], dtype=dtype)
    gap = torch.tensor([28.0, 35.0], dtype=dtype)
    delta_v = v_follower - v_leader

    spacing = idm_optimal_spacing(v_follower=v_follower, delta_v=delta_v, params=params)
    diffidm_spacing = IDMLayer.compute_optimal_spacing(
        params.a_max,
        params.b_comfort,
        v_follower,
        delta_v,
        params.s0,
        params.time_headway,
    )
    assert torch.allclose(spacing, diffidm_spacing, rtol=1e-12, atol=1e-12)

    acc = idm_acceleration_from_spacing(
        v_follower=v_follower,
        gap=gap,
        optimal_spacing=spacing,
        params=params,
    )
    diffidm_acc = IDMLayer.compute_acceleration(params.a_max, v_follower, params.v0, gap, spacing)
    textbook_acc = textbook_idm_acceleration(
        v_follower=v_follower,
        v_leader=v_leader,
        gap=gap,
        params=params,
    )
    assert torch.allclose(acc, diffidm_acc, rtol=1e-12, atol=1e-12)
    assert torch.allclose(acc, textbook_acc, rtol=1e-12, atol=1e-12)


def test_smooth_reference_matches_diffidm_apply():
    dtype = torch.float64
    params = _params(dtype)
    v_follower = torch.tensor([12.0, 18.0], dtype=dtype, requires_grad=True)
    v_leader = torch.tensor([15.0, 17.0], dtype=dtype)
    gap = torch.tensor([28.0, 35.0], dtype=dtype)
    dt = torch.tensor(0.2, dtype=dtype)

    ref = smooth_clamped_idm_acceleration_reference(
        v_follower=v_follower,
        v_leader=v_leader,
        gap=gap,
        params=params,
        dt=dt,
    )
    wrapped = diffidm_acceleration(
        v_follower=v_follower,
        v_leader=v_leader,
        gap=gap,
        params=params,
        dt=dt,
    )
    assert ref.dtype == dtype
    assert wrapped.dtype == dtype
    assert torch.allclose(ref, wrapped, rtol=1e-12, atol=1e-12)


def test_diffidm_gradients_are_finite():
    dtype = torch.float64
    params = _params(dtype)
    v_follower = torch.tensor([12.0, 18.0], dtype=dtype, requires_grad=True)
    v_leader = torch.tensor([15.0, 17.0], dtype=dtype)
    gap = torch.tensor([28.0, 35.0], dtype=dtype)
    dt = torch.tensor(0.2, dtype=dtype)

    acc = diffidm_acceleration(
        v_follower=v_follower,
        v_leader=v_leader,
        gap=gap,
        params=params,
        dt=dt,
    )
    acc.sum().backward()
    assert v_follower.grad is not None
    assert torch.isfinite(v_follower.grad).all()
