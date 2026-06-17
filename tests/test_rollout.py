import torch

from differential_sim.idm import IDMParameters
from differential_sim.rollout import RolloutConfig, rollout_follower
from differential_sim.scenarios import ScenarioConfig, leader_profile


def test_leader_profiles_are_deterministic_and_finite():
    for kind in ("constant", "braking_recovery", "sinusoidal"):
        config = ScenarioConfig(kind=kind, steps=20, dt=0.2)
        first = leader_profile(config, dtype=torch.float64)
        second = leader_profile(config, dtype=torch.float64)
        assert torch.equal(first.time, second.time)
        assert torch.equal(first.position, second.position)
        assert torch.equal(first.speed, second.speed)
        assert torch.isfinite(first.position).all()
        assert torch.isfinite(first.speed).all()


def test_rollout_shapes_and_determinism():
    dtype = torch.float64
    scenario = ScenarioConfig(kind="braking_recovery", steps=30, dt=0.2)
    leader = leader_profile(scenario, dtype=dtype)
    params = IDMParameters(a_max=1.4, b_comfort=2.0, v0=28.0, s0=2.0, time_headway=1.4)
    rollout_config = RolloutConfig(dt=scenario.dt, acceleration_mode="diffidm")

    first = rollout_follower(leader, params, rollout_config, dtype=dtype)
    second = rollout_follower(leader, params, rollout_config, dtype=dtype)

    assert first.leader_x.shape == (scenario.steps + 1,)
    assert first.leader_v.shape == (scenario.steps + 1,)
    assert first.follower_x.shape == (scenario.steps + 1,)
    assert first.follower_v.shape == (scenario.steps + 1,)
    assert first.follower_a.shape == (scenario.steps,)
    assert first.gap.shape == (scenario.steps + 1,)
    assert first.delta_v.shape == (scenario.steps + 1,)

    for field in ("follower_x", "follower_v", "follower_a", "gap", "delta_v"):
        assert torch.equal(getattr(first, field), getattr(second, field))
        assert torch.isfinite(getattr(first, field)).all()
    assert torch.all(first.gap > 0.0)
