import torch

from differential_sim.idm import IDMParameters
from differential_sim.rollout import RolloutConfig, rollout_follower
from differential_sim.scenarios import ScenarioConfig, leader_profile, random_braking_cycle_events


def test_leader_profiles_are_deterministic_and_finite():
    configs = [
        ScenarioConfig(kind="constant", steps=20, dt=0.2),
        ScenarioConfig(kind="braking_recovery", steps=20, dt=0.2),
        ScenarioConfig(kind="sinusoidal", steps=20, dt=0.2),
        ScenarioConfig(kind="random_braking_cycles", steps=20, dt=0.2, seed=101),
        ScenarioConfig(
            kind="multi_pulse_braking",
            steps=20,
            dt=0.2,
            pulse_starts=(1.0, 2.5),
            pulse_brake_delta_v=(1.0, 1.5),
            pulse_brake_durations=(0.6, 0.8),
            pulse_recovery_durations=(0.8, 1.0),
        ),
        ScenarioConfig(kind="chirp_sinusoidal", steps=20, dt=0.2),
        ScenarioConfig(kind="mixed_regime", steps=80, dt=0.2),
    ]
    for config in configs:
        first = leader_profile(config, dtype=torch.float64)
        second = leader_profile(config, dtype=torch.float64)
        assert torch.equal(first.time, second.time)
        assert torch.equal(first.position, second.position)
        assert torch.equal(first.speed, second.speed)
        assert torch.isfinite(first.position).all()
        assert torch.isfinite(first.speed).all()


def test_random_braking_cycle_sampled_events_are_deterministic():
    config = ScenarioConfig(kind="random_braking_cycles", steps=20, dt=0.2, seed=101)

    assert random_braking_cycle_events(config) == random_braking_cycle_events(config)
    assert random_braking_cycle_events(config)


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
