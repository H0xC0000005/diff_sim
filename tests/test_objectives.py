import torch

from differential_sim.gradient_modes import default_rollout_config
from differential_sim.objectives import default_idm_params, rollout_objective, semantic_inverse_mean_weights
from differential_sim.rollout import rollout_follower
from differential_sim.scenarios import ScenarioConfig, leader_profile


def test_objective_components_are_finite_and_total_is_weighted_sum():
    dtype = torch.float64
    scenario = ScenarioConfig(kind="braking_recovery", steps=20, dt=0.2)
    leader = leader_profile(scenario, dtype=dtype)
    result = rollout_follower(leader, default_idm_params(), default_rollout_config(), dtype=dtype)

    components = rollout_objective(result)

    assert torch.isfinite(components.total)
    assert torch.isfinite(components.progress)
    assert torch.isfinite(components.safety)
    assert torch.isfinite(components.jerk)
    assert torch.isfinite(components.weighted_progress)
    assert torch.isfinite(components.weighted_safety)
    assert torch.isfinite(components.weighted_jerk)
    assert torch.allclose(
        components.total,
        components.weighted_progress + components.weighted_safety + components.weighted_jerk,
        rtol=0.0,
        atol=1e-12,
    )


def test_jerk_is_zero_for_one_step_rollout():
    dtype = torch.float64
    scenario = ScenarioConfig(kind="constant", steps=1, dt=0.2, initial_speed=16.0)
    leader = leader_profile(scenario, dtype=dtype)
    result = rollout_follower(leader, default_idm_params(), default_rollout_config(), dtype=dtype)

    components = rollout_objective(result)

    assert torch.equal(components.jerk, torch.tensor(0.0, dtype=dtype))


def test_semantic_weights_are_diagnostic_only_inverse_means():
    weights = semantic_inverse_mean_weights(
        [
            {"progress": 2.0, "safety": 4.0, "jerk": 8.0},
            {"progress": 6.0, "safety": 4.0, "jerk": 16.0},
        ]
    )

    assert weights == {"progress": 0.25, "safety": 0.25, "jerk": 1.0 / 12.0}
