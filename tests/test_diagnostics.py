import torch

from differential_sim.diagnostics import PRIMARY_ALPHA, diagnostic_scenarios, generate_random_directions, run_one_step_diagnostics


def test_random_directions_are_unit_norm_and_deterministic():
    first = generate_random_directions(
        count=4,
        init_index=0,
        input_parameterization="normalized",
        dtype=torch.float64,
        device="cpu",
    )
    second = generate_random_directions(
        count=4,
        init_index=0,
        input_parameterization="normalized",
        dtype=torch.float64,
        device="cpu",
    )

    for a, b in zip(first, second, strict=True):
        assert torch.equal(a, b)
        assert torch.allclose(torch.linalg.vector_norm(a), torch.tensor(1.0, dtype=torch.float64), atol=1e-12)


def test_diagnostics_smoke_outputs_are_separated_and_include_zero_flags():
    artifacts = run_one_step_diagnostics(
        input_parameterization="normalized",
        random_direction_count=2,
        dtype=torch.float64,
        device="cpu",
    )

    assert artifacts.rows
    assert artifacts.summary["input_parameterization"] == "normalized"
    assert artifacts.summary["normalization"] is not None
    assert len(artifacts.summary["scenarios"]["diagnostic"]) == 14
    assert len(artifacts.summary["scenarios"]["held_out"]) == 8
    assert artifacts.summary["objective_weights"] == {"progress": 1.0, "safety": 0.7, "jerk": 5.0}
    assert all(row["input_parameterization"] == "normalized" for row in artifacts.rows)
    assert any(row["direction_type"] == "gradient" for row in artifacts.rows)
    assert any(row["direction_type"] == "random" for row in artifacts.rows)
    assert any(row["alpha"] == PRIMARY_ALPHA for row in artifacts.rows)
    assert all("gradient_cosine_to_full" in row for row in artifacts.rows)
    assert all("weighted_components_before" in row for row in artifacts.rows)
    assert artifacts.held_out["used_for_selection"] is False
    assert all(row["grad_enabled"] is False for row in artifacts.held_out["rows"])


def test_si_unit_diagnostics_have_no_normalization():
    artifacts = run_one_step_diagnostics(
        input_parameterization="si_units",
        random_direction_count=1,
        dtype=torch.float64,
        device="cpu",
    )

    assert artifacts.summary["input_parameterization"] == "si_units"
    assert artifacts.summary["normalization"] is None


def test_diagnostic_scenario_set_has_amended_size():
    assert len(diagnostic_scenarios()) == 14
