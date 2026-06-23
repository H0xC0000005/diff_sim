import torch

from differential_sim.controllers import (
    HeadwayBounds,
    InputNormalization,
    StructuredHeadwayController,
    center_beta,
    noisy_center_betas,
)


def test_structured_headway_controller_bounds_for_extreme_inputs():
    dtype = torch.float64
    normalization = InputNormalization(
        mean=torch.zeros(3, dtype=dtype),
        sigma=torch.ones(3, dtype=dtype),
        sigma_floor_used=False,
    )
    controller = StructuredHeadwayController(normalization=normalization)
    beta = torch.tensor([0.0, 10.0, -10.0, 4.0], dtype=dtype)
    inputs = torch.tensor(
        [[0.0, 0.0, 0.0], [100.0, -100.0, 200.0], [-100.0, 100.0, -200.0]],
        dtype=dtype,
    )

    headway = controller(beta, inputs)

    assert torch.all(headway >= 0.5)
    assert torch.all(headway <= 3.0)


def test_center_beta_maps_to_requested_headway():
    dtype = torch.float64
    controller = StructuredHeadwayController(input_parameterization="si_units")
    beta = center_beta(1.4, dtype=dtype)

    headway = controller(beta, torch.zeros(3, dtype=dtype))

    assert torch.allclose(headway, torch.tensor(1.4, dtype=dtype), rtol=0.0, atol=1e-12)


def test_noisy_center_betas_are_seeded_and_shaped():
    first = noisy_center_betas(dtype=torch.float64)
    second = noisy_center_betas(dtype=torch.float64)

    assert len(first) == 6
    assert all(beta.shape == (4,) for beta in first)
    for a, b in zip(first, second, strict=True):
        assert torch.equal(a, b)
