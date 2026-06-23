"""Bounded structured headway controller for Milestone 1."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Literal, Sequence

import torch

from differential_sim.idm import IDMParameters
from differential_sim.rollout import RolloutConfig, rollout_follower
from differential_sim.scenarios import ScenarioConfig, leader_profile


InputParameterization = Literal["normalized", "si_units"]


@dataclass(frozen=True)
class HeadwayBounds:
    minimum: float = 0.5
    maximum: float = 3.0


@dataclass(frozen=True)
class InputNormalization:
    mean: torch.Tensor
    sigma: torch.Tensor
    sigma_floor_used: bool


@dataclass(frozen=True)
class StructuredHeadwayController:
    bounds: HeadwayBounds = HeadwayBounds()
    normalization: InputNormalization | None = None
    input_parameterization: InputParameterization = "normalized"

    def transform_inputs(self, inputs: torch.Tensor) -> torch.Tensor:
        """Transform ``[..., 3]`` inputs ``[v, delta_v, gap]``."""

        if inputs.shape[-1] != 3:
            raise ValueError(f"expected last dimension 3, got {inputs.shape}")
        if self.input_parameterization == "si_units":
            return inputs
        if self.normalization is None:
            raise ValueError("normalized controller requires InputNormalization")
        return (inputs - self.normalization.mean.to(dtype=inputs.dtype, device=inputs.device)) / (
            self.normalization.sigma.to(dtype=inputs.dtype, device=inputs.device)
        )

    def __call__(self, beta: torch.Tensor, inputs: torch.Tensor) -> torch.Tensor:
        """Return bounded time headway in seconds.

        ``beta`` has shape ``[4]`` and inputs have shape ``[..., 3]``.
        """

        if beta.shape != (4,):
            raise ValueError(f"expected beta shape [4], got {tuple(beta.shape)}")
        z = self.transform_inputs(inputs)
        logits = beta[0] + torch.sum(beta[1:] * z, dim=-1)
        return self.bounds.minimum + (self.bounds.maximum - self.bounds.minimum) * torch.sigmoid(logits)


def center_beta(
    time_headway: float,
    *,
    bounds: HeadwayBounds = HeadwayBounds(),
    dtype: torch.dtype = torch.float64,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Return zero-slope beta centered at a physical time headway."""

    y = (time_headway - bounds.minimum) / (bounds.maximum - bounds.minimum)
    if not 0.0 < y < 1.0:
        raise ValueError("time_headway must be strictly inside bounds")
    beta0 = math.log(y / (1.0 - y))
    return torch.tensor([beta0, 0.0, 0.0, 0.0], dtype=dtype, device=device)


def noisy_center_betas(
    centers: Sequence[float] = (0.9, 1.2, 1.4, 1.6, 1.9, 2.2),
    seeds: Sequence[int] = (0, 1, 2, 3, 4, 5),
    *,
    noise_scale: float = 0.01,
    bounds: HeadwayBounds = HeadwayBounds(),
    dtype: torch.dtype = torch.float64,
    device: torch.device | str = "cpu",
) -> list[torch.Tensor]:
    if len(centers) != len(seeds):
        raise ValueError("centers and seeds must have the same length")
    betas: list[torch.Tensor] = []
    for center, seed in zip(centers, seeds, strict=True):
        generator = torch.Generator(device=str(device) if str(device) != "cpu" else "cpu")
        generator.manual_seed(int(seed))
        xi = torch.randn(4, generator=generator, dtype=dtype, device=device)
        betas.append(center_beta(center, bounds=bounds, dtype=dtype, device=device) + noise_scale * xi)
    return betas


def collect_controller_inputs(
    scenario_configs: Sequence[ScenarioConfig],
    base_params: IDMParameters,
    rollout_config: RolloutConfig,
    *,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Collect controller inputs ``[v_t, delta_v_t, gap_t]`` over rollout times."""

    rows: list[torch.Tensor] = []
    for config in scenario_configs:
        leader = leader_profile(config, dtype=dtype, device=device)
        result = rollout_follower(leader, base_params, rollout_config, dtype=dtype, device=device)
        rows.append(torch.stack([result.follower_v[:-1], result.delta_v[:-1], result.gap[:-1]], dim=-1))
    return torch.cat(rows, dim=0)


def input_normalization_from_scenarios(
    scenario_configs: Sequence[ScenarioConfig],
    base_params: IDMParameters,
    rollout_config: RolloutConfig,
    *,
    sigma_floor: float = 1e-12,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str = "cpu",
) -> InputNormalization:
    inputs = collect_controller_inputs(
        scenario_configs,
        base_params,
        rollout_config,
        dtype=dtype,
        device=device,
    )
    mean = torch.mean(inputs, dim=0)
    sigma_raw = torch.std(inputs, dim=0, unbiased=False)
    floor = torch.tensor(sigma_floor, dtype=dtype, device=device)
    sigma = torch.maximum(sigma_raw, floor)
    return InputNormalization(
        mean=mean.detach().clone(),
        sigma=sigma.detach().clone(),
        sigma_floor_used=bool(torch.any(sigma_raw < floor).item()),
    )


def tensor_to_list(tensor: torch.Tensor) -> list[float]:
    return [float(v) for v in tensor.detach().cpu().reshape(-1)]
