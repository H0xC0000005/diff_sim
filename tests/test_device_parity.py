from dataclasses import dataclass

import pytest
import torch

from differential_sim.device_parity import (
    build_parity_context,
    device_metadata,
    json_safe,
    objective_snapshot,
    resolve_device,
    tolerance_passed,
    vector_abs_rel,
)


@dataclass(frozen=True)
class DummyData:
    value: torch.Tensor


def test_json_safe_handles_tensors_and_dataclasses():
    value = {
        "scalar": torch.tensor(1.25, dtype=torch.float64),
        "vector": torch.tensor([1.0, 2.0], dtype=torch.float64),
        "dataclass": DummyData(torch.tensor([3.0], dtype=torch.float64)),
    }

    converted = json_safe(value)

    assert converted == {"scalar": 1.25, "vector": [1.0, 2.0], "dataclass": {"value": [3.0]}}


def test_tolerance_and_vector_difference_helpers():
    left = torch.tensor([1.0, 2.0], dtype=torch.float64)
    right = torch.tensor([1.0 + 1e-8, 2.0 - 2e-8], dtype=torch.float64)

    abs_diff, rel_diff = vector_abs_rel(left, right)

    assert abs_diff > 0.0
    assert rel_diff > 0.0
    assert tolerance_passed(abs_diff, rel_diff, atol=1e-7, rtol=1e-6)
    assert not tolerance_passed(abs_diff, rel_diff, atol=1e-12, rtol=1e-12)


def test_device_metadata_cpu_schema():
    metadata = device_metadata("cpu")

    assert metadata.requested == "cpu"
    assert metadata.actual == "cpu"
    assert metadata.dtype == "torch.float64"
    assert metadata.torch_version
    assert metadata.cuda_device_count >= 0
    assert metadata.gpu_name is None


def test_objective_snapshot_heldout_runs_under_no_grad_on_cpu():
    context = build_parity_context(train_limit=1, held_out_limit=1, horizon_limit=1, init_limit=1)
    snapshot = objective_snapshot(
        context,
        beta=context.init_betas_cpu[0],
        horizon=context.horizons[0],
        initialization_id=0,
        update=0,
        device="cpu",
    )

    assert snapshot.heldout_grad_enabled is False
    assert torch.isfinite(snapshot.components.total)
    assert torch.isfinite(snapshot.heldout_components.total)
    assert snapshot.grad.shape == (4,)


def test_resolve_cuda_device_raises_when_unavailable(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    with pytest.raises(RuntimeError, match="CUDA was requested"):
        resolve_device("cuda")
