"""Run Milestone 2 D.1 CPU/CUDA infrastructure parity checks."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from differential_sim.device_parity import run_cpu_cuda_parity


DEFAULT_OUTPUT_DIR = ROOT / "reports" / "milestone2" / "infrastructure"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--device",
        default="cpu,cuda",
        help="Required D.1 device pair. Only 'cpu,cuda' is accepted for the full check.",
    )
    parser.add_argument("--probe-lr", type=float, default=0.03)
    parser.add_argument("--probe-updates", type=int, default=10)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run a tiny implementation smoke path for tests; not a D.1 evidence run.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.device != "cpu,cuda":
        raise SystemExit("D.1 parity requires --device cpu,cuda")
    if args.probe_lr <= 0.0:
        raise SystemExit("--probe-lr must be positive")
    if args.probe_updates < 1:
        raise SystemExit("--probe-updates must be >= 1")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable to PyTorch; rerun with escalation if sandboxing is suspected")

    limits = {}
    if args.smoke:
        limits = {"train_limit": 1, "held_out_limit": 1, "horizon_limit": 1, "init_limit": 1}

    run_cpu_cuda_parity(
        probe_lr=args.probe_lr,
        probe_updates=args.probe_updates,
        output_dir=args.output_dir,
        **limits,
    )


if __name__ == "__main__":
    main()
