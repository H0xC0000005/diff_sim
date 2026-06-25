# Milestone 2 D.1: CPU/GPU Infrastructure And Parity

Status: Completed and closed as a passing Milestone 2 prerequisite
Approval date: 2026-06-23
Closure date: 2026-06-24

## Scope

This is the approved plan for Milestone 2 Phase D.1 only. Its responsibility is
to make CPU/GPU execution infrastructure explicit, verify parity on existing
rollout/objective/temporal-gradient utilities, measure runtime and GPU memory,
and report evidence needed before the Milestone 2 direct optimization
experiment plan is frozen.

D.1 is not the Milestone 2 direct structured optimization experiment. It must
not run LR calibration, full optimization experiments, or any MLP/Milestone 3
work.

The approved clarity rename has already been carried out and must be preserved:

- `src/differential_sim/fit.py` -> `src/differential_sim/milestone0_fit.py`;
- `src/differential_sim/diagnostics.py` ->
  `src/differential_sim/milestone1_diagnostics.py`;
- `src/differential_sim/gradient_modes.py` ->
  `src/differential_sim/temporal_gradients.py`.

## Fixed D.1 Decisions

- Use the designated `differential_sim` conda environment.
- Use `torch.float64` on CPU and CUDA.
- Use normalized controller inputs only.
- Use horizons `K=[1,3,6,10,T]` with `T=80`.
- Use the six approved initializations
  `T_init=[0.9,1.2,1.4,1.6,1.9,2.2]`.
- Use the fixed 14 Milestone 1 diagnostic/training scenarios and approved
  normalization statistics.
- Use the fixed 8 Milestone 1 held-out scenarios only for no-grad evaluation.
- Use a dedicated D.1 script:
  `scripts/check_cpu_cuda_parity.py`.
- Use a focused support module:
  `src/differential_sim/device_parity.py`.
- D.1 optimizer probe uses all five horizons, all six initializations,
  provisional `lr=0.03`, Adam `betas=(0.9,0.999)`, `eps=1e-8`,
  `weight_decay=0.0`, `amsgrad=False`, and exactly `10` updates per
  horizon/initialization/device.
- The provisional D.1 LR is not the selected Milestone 2 LR.
- D.1 produces evidence and recommendations only. It does not silently choose
  final `N_updates` or the final CPU/CUDA evidence policy for D.2.

## Explicit Non-Goals

- No LR calibration.
- No full Milestone 2 optimization experiment.
- No final `N_updates` selection.
- No final CPU/CUDA main-evidence policy selection.
- No batching implementation.
- No scenario, objective, simulator, normalization, controller, or horizon
  changes.
- No per-horizon tuning, clipping, scheduler, early stopping, or recovery
  policy.
- No MLP training or Milestone 3 work.

## Files To Create Or Modify

- Create `src/differential_sim/device_parity.py`.
  Responsibilities: device metadata capture, deterministic CPU/CUDA comparison
  helpers, tolerance checks, runtime measurement, CUDA peak-memory capture, and
  JSON-serializable rows. It must call existing simulator/objective utilities
  rather than duplicating simulator logic.
- Create `scripts/check_cpu_cuda_parity.py`.
  Responsibilities: command-line D.1 parity/cost probe, report writing, and
  explicit CPU/CUDA device handling.
- Create `tests/test_device_parity.py`.
  Responsibilities: metadata/schema checks, no-grad held-out behavior,
  tolerance helper behavior, deterministic CPU path behavior, and CUDA skip
  behavior when CUDA is unavailable.
- Modify imports only if needed to use the already-renamed modules.
- Do not create or modify `src/differential_sim/optimize.py` or
  `scripts/run_milestone2.py` in D.1.

## Implementation Steps

1. Re-read `AGENTS.md`, `PROJECT_CONTEXT.md`, `PLANS.md`, and this approved
   D.1 plan before coding.
2. Confirm the current rename state and run the full existing test suite before
   CPU/GPU probing.
3. Implement `device_parity.py` with typed, small functions for:
   device metadata, JSON-safe serialization, max-difference/tolerance checks,
   objective/gradient snapshots, held-out no-grad snapshots, optimizer probe
   rows, runtime timing, and CUDA peak-memory capture.
4. Implement `scripts/check_cpu_cuda_parity.py` with arguments for device pair,
   probe LR, probe updates, output directory, and optional small smoke mode if
   needed for tests.
5. Add focused tests for the D.1 utilities and script behavior. CUDA-dependent
   tests may skip only when CUDA is unavailable.
6. Run D.1 validation commands. GPU-required commands should be escalated when
   sandboxing would otherwise hide or block GPU access.
7. Write D.1 artifacts under `reports/milestone2/infrastructure/`.
8. Stop and report. Do not start D.2.

## D.1 Parity Protocol

For every horizon and six initializations, compare CPU and CUDA at update `0`
before any Adam step:

- forward trajectories;
- training objective and progress/safety/jerk components;
- beta vectors;
- gradient vectors;
- held-out objective/components under `torch.no_grad()`.

Then run the D.1 optimizer probe on CPU and CUDA:

- all horizons;
- all six initializations;
- provisional `lr=0.03`;
- approved Adam parameters;
- exactly `10` updates;
- fixed training and held-out datasets;
- no batching.

Compare CPU and CUDA at update `0` and update `10`:

- training objective/components;
- beta vectors;
- gradient vectors;
- held-out no-grad objective/components;
- improvement and failure flags;
- runtime per objective/backward step;
- CUDA peak memory.

## Numerical Tolerances

- Forward trajectory max absolute difference: `<=1e-8`.
- Scalar objective/component absolute difference `<=1e-8` or relative
  difference `<=1e-7`.
- Beta and gradient vector max absolute difference `<=1e-7` or relative
  difference `<=1e-6`.
- Improvement/failure flags must match in the optimizer probe.

If a tolerance fails, report the failure and do not use CUDA as main evidence
for D.2 without a plan amendment.

## Commands

Baseline and tests:

```bash
/home/zpz/miniconda3/envs/differential_sim/bin/python -m pytest -p no:cacheprovider
/home/zpz/miniconda3/envs/differential_sim/bin/python -m pytest -p no:cacheprovider tests/test_device_parity.py
```

GPU metadata and D.1 run:

```bash
nvidia-smi
/home/zpz/miniconda3/envs/differential_sim/bin/python scripts/check_cpu_cuda_parity.py --device cpu,cuda --probe-lr 0.03 --probe-updates 10
```

If `nvidia-smi` or CUDA execution fails due to Codex sandboxing, rerun the
command with escalation for user approval before drawing conclusions.

## Result Artifacts

Write under `reports/milestone2/infrastructure/`:

- `parity.jsonl`: machine-readable parity rows.
- `runtime_summary.json`: machine-readable runtime and memory summary.
- `summary.md`: human-readable D.1 report.

Minimum report content:

- environment and device metadata;
- commands run;
- test results;
- parity status by horizon and initialization;
- maximum observed CPU/CUDA differences by quantity;
- runtime per objective/backward step;
- CUDA peak memory;
- whether all D.1 tolerances passed;
- estimated cost and scientific tradeoff for `N_updates` values
  `100`, `200`, `300`, and `500`;
- recommendation for final `N_updates`;
- recommendation for D.2 CPU/CUDA evidence policy;
- explicit statement that recommendations are not frozen D.2 settings.

## Acceptance Criteria

1. Full existing test suite passes before D.1 parity probing.
2. New D.1 tests pass without weakening existing tolerances.
3. No old live imports of `differential_sim.fit`,
   `differential_sim.diagnostics`, or `differential_sim.gradient_modes` remain
   in `src`, `scripts`, or `tests`.
4. D.1 script records CPU and CUDA device metadata.
5. Held-out evaluation in D.1 runs under `torch.no_grad()`.
6. Update-0 CPU/CUDA parity rows are produced for all five horizons and six
   initializations.
7. The 10-update D.1 optimizer probe rows are produced for all five horizons
   and six initializations on both CPU and CUDA, unless CUDA is unavailable or
   parity fails first.
8. Numerical tolerances pass, or the failure is reported without weakening the
   check.
9. Runtime and CUDA peak-memory evidence are reported.
10. `summary.md` recommends, but does not freeze, final `N_updates` and D.2
    CPU/CUDA evidence policy.
11. No LR calibration, full optimization experiment, batching implementation, or
    D.2 code is added.

## Stop Conditions

Stop before continuing and report if:

- CUDA becomes unavailable in the designated environment after escalation checks.
- CPU/GPU parity fails any approved tolerance.
- Runtime or memory is impractical for the approved D.1 probe.
- Implementing D.1 requires changing simulator equations, objective weights,
  scenario lists, normalization, controller parameterization, horizons, or
  initializations.
- Implementing D.1 appears to require batching or scenario subsampling.
- Any change would start D.2 or alter Milestone 2 scientific meaning.

## Deferred To D.2 Planning

- Final `N_updates`.
- Final CPU-only versus CPU-plus-CUDA main-evidence policy.
- LR calibration execution.
- Full direct structured optimization experiment.
- Result interpretation of whether Milestone 1 ranking predicts iterative
  optimization.

## Plans Superseded

This approved D.1 plan supersedes only the D.1 portions of
`temp_content/temp_plan/milestone_2_d1_cpu_gpu_infrastructure_phase_a_b_proposal.md`.
It does not approve or freeze the D.2 direct optimization plan.

## Phase F Closure

This D.1a prerequisite is closed as passing within the closed Milestone 2.

Accepted evidence:

- CPU/CUDA parity passed for all approved horizons and initializations;
- maximum trajectory, objective, beta, and gradient differences remained within
  the approved tolerances;
- CUDA was valid scientifically but slower than CPU for the unbatched
  structured-controller workload;
- CUDA visibility from Codex requires escalation when sandboxing hides host GPU
  resources.

Later D.1b and D.2 decisions superseded the provisional D.1a runtime and
execution recommendations. The raw D.1a evidence remains authoritative for
device parity, not for Milestone 3 device selection.
