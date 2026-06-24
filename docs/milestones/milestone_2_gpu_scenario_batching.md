# Milestone 2 D.1b: GPU Scenario Batching Infrastructure

Status: Approved
Approval date: 2026-06-24

## Scope

This is the approved plan for Milestone 2 Phase D.1b only. Its responsibility
is to implement and validate deterministic full-dataset vectorization over the
scenario axis, compare it with the existing unbatched execution path on CPU and
CUDA, and report numerical equivalence, runtime, and memory evidence.

D.1b is a CPU/GPU infrastructure subphase. It is not the Milestone 2 direct
structured optimization experiment. It must complete its own implementation,
Phase E validation, and Phase F review before D.2 is planned or frozen.

The completed unbatched D.1a plan and evidence remain authoritative:

- `docs/milestones/milestone_2_cpu_gpu_infrastructure.md`;
- `reports/milestone2/infrastructure/parity.jsonl`;
- `reports/milestone2/infrastructure/runtime_summary.json`;
- `reports/milestone2/infrastructure/summary.md`;
- `reports/milestone2/infrastructure/phase_e_report.md`.

D.1b must preserve those artifacts and must not retroactively change D.1a.

## Fixed Decisions

- Use the designated `differential_sim` conda environment.
- Use `torch.float64` on CPU and CUDA.
- Use normalized controller inputs only.
- Use horizons `K=[1,3,6,10,T]` with `T=80`.
- Use all six approved controller initializations:
  `T_init=[0.9,1.2,1.4,1.6,1.9,2.2]`.
- Use the fixed 14 Milestone 1 training scenarios, in their approved order,
  with equal scenario weighting.
- Use the fixed 8 held-out scenarios only under `torch.no_grad()`.
- Batching means full-dataset vectorization over the scenario axis only.
- Keep batched and unbatched implementations as separate execution flows.
- Select execution explicitly through an API/CLI mode:
  `unbatched`, `scenario-batched`, or `compare`.
- Do not automatically select execution mode based on device.
- Correctness and optimizer-probe coverage use all five horizons and all six
  initializations.
- Performance coverage times every horizon and all six initializations.
- Benchmark timing uses `3` untimed warm-up repetitions followed by `10` timed
  repetitions for every measured `(device, execution mode, K,
  initialization)` training objective/backward case.
- Held-out no-grad timing uses `3` untimed warm-up repetitions followed by
  `10` timed repetitions for every measured case.
- Timing warm-up is measurement-only. It uses disposable tensors and optimizer
  state, performs no `optimizer.step()`, does not change beta, does not advance
  the optimizer probe, and is excluded from result rows.
- There is no predefined GPU speed threshold and no timing-based acceptance
  criterion.
- D.1b reports objective correctness and descriptive runtime/memory evidence.
  It does not select or recommend the D.2 device or execution policy.

## Batching Semantics

For a scenario split of size `S`:

- batched leader position and speed: `[S,T+1]`;
- recurrent follower position and speed: `[S]`;
- controller input: `[S,3]`;
- shared controller beta: `[4]`;
- per-step acceleration: `[S]`;
- stored follower position, speed, gap, and relative speed: `[S,T+1]`;
- stored acceleration: `[S,T]`.

The time loop remains explicit over `T=80`. Only the Python scenario loop is
vectorized.

Temporal truncation must preserve the existing unbatched semantics:

- `retain_start = T - K`;
- before `retain_start`, detach the complete recurrent scenario tensors;
- from `retain_start` onward, retain recurrent graph connectivity;
- batching may change only tensor organization and floating-point reduction
  order, not forward values, temporal graph meaning, or scenario weighting.

The objective must:

1. calculate total and progress/safety/jerk components per scenario;
2. preserve each scenario as one equally weighted observation;
3. average each component over the scenario axis;
4. produce one scalar full-dataset total for one backward pass;
5. preserve the existing objective weights and normalization.

## Explicit Non-Goals

- No stochastic mini-batching or scenario sampling.
- No resampling, curriculum, or online dataset generation.
- No batching across horizons.
- No batching across controller initializations.
- No averaging gradients across independent optimization runs.
- No simulator equation, integration-order, objective, scenario, normalization,
  controller, horizon, or initialization changes.
- No LR calibration.
- No final `N_updates` selection.
- No full Milestone 2 structured optimization experiment.
- No D.2 optimizer runner.
- No final CPU/CUDA evidence policy.
- No decision to admit batched execution into D.2.
- No decision about the execution mode of a future complete CPU duplicate.
- No speed threshold, performance pass/fail rule, or automatic execution-mode
  selection.
- No MLP training or Milestone 3 work.

## Files To Create Or Modify

Create:

- `src/differential_sim/batched_temporal_gradients.py`
  - construct batched scenario tensors;
  - implement batched controller rollout;
  - preserve temporal detachment semantics;
  - compute per-scenario and aggregate objective components;
  - expose typed batched result structures.
- `scripts/check_milestone2_gpu_batching.py`
  - dedicated D.1b entry point;
  - expose `--execution-mode unbatched|scenario-batched|compare`;
  - expose explicit device, probe, warm-up, repeat, smoke, and output options;
  - run correctness, optimizer-probe, timing, memory, and report generation.
- `tests/test_batched_temporal_gradients.py`
  - validate tensor shapes, forward values, objective reduction, gradients,
    detachment, held-out no-grad behavior, and CLI/result schema.

Modify only if needed:

- `src/differential_sim/device_parity.py`
  - reuse or minimally generalize metadata, tolerance, serialization, and
    cross-device comparison helpers;
  - do not place batched rollout or objective logic here.
- `src/differential_sim/temporal_gradients.py`
  - preferably unchanged;
  - a small helper extraction is permitted only when mechanical and covered by
    existing and new equivalence tests.

Do not create or modify D.2 files such as:

- `src/differential_sim/optimize.py`;
- `scripts/run_milestone2.py`;
- `tests/test_optimize.py`.

## Implementation Steps

1. Re-read `AGENTS.md`, `PROJECT_CONTEXT.md`, `PLANS.md`, the completed D.1a
   plan/report, and this approved D.1b plan.
2. Run the full existing test suite before editing batching behavior.
3. Implement the separate batched temporal-gradient path without modifying
   scientific semantics.
4. Add the explicit execution-mode API and D.1b CLI.
5. Add focused CPU tests before running CUDA checks.
6. Run a small smoke comparison to verify schema and control flow.
7. Run the complete D.1b correctness and optimizer-probe grid.
8. Run the complete CPU/CUDA timing and memory grid with the approved warm-up
   and repeat policy.
9. Write D.1b artifacts under the dedicated batching report directory.
10. Run the full test suite again.
11. Write the Phase E report and stop for Phase F review.

GPU-required commands must be escalated if sandboxing hides or blocks host GPU
access. A sandbox failure must not be treated as evidence that CUDA is
unavailable until the same command is rerun with approved escalation.

## Correctness Protocol

For all five horizons and six initializations, compare:

1. unbatched CPU versus scenario-batched CPU;
2. unbatched CUDA versus scenario-batched CUDA;
3. scenario-batched CPU versus scenario-batched CUDA.

At update `0`, compare:

- per-scenario follower position, speed, acceleration, gap, and relative speed;
- aggregate and per-scenario total objective;
- aggregate and per-scenario progress/safety/jerk components;
- beta gradient;
- minimum gap and speed;
- held-out aggregate and per-scenario components under `torch.no_grad()`;
- held-out gradient-enabled flag.

Run the infrastructure optimizer probe for every execution path:

- provisional `lr=0.03`;
- Adam `betas=(0.9,0.999)`;
- `eps=1e-8`;
- `weight_decay=0.0`;
- `amsgrad=False`;
- exactly `10` updates;
- no clipping, scheduler, early stopping, or recovery.

At update `10`, compare:

- final beta;
- training aggregate and per-scenario objective components;
- held-out aggregate and per-scenario objective components;
- improvement flags;
- finite/failure flags.

The provisional LR and 10-update probe are infrastructure checks only. They do
not select the D.2 LR or update budget.

## Numerical Tolerances

- Trajectory max absolute difference: `<=1e-8`.
- Scalar objective/component absolute difference: `<=1e-8`, or relative
  difference `<=1e-7`.
- Beta and gradient max absolute difference: `<=1e-7`, or relative difference
  `<=1e-6`.
- Improvement and failure flags must match.
- Held-out gradient-enabled flags must be `False`.

Per-scenario comparisons are required even when aggregate values pass. Failed
tolerances must be reported; they must not be weakened to complete D.1b.

## Performance Protocol

Measure these execution paths on both CPU and CUDA:

- unbatched training objective plus backward;
- scenario-batched training objective plus backward;
- unbatched held-out no-grad forward;
- scenario-batched held-out no-grad forward.

For every horizon and initialization:

1. run `3` untimed warm-up repetitions on disposable state;
2. run `10` timed repetitions;
3. synchronize CUDA immediately before and after each timed CUDA region;
4. alternate mode order or repeat in separate orders to limit first-mode and
   thermal bias;
5. exclude scenario construction and report serialization from core timing;
6. reset and record CUDA peak allocated memory by execution mode.

Report by device, mode, horizon, and initialization:

- sample count;
- median, mean, minimum, and maximum runtime;
- batched/unbatched runtime ratio;
- CUDA peak allocated memory;
- failures and non-finite values.

The report must present measurements objectively. No speed threshold or
performance acceptance decision is permitted in D.1b.

## Commands

Baseline and focused tests:

```bash
/home/zpz/miniconda3/envs/differential_sim/bin/python -m pytest -p no:cacheprovider
/home/zpz/miniconda3/envs/differential_sim/bin/python -m pytest -p no:cacheprovider tests/test_batched_temporal_gradients.py
```

Smoke:

```bash
/home/zpz/miniconda3/envs/differential_sim/bin/python scripts/check_milestone2_gpu_batching.py --execution-mode compare --smoke
```

Full D.1b run:

```bash
/home/zpz/miniconda3/envs/differential_sim/bin/python scripts/check_milestone2_gpu_batching.py --execution-mode compare --device cpu,cuda --probe-lr 0.03 --probe-updates 10 --warmup-repeats 3 --timed-repeats 10
```

The full command must be escalated when required for CUDA visibility.

## Result Artifacts

Write only under:

`reports/milestone2/infrastructure/batching/`

Required artifacts:

- `parity.jsonl`: machine-readable update-0 and update-10 comparisons;
- `timing.jsonl`: individual timed samples;
- `summary.json`: aggregate correctness, timing, memory, and metadata;
- `summary.md`: human-readable objective/parity and descriptive cost results;
- `phase_e_report.md`: commands, tests, acceptance status, deviations,
  assumptions, and remaining risks.

Every artifact must record:

- requested and actual device;
- execution mode;
- dtype;
- Python, PyTorch, CUDA, and `diffidm` versions;
- GPU name and compute capability where applicable;
- horizon, initialization ID, and `T_init`;
- scenario split and count;
- probe LR and update count where applicable;
- warm-up and timed-repeat counts for timing rows.

D.1a artifacts must not be overwritten or deleted.

## Acceptance Criteria

1. The full existing test suite passes before D.1b implementation.
2. New focused tests pass without weakening existing tolerances.
3. Batched and unbatched flows are separate and explicitly selectable.
4. Scenario batching uses exactly all 14 training or all 8 held-out scenarios
   with unchanged order and equal weights.
5. No stochastic mini-batching, scenario sampling, batched horizons, or batched
   initializations is introduced.
6. Update-0 correctness rows cover all five horizons and six initializations
   for all three approved comparisons.
7. The 10-update optimizer probe covers all five horizons, six initializations,
   CPU and CUDA, and both execution modes.
8. Per-scenario trajectories and aggregate/per-scenario objective components
   pass approved tolerances.
9. Beta gradients and final beta values pass approved tolerances.
10. Improvement and failure flags match.
11. Held-out evaluation uses `torch.no_grad()` in both execution modes.
12. Timing covers every horizon and initialization on CPU and CUDA, with
    `3` untimed warm-ups and `10` timed repetitions.
13. CUDA timing is synchronized and reports sample distributions and peak
    allocated memory.
14. D.1b artifacts are written under the batching subdirectory without
    overwriting D.1a evidence.
15. The report contains objective correctness and descriptive runtime/memory
    comparisons without a speed threshold.
16. No LR calibration, full D.2 optimization experiment, D.2 execution-policy
    selection, or MLP work is added.
17. The full test suite passes after implementation and validation.

## Stop Conditions

Stop and report before continuing if:

- CUDA is unavailable after an approved escalated check.
- Batched/unbatched equivalence fails an approved tolerance.
- Batched CPU/CUDA parity fails an approved tolerance.
- Scenario aggregation, temporal detachment, or held-out no-grad semantics
  cannot be preserved.
- Runtime or memory prevents completion of the approved D.1b grid.
- Implementation requires changing simulator equations, objective weights,
  scenario lists, normalization, controller parameterization, horizons, or
  initializations.
- Implementation requires stochastic mini-batching, scenario resampling,
  batched horizons, or batched initializations.
- Any change would begin D.2 or select its device/execution policy.

If a material deviation is required, return to Phase B and amend this plan
before implementing the deviation.

## Deferred Until D.1b Closure

- Whether scenario-batched execution is admitted into D.2.
- Final CPU-only versus CPU-plus-CUDA evidence policy.
- Execution mode for any complete CPU duplicate.
- Final `N_updates`.
- LR calibration execution.
- Full direct structured optimization implementation and experiments.
- Interpretation of whether Milestone 1 ranking predicts iterative
  optimization.

These items do not block D.1b implementation, but none may be decided silently
during D.1b.

## Plans Superseded

This plan freezes and supersedes only the D.1b batching portions of:

`temp_content/temp_plan/milestone_2_d1_cpu_gpu_infrastructure_phase_a_b_proposal.md`

It supplements, and does not supersede, the completed D.1a plan:

`docs/milestones/milestone_2_cpu_gpu_infrastructure.md`

It does not approve or freeze the Milestone 2 D.2 direct optimization plan.
