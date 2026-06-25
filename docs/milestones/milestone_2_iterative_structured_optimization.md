# Milestone 2 D.2: Iterative Structured Optimization

Status: Closed as passing
Approval date: 2026-06-24
Closure date: 2026-06-24

## Scope

This is the approved plan for Milestone 2 Phase D.2 only. Its responsibility is
to implement and run iterative direct optimization of the bounded
four-parameter structured headway controller, compare the five approved
temporal-gradient horizons, and test whether the Milestone 1 one-step utility
ranking predicts iterative optimization performance.

D.2 uses the validated scenario-batched CPU path from D.1b. It does not reopen
CPU/GPU infrastructure work and does not begin Milestone 3.

Authoritative prerequisite plans and evidence:

- `docs/milestones/milestone_2_cpu_gpu_infrastructure.md`;
- `docs/milestones/milestone_2_gpu_scenario_batching.md`;
- `reports/milestone2/infrastructure/phase_e_report.md`;
- `reports/milestone2/infrastructure/batching/phase_e_report.md`.

## Research Question

> Does the Milestone 1 normalized one-step utility ranking of temporal gradient
> horizons predict iterative direct optimization performance for the bounded
> four-parameter structured controller when all horizons use one identical
> full-dataset Adam procedure?

Milestone 1 predictor ranking:

`K=80 > K=10 > K=6 > K=3 > K=1`

The primary D.2 outcome is held-out total-objective relative improvement from
update `0` to update `500`, aggregated over six fixed initializations.

## Fixed Experimental Decisions

- Use the designated `differential_sim` conda environment.
- Use CPU and `torch.float64`.
- Use deterministic scenario-batched full-dataset execution.
- Use normalized controller inputs only.
- Use horizons `K=[1,3,6,10,T]`, with `T=80`.
- Use controller initializations:
  `T_init=[0.9,1.2,1.4,1.6,1.9,2.2]`.
- Generate initial beta values with the approved Milestone 1 seeds `0..5` and
  noise scale `0.01`.
- Use the fixed 14 training scenarios in their approved order.
- Use the fixed 8 held-out scenarios only under `torch.no_grad()`.
- Give every training and held-out scenario equal weight within its split.
- Use one independent Adam optimizer for each `(K, initialization)` run.
- Use one globally selected LR for every horizon and initialization.
- Use exactly `500` updates for every nonfailed main run.
- Do not use early stopping or checkpoint selection.
- Log aggregate training metrics at update `0` and every update through `500`.
- Log per-scenario training details and held-out metrics at update `0`, every
  `10` updates, and update `500`.
- Keep failed/non-finite runs in aggregate denominators.
- Do not retry a failed run with changed settings.

## Optimizer Policy

Use:

```text
torch.optim.Adam(
    [beta],
    lr=selected_shared_lr,
    betas=(0.9,0.999),
    eps=1e-8,
    weight_decay=0.0,
    amsgrad=False,
)
```

There is:

- no scheduler;
- no gradient clipping;
- no regularization;
- no per-horizon LR or optimizer tuning;
- no recovery policy.

## Dataset And Tensor Flow

Training split:

- scenario count: `14`;
- leader position and speed: `[14,81]`;
- recurrent follower position and speed: `[14]`;
- controller inputs: `[14,3]`;
- shared beta: `[4]`;
- acceleration: `[14,80]`;
- other stored trajectory fields: `[14,81]`.

Held-out split:

- scenario count: `8`;
- corresponding leading dimensions use `8`.

Each optimizer update:

1. Select one independent `(K, initialization)` run.
2. Evaluate all 14 training scenarios simultaneously.
3. Compute total and progress/safety/jerk components per scenario.
4. Average each component equally over all 14 scenarios.
5. Backpropagate the aggregate total once.
6. Apply one Adam step to the shared beta.

This is deterministic full-batch optimization. No scenario subset is sampled
and no intermediate parameter update occurs between scenarios.

## Update Ordering And Failure Handling

For each main run:

1. Clone the approved initial beta to CPU `torch.float64`.
2. Set `requires_grad=True`.
3. Construct a fresh Adam optimizer.
4. Record update-0 training and held-out metrics.
5. For updates `1..500`:
   - call `optimizer.zero_grad(set_to_none=True)`;
   - calculate the scenario-batched training objective;
   - check finite loss;
   - call `loss.backward()`;
   - record gradient norm and finite status;
   - if loss or gradient is non-finite, mark the run failed and write a final
     failure row at that update;
   - otherwise call `optimizer.step()`;
   - record scheduled metrics.
6. Every nonfailed run completes exactly `500` optimizer steps.

Failed runs:

- remain in failure counts and denominators;
- are not restarted;
- receive no altered LR, clipping, horizon, objective, or update budget;
- have unavailable convergence/stability metrics reported as missing.

## Shared LR Calibration

Candidate grid:

`[0.003,0.01,0.03,0.1]`

For each candidate:

- use CPU and scenario-batched execution;
- run all five horizons and six initializations;
- use all 14 training scenarios;
- run exactly `40` Adam updates;
- do not evaluate held-out scenarios;
- use the main-run objective and failure policy.

Report:

- all 30 run outcomes;
- finite-run count;
- failure count and reasons;
- median and mean relative training-objective change;
- distribution by horizon and initialization;
- runtime.

A candidate is eligible only if:

- all 30 runs are finite;
- aggregate median relative training-objective change is negative.

Select the largest eligible LR, except that a near tie selects the smaller LR.

For eligible candidates with median changes `m_a` and `m_b`, let `m_best` be
the more negative median. A near tie means:

```text
abs(m_a - m_b) <= 0.02 * abs(m_best)
```

If no candidate is eligible, stop for a plan amendment. Held-out data must not
influence LR selection.

## Longer Batched/Unbatched Equivalence Smoke

Before LR calibration and full experiments:

- use all five horizons;
- use the initialization centered at `T_init=1.4`;
- use provisional `lr=0.03`;
- use identical Adam settings and initial state;
- run batched and unbatched paths for exactly `50` updates;
- compare at updates `0`, `10`, and `50`.

Compare:

- training total and components;
- held-out total and components under no-grad;
- beta;
- finite and failure flags;
- improvement flags.

Use the approved D.1 tolerances:

- scalar/component absolute difference `<=1e-8`, or relative difference
  `<=1e-7`;
- beta max absolute difference `<=1e-7`, or relative difference `<=1e-6`;
- flags must match.

The smoke is an engineering guard only. It must not tune LR, objective,
horizons, update budget, or evaluation policy.

## Logging And Result Schema

Aggregate training rows at every update include:

- milestone, stage, run ID, seed, initialization ID, and `T_init`;
- `K`, horizon label, and update;
- initial and current beta;
- optimizer, LR, and Adam parameters;
- total and weighted/unweighted progress/safety/jerk components;
- gradient norm and finite flags;
- minimum gap and speed;
- controller headway minimum and maximum;
- per-update runtime;
- failure flag and reason;
- requested/actual device, dtype, execution mode, Python, PyTorch, CUDA
  availability, and dependency metadata.

At update `0`, every `10` updates, and update `500`, also record:

- per-scenario training total and components;
- held-out aggregate and per-scenario total/components;
- held-out minimum gap and speed;
- held-out `grad_enabled=False`.

Held-out results must not affect calibration, stopping, checkpoint selection,
or policy changes.

## Primary And Secondary Analysis

For each `(K, initialization)`, primary relative improvement is:

```text
(heldout_total_update500 - heldout_total_update0)
-------------------------------------------------
max(abs(heldout_total_update0), epsilon)
```

More negative values indicate greater improvement.

Aggregate each horizon over the six initializations using:

- median;
- mean;
- minimum and maximum;
- interquartile range;
- failure count.

Rank horizons by median final held-out relative improvement.

Two horizons are operationally tied only when:

- their medians differ by at most `0.01` absolute relative-change units; and
- their six-run ranges overlap.

Report:

- Spearman rank correlation with the Milestone 1 ranking, descriptively;
- no p-value or significance claim;
- top-group overlap with Milestone 1 `{K=80,K=10,K=6}`;
- whether `K=80` remains best, is operationally tied, or is surpassed;
- whether H1 is supported, contradicted/weakened, or unresolved.

Secondary per-run metrics:

- final training relative improvement;
- normalized training-objective AUC over updates `0..500`;
- first update reaching `50%` of final achieved training improvement;
- first update reaching `90%` of final achieved training improvement;
- maximum positive training-objective rebound after first reaching `90%`;
- training-objective standard deviation over updates `451..500`;
- progress/safety/jerk component changes;
- final beta and headway range;
- runtime.

Aggregate secondary metrics by horizon. For failed runs, threshold-reaching and
stability metrics are missing, while failures remain in denominators. Do not
treat shortened curves as completed runs.

## Files To Create Or Modify

Create:

- `src/differential_sim/structured_optimization.py`
  - D.2 configuration and result dataclasses;
  - one-run Adam optimization;
  - shared LR calibration;
  - held-out no-grad evaluation;
  - failure handling;
  - convergence/stability metrics;
  - horizon aggregation and H1 ranking helpers.
- `scripts/run_milestone2_d2.py`
  - explicit `smoke`, `calibrate`, `full`, and `summarize` stages;
  - CPU/scenario-batched fixed defaults;
  - result writing and metadata capture.
- `tests/test_structured_optimization.py`
  - update ordering;
  - independent optimizer state;
  - shared-policy enforcement;
  - held-out no-grad isolation;
  - failure handling;
  - LR calibration selection;
  - convergence/stability metrics;
  - schema and deterministic smoke behavior.

Modify only if necessary:

- `src/differential_sim/batched_temporal_gradients.py`
  - small reusable helpers only;
  - no simulator or objective semantic changes.
- `src/differential_sim/device_parity.py`
  - reuse or mechanically generalize metadata/serialization helpers;
  - no CUDA experiment additions.

Do not add dependencies or modify environment files.

## Implementation Steps

1. Re-read `AGENTS.md`, `PROJECT_CONTEXT.md`, `PLANS.md`, this approved plan,
   and the D.1a/D.1b plans and reports.
2. Run the full existing test suite before D.2 edits.
3. Implement typed D.2 configuration/result structures.
4. Implement one scenario-batched CPU optimization run with the fixed update
   ordering, logging, held-out isolation, and failure policy.
5. Implement LR calibration and the approved global selection rule.
6. Implement convergence/stability aggregation and H1 ranking.
7. Implement the D.2 CLI stages and artifact writing.
8. Add focused tests.
9. Run the longer equivalence smoke and stop if it fails.
10. Run LR calibration and stop if no candidate is eligible.
11. Freeze the selected LR in result metadata for all main runs.
12. Run all 30 main optimization runs for 500 updates.
13. Summarize results and write the Phase E report.
14. Run the full test suite again.
15. Stop after Phase E reporting. Do not begin Phase F or Milestone 3.

## Commands

```bash
/home/zpz/miniconda3/envs/differential_sim/bin/python -m pytest -p no:cacheprovider
/home/zpz/miniconda3/envs/differential_sim/bin/python -m pytest -p no:cacheprovider tests/test_structured_optimization.py
/home/zpz/miniconda3/envs/differential_sim/bin/python scripts/run_milestone2_d2.py --stage smoke
/home/zpz/miniconda3/envs/differential_sim/bin/python scripts/run_milestone2_d2.py --stage calibrate
/home/zpz/miniconda3/envs/differential_sim/bin/python scripts/run_milestone2_d2.py --stage full
/home/zpz/miniconda3/envs/differential_sim/bin/python scripts/run_milestone2_d2.py --stage summarize
/home/zpz/miniconda3/envs/differential_sim/bin/python -m pytest -p no:cacheprovider
```

## Result Artifacts

Write under:

`reports/milestone2/structured_optimization/`

Required artifacts:

- `equivalence_smoke.jsonl`;
- `equivalence_smoke_summary.json`;
- `calibration.jsonl`;
- `calibration_summary.json`;
- `calibration_summary.md`;
- `optimization.jsonl`;
- `summary.json`;
- `summary.md`;
- `phase_e_report.md`.

Raw and human-readable outputs must remain together. Existing D.1a/D.1b
artifacts must not be overwritten.

## Acceptance Criteria

1. Existing and new tests pass without weakening Milestone 1 or D.1 checks.
2. D.2 uses CPU, `torch.float64`, and scenario-batched full-dataset execution.
3. Horizons and initializations exactly match the approved sets.
4. All modes share identical dataset, objective, normalization, Adam
   parameters, selected LR, update budget, evaluation cadence, and schema.
5. The 50-update batched/unbatched smoke covers all five horizons and passes
   approved tolerances.
6. LR calibration uses training data only and selects one global LR by the
   approved eligibility and near-tie rule.
7. Held-out evaluation always runs under `torch.no_grad()` and never affects
   calibration, stopping, checkpoint selection, or policy.
8. Every nonfailed main run completes exactly `500` updates.
9. Failed runs remain in denominators and are not retried with changed policy.
10. Aggregate training metrics are recorded at every update.
11. Per-scenario training and held-out details are recorded every 10 updates,
    including update `0` and update `500`.
12. Reports include total and progress/safety/jerk components.
13. Reports include per-initialization curves, final beta, headway range,
    failures, runtime, and environment/execution metadata.
14. Convergence speed and stability metrics follow the approved definitions.
15. The primary ranking uses update-500 held-out relative improvement and the
    approved operational-tie rule.
16. The report compares Milestone 1 and D.2 rankings and classifies H1.
17. Raw and human-readable artifacts are written under the D.2 report
    directory.
18. No CUDA full run, MLP work, or Milestone 3 work is added.
19. The full test suite passes after implementation and validation.

## Stop Conditions

Stop and report before continuing if:

- the longer batched/unbatched smoke exceeds approved tolerances;
- smoke failure or improvement flags differ;
- no LR candidate is eligible;
- pervasive non-finite losses or gradients occur;
- the fixed objective or scenario set appears invalid for iterative
  optimization;
- implementation requires changing simulator equations, objective weights,
  scenarios, normalization, controller parameterization, horizons, or
  initializations;
- per-horizon tuning, clipping, scheduling, early stopping, recovery, or
  stochastic mini-batching is proposed;
- CPU runtime materially exceeds D.1b estimates and motivates a device or
  execution-policy change;
- any change would begin MLP training or Milestone 3.

Individual non-finite main runs follow the fixed failure policy. A systemic
failure pattern requires stopping and reporting rather than changing settings.

Any material deviation requires returning to Phase B, obtaining approval, and
amending this plan before implementation continues.

## Explicit Non-Goals

- No SI-unit controller branch.
- No stochastic mini-batching, scenario sampling, resampling, or curriculum.
- No batching across horizons or initializations.
- No CUDA full experiment or CPU/CUDA D.2 comparison.
- No per-horizon tuning.
- No held-out-based LR selection, stopping, or checkpoint selection.
- No objective, simulator, scenario, normalization, or controller redesign.
- No additional initializations, repeated datasets, or new scenario families.
- No p-value or significance claim.
- No MLP or Milestone 3 work.

## Unresolved And Deferred

No D.2 implementation decision remains unresolved.

Deferred:

- Milestone 3 architecture, device timing, and experiment plan;
- any broader statistical generalization or additional datasets.

## Plans Superseded

This plan freezes and supersedes the D.2 portions of:

`temp_content/temp_plan/milestone_2_d2_iterative_optimization_phase_a_b_proposal.md`

It supplements, and does not supersede:

- `docs/milestones/milestone_2_cpu_gpu_infrastructure.md`;
- `docs/milestones/milestone_2_gpu_scenario_batching.md`.

It does not approve Milestone 3.

## Phase F Closure And Milestone 3 Handoff

Milestone 2 Phase F is closed as of 2026-06-24. Milestone 2 passes its approved
Stage 2 purpose.

### Accepted Primary Result

The Milestone 1 one-step ranking exactly predicts the Milestone 2 iterative
held-out ranking:

```text
Milestone 1: K=80 > K=10 > K=6 > K=3 > K=1
Milestone 2: K=80 > K=10 > K=6 > K=3 > K=1
Spearman rank correlation: 1.0
```

All 30 main runs were finite and completed 500 updates. No operational ties
were identified. Median update-500 held-out relative changes were:

```text
K=1   -0.08182
K=3   -0.12341
K=6   -0.19448
K=10  -0.22315
K=80  -0.42791
```

H1 is supported for the fixed deterministic dataset, normalized structured
controller, shared Adam policy, and approved update budget.

### Accepted Phase E.1 Infrastructure Observations

- Deterministic scenario batching preserves full-dataset optimization
  semantics and is not stochastic mini-batching.
- The 50-update D.2 batched/unbatched smoke passed all 15 comparisons near
  machine precision.
- CPU was the fastest measured path for the structured-controller workload,
  but that conclusion is not transferable to an MLP without new timing
  evidence.

### Accepted Phase E.2 Scientific Observations

- `K=80` is materially better than all truncated horizons; `K=10` is the best
  non-full horizon and the only truncated mode with held-out improvement in
  all six initializations.
- Within each horizon, all six initializations converge to essentially the same
  final controller, while different horizons converge to different stable
  controllers. The horizon effect is therefore persistent optimization bias,
  not incomplete convergence or initialization noise.
- The wide range of relative improvement for `K=80` reflects different
  starting objective values, not different final solutions.
- The horizon ranking is stable by update 20 and effectively converged by
  approximately update 200. The approved 500 updates are conservative and
  establish late-stage stability.
- Raw component changes are not directly comparable because components are
  normalized and weighted differently. Weighted contributions are not
  pathologically dominated by one term.
- Truncated modes obtain relatively more improvement from the local jerk term,
  while the full gradient obtains larger progress and safety improvements.
  This supports, but does not prove, a local-effect versus long-lag
  state-mediated credit explanation.
- The most supported mechanism is that detachment removes recurrent
  state-mediated temporal sensitivity. Adam and nonlinear objective geometry
  then converge to a stable horizon-specific attractor.
- More scenario samples can reduce sampling variance but do not generally
  remove structural truncation bias.
- Training and held-out objective changes can disagree. Held-out downstream
  evaluation is therefore essential and must not be replaced by training-loss
  evidence.

### Milestone 3 Planning Handoff

The following are evidence-backed defaults for Milestone 3 planning, not an
approved Milestone 3 plan:

- compare `K=80` with `K=10`;
- use normalized controller/model inputs and physically bounded headway output;
- preserve the same simulator, objective, scenario split, and held-out no-grad
  isolation unless a separately approved decision changes them;
- pair identical initial MLP weights across horizons and use multiple fixed
  model seeds;
- recalibrate one shared optimizer/LR policy for the MLP using training data
  only; do not reuse LR `0.03` automatically and do not tune per horizon;
- select a fixed update budget from MLP-specific convergence and timing
  evidence rather than inheriting 500 updates automatically;
- remeasure CPU/CUDA cost for the fixed MLP architecture; if CUDA is used as
  main evidence, complete approved parity checks first;
- keep total, progress, safety, jerk, minimum-gap, minimum-speed, headway-range,
  convergence, failure, and held-out metrics;
- treat gradient cosine, gradient norm, and update norm at a few predeclared
  checkpoints as optional reporting diagnostics, not optimizer inputs;
- do not assume that more sampled scenarios eliminate truncation bias.

Open Milestone 3 decisions still require Phase A/B discussion:

- exact MLP architecture and width;
- number of paired model seeds;
- shared optimizer family and LR calibration grid;
- update budget and logging cadence;
- CPU/GPU execution policy and any batching scope;
- whether optional gradient-alignment diagnostics are worth their cost.

Authoritative closure and handoff index:

`reports/milestone2/closure_and_milestone3_handoff.md`
