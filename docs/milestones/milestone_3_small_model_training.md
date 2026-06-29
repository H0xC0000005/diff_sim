# Milestone 3: Small-Model Training

Status: Approved

Approval date: 2026-06-25

## Authority And Scope

This is the authoritative plan for Milestone 3 implementation and validation.
It freezes the accepted Phase A/B proposal for training a small bounded MLP
through the deterministic differentiable IDM simulator.

Milestone 3 tests whether the temporal-gradient utility observed for the
Milestone 2 structured controller transfers to a nonlinear state-dependent
model. It compares four temporal horizons while preserving the existing
forward simulator, objective, normalization, scenario split, and held-out
evaluation procedure.

Implementation must occur in a separate Phase D task. This Phase C task does
not authorize implementation or experiments.

## Research Question

> When the four-parameter structured controller is replaced by a small
> state-dependent MLP, how does held-out downstream utility change across
> temporal horizons `K=[6,10,20,35,50,80]` under paired initialization and one shared
> training policy, and does full temporal credit assignment remain useful and
> competitive with the best truncated horizon?

This is the Milestone 3 test of H2. The result may support, contradict, or leave
H2 unresolved. No outcome is assumed.

## Fixed Scientific Semantics

Milestone 3 must preserve:

- deterministic one-lane, one-leader/one-follower simulation;
- fixed vehicle ordering and leader identity;
- rollout length `T=80` and step size `dt=0.2 s`;
- initial follower position `-22 m` and speed `16 m/s`;
- leader length `5 m`;
- existing smooth-clamped `diffidm==0.0.3` acceleration semantics;
- existing speed-then-position update ordering;
- fixed IDM parameters other than the model-selected desired headway;
- normalized model inputs derived from training scenarios only;
- time-headway bounds `(0.5,3.0)` seconds;
- the existing 14 training and 8 held-out scenarios in their approved order;
- equal weighting of scenarios within each split;
- the existing objective definitions, scales, and weights:

  ```text
  progress_weight = 1.0
  safety_weight = 0.7
  jerk_weight = 5.0
  ```

Default-objective results remain the main Milestone 3 evidence unless a later
closure decision says otherwise.

- held-out evaluation under `torch.no_grad()`;
- `torch.float64` for training, numerical validation, and main evidence;
- temporal truncation that changes only recurrent backward connectivity.

Every horizon must produce identical forward values at identical model
parameters. Held-out data must not affect model initialization, normalization,
LR selection, update-budget selection, stopping, or checkpoint selection.

## Approved Horizons

Compare exactly:

```text
K=[6,10,20,35,50,80]
```

`K=80` is full backpropagation through the rollout. `K=6`, `K=10`, `K=20`,
`K=35`, and `K=50` are truncated temporal gradients using the existing detach
semantics:

```text
retain_start = T - K
```

Before `retain_start`, detach the complete recurrent follower state passed to
later transitions. Preserve local same-step gradients and all forward values.

`K=20`, `K=35`, and `K=50` are included by approved amendment to make the
truncated-to-full response curve less sparse and identify where performance
begins to rise from the initial plateau. This does not reopen or modify the
closed Milestone 2 result.

## Explicit Non-Goals

- No horizon outside `[6,10,20,35,50,80]`.
- No architecture, width, or activation search.
- No per-horizon optimizer, LR, budget, scheduler, clipping, regularization,
  early stopping, recovery, or checkpoint selection.
- No stochastic mini-batching, scenario resampling, curriculum, or online data
  generation.
- No batching across model seeds or horizons.
- No change to scenarios, normalization, simulator equations, integration
  order, headway bounds, or held-out split.
- No objective change except the approved separate Phase E sensitivity group
  defined below.
- No SI-unit input branch.
- No reconstruction, imitation, or trajectory-prediction objective.
- No additional simulator, RL, policy gradient, SUMO, lane changing, route
  choice, traffic signals, platoons, or larger network.
- No p-values or inferential significance claims from six seeds.
- No CUDA main experiment or CPU/CUDA duplicate experiment.
- No gradient-field analysis beyond the approved shared-state diagnostic.
- No Milestone 4 or later work.

## MLP Architecture

Use one hidden layer:

```text
normalized input [3]
  -> Linear(3,16)
  -> tanh
  -> Linear(16,1)
  -> bounded sigmoid headway
```

For a scenario batch size `S`:

```text
raw input:            [S,3],  float64
normalized input:     [S,3],  float64
hidden preactivation: [S,16], float64
hidden activation:    [S,16], float64
output logit:         [S],    float64
time headway:         [S],    float64, seconds
```

The output is:

```text
T_t = 0.5 + 2.5 * sigmoid(logit_t)
```

The model therefore has 81 trainable parameters:

```text
Linear(3,16):  3*16 + 16 = 64
Linear(16,1): 16*1 + 1  = 17
total:                     81
```

No additional clipping or safety branch may be added to the model output.

## Initialization And Seed Protocol

Use model seeds:

```text
[1000,1001,1002,1003,1004,1005]
```

For each seed:

1. Enter a controlled `torch.random.fork_rng` context.
2. Call `torch.manual_seed(model_seed)`.
3. Construct the MLP on CPU in `torch.float64`.
4. Use PyTorch 2.5.1 default `nn.Linear.reset_parameters()` initialization:
   Kaiming-uniform weights with `a=sqrt(5)` and fan-in-based uniform biases.
5. Serialize the ordered parameter names, shapes, flattened values, and a
   deterministic SHA-256 hash of the CPU float64 state.
6. Create four independent model instances and load the exact same state into
   `K=[6,10,20,35,50,80]`.
7. Construct one fresh optimizer for each `(seed,K)` run.

Model objects, optimizer state, and gradient buffers must not be shared across
horizons. CUDA timing must copy the canonical CPU-created state rather than
initializing independently on CUDA.

## Dataset And Tensor Flow

Training uses all 14 approved scenarios simultaneously:

```text
leader position/speed: [14,81]
recurrent position:    [14]
recurrent speed:       [14]
model input:           [14,3]
headway output:        [14]
acceleration:          [14,80]
other trajectory data: [14,81]
```

Held-out evaluation uses corresponding leading dimension `8`.

At every training update:

1. Evaluate all 14 training scenarios in one deterministic scenario-batched
   rollout.
2. Compute total, progress, safety, and jerk per scenario.
3. Average each component equally over scenarios.
4. Backpropagate the aggregate total once.
5. Apply exactly one optimizer update.

Scenario batching is full-dataset vectorization, not stochastic mini-batch SGD.
No parameter update may occur between scenarios.

## Optimizer Policy

Use one shared Adam policy:

```text
torch.optim.Adam(
    model.parameters(),
    lr=selected_shared_lr,
    betas=(0.9,0.999),
    eps=1e-8,
    weight_decay=0.0,
    amsgrad=False,
)
```

There is no scheduler, clipping, regularization, early stopping, recovery
policy, or best-checkpoint selection.

## Training-Only LR Calibration

Candidate grid:

```text
[1e-4,3e-4,1e-3,3e-3,1e-2]
```

For every candidate:

- CPU, scenario-batched, `torch.float64`;
- all four horizons and all six model seeds;
- fresh canonical initial states and optimizer states;
- exactly 100 updates;
- all 14 training scenarios;
- no held-out evaluation.

For each horizon, compute the median 100-update relative training-objective
change over seeds. A candidate is eligible only if:

- all 24 runs are finite;
- every run completes 100 updates;
- the median relative training change is negative for every horizon.

Define:

```text
score(lr) = max(
    median_change_K6,
    median_change_K10,
    median_change_K35,
    median_change_K80
)
```

More negative is better. Select the eligible LR with the most negative score.
If candidate scores differ by at most `2%` of the absolute best score, select
the smaller LR. If no candidate is eligible, stop for a plan amendment.

Held-out data must not be loaded or evaluated during LR calibration.

## Training-Only Update-Budget Calibration

After LR selection, run fresh training-only pilots for all 24 `(seed,K)` runs
through update 1200.

Candidate budgets:

```text
[200,400,600,800,1000,1200]
```

For each seed and horizon, record:

- relative training change at every candidate budget;
- remaining improvement from that candidate to update 1200;
- objective standard deviation over the preceding 100 updates;
- mean absolute relative per-update objective change over the preceding 100
  updates.

A candidate budget qualifies only when, for all four horizons:

- all six runs are finite through that budget;
- median remaining improvement to update 1200 is at most `2%` of median total
  improvement achieved by update 1200;
- at least five of six seeds have remaining improvement at most `5%` of their
  own total achieved improvement;
- median mean absolute relative per-update change over the preceding 100
  updates is `<=1e-4`.

Select the smallest qualifying budget.

Update 1200 is not accepted automatically. If its final-100-update stability
criterion fails for any horizon, stop for a plan amendment. Held-out data must
not be loaded or evaluated during budget selection.

The main experiment must restart from the canonical initial weights with fresh
optimizer state; calibration trajectories are not main evidence.

## Main-Run Update Ordering

For every `(seed,K)` main run:

1. Load the canonical initial state.
2. Construct a fresh Adam optimizer using the selected shared LR.
3. Record update-0 training metrics.
4. Evaluate update-0 held-out metrics under `torch.no_grad()`.
5. For updates `1..N`:
   - call `optimizer.zero_grad(set_to_none=True)`;
   - evaluate all training scenarios;
   - check the aggregate total for finiteness;
   - call backward once;
   - check every parameter gradient for finiteness;
   - record the global gradient norm;
   - call `optimizer.step()`;
   - record global parameter-update and parameter norms;
   - record scheduled training and held-out metrics.
6. Every nonfailed run completes exactly `N` optimizer steps.
7. The fixed update-`N` held-out result is primary.

No checkpoint may be selected using training or held-out curves.

## Logging And Evaluation Cadence

At every update, record aggregate:

- total and progress/safety/jerk components;
- weighted component contributions;
- gradient norm;
- parameter-update norm;
- parameter norm;
- minimum gap and speed;
- model headway minimum and maximum;
- finite/failure flags;
- step and cumulative runtime.

At update 0, every 30 updates, and the final update, additionally record:

- per-scenario training total and components;
- held-out aggregate and per-scenario total/components;
- held-out minimum gap and speed;
- held-out headway minimum and maximum;
- `heldout_grad_enabled=False`.

The final update must be logged even when it is not divisible by 30.

Save initial and final model state dictionaries separately from JSONL metric
rows. The cadence must be identical across horizons.

## Failure Handling

- A non-finite loss or any non-finite parameter gradient marks the run failed
  at that update.
- Failed runs are not restarted.
- Failed runs receive no altered LR, clipping, budget, horizon, or recovery.
- Failed runs remain in failure counts and aggregate denominators.
- Unavailable final or convergence metrics are explicit missing values.
- More than one failed main run per horizon, or another systemic failure
  pattern, triggers a stop for plan amendment.

## Execution Architecture And Device Policy

Use deterministic scenario-batched CPU `torch.float64` for:

- execution checks;
- LR calibration;
- budget calibration;
- gradient-field diagnostics;
- main evidence.

Do not batch model seeds or horizons.

Before calibration:

1. Time scenario-batched CPU for all four horizons using 3 untimed warm-ups and
   10 timed objective/backward repetitions.
2. Briefly time scenario-batched CUDA for `K=10` and `K=80` using the same
   warm-up/repeat policy. CUDA timing is descriptive only.
3. Compare CPU scenario-batched and CPU unbatched execution for two seeds and
   all four horizons at update 0 and after a 10-update Adam probe.

The CPU equivalence comparison must cover:

- trajectories;
- aggregate and per-scenario objective/components;
- model outputs and headway ranges;
- flattened parameter gradients;
- final parameters;
- minimum gap and speed;
- held-out no-grad results;
- improvement and failure flags.

Approved tolerances:

```text
trajectory max absolute difference:        <=1e-8
scalar/component:                          abs<=1e-8 or rel<=1e-7
parameter/gradient vector:                 abs<=1e-7 or rel<=1e-6
headway/min-gap/min-speed absolute:         <=1e-8
improvement/failure flags:                 exact match
held-out grad-enabled:                     False
```

CUDA is not a main evidence path and does not require a CPU/CUDA optimizer
parity grid. Proposing CUDA main evidence requires a plan amendment.

## Gradient-Field Diagnostics

The diagnostics are reporting-only and must never alter optimization.

Record every update:

- global gradient L2 norm;
- parameter-update L2 norm;
- parameter L2 norm.

At update 0, evaluate all pairwise cosine similarities among
`K=[6,10,20,35,50,80]` at the canonical shared initial state for every seed.

For repeated diagnostics, use the full-gradient reference trajectory. Let `N`
be the selected shared update budget. For each seed, save the `K=80` state at:

```text
[0,N/10,N/4,N/2,3N/4,N]
```

All budget candidates are divisible by 20, so these updates are integral.

At every saved state:

1. Evaluate the training-objective gradient for every horizon at the identical
   parameter vector.
2. Perform no optimizer step.
3. Do not mutate the live model or optimizer.
4. Report:
   - all four gradient norms;
   - the symmetric `4 x 4` pairwise cosine matrix;
   - undefined cosine explicitly when either norm is zero;
   - seed, reference update, reference-state hash, horizon pair, and runtime.

This diagnostic measures field divergence along the full-gradient optimization
path. It is not a field analysis along every horizon trajectory and does not
affect H2 classification.

## Primary And Secondary Analysis

For seed `s` and horizon `K`, define final held-out relative change:

```text
r_s,K =
  (heldout_total_final - heldout_total_update0)
  / max(abs(heldout_total_update0),1e-12)
```

More negative is better.

For each truncated horizon:

```text
d_s,K = r_s,80 - r_s,K
```

Negative favors the full gradient.

Report:

- all per-seed results;
- median, mean, minimum, maximum, and IQR by horizon;
- median and mean paired differences;
- paired win/tie counts;
- descriptive horizon ranking and operational ties;
- failure counts.

Use `0.01` absolute relative-change units as the practical-equivalence
threshold. Material pairwise advantage also requires at least four of six
paired-seed wins.

### H2 Classification

Rank all four horizons by median final held-out relative change. Define the
best truncated horizon as the member of `{6,10,20,35,50}` with the most negative
median. Held-out data is used here only for final scientific classification.

Classify H2 as supported when:

- at most one run fails per horizon;
- `K=80` has negative median held-out change; and
- `K=80` is either:
  - materially better than the best truncated horizon, with median paired
    difference `<=-0.01` and at least four full-gradient wins; or
  - operationally tied with the best truncated horizon, with absolute median
    paired difference `<=0.01`, overlapping ranges, and neither horizon having
    more than four paired wins.

Classify H2 as contradicted when finite convergence evidence is sufficient and:

- the best truncated horizon materially beats `K=80`, with median paired
  difference `>=0.01` and at least four truncated wins; or
- `K=80` has non-improving median held-out change while the best truncated
  horizon improves; or
- all horizons are finite and converged but none improves held-out median.

Classify H2 as unresolved otherwise, including excessive failures, material
paired-seed disagreement, no qualifying budget, execution-equivalence failure,
or held-out-isolation failure.

The complete four-horizon curve is secondary evidence. No monotonic
horizon-response claim is assumed.

Secondary per-run metrics:

- final training relative improvement;
- normalized training-objective AUC;
- first update reaching 50% and 90% of final achieved training improvement;
- maximum positive rebound after first reaching 90%;
- objective standard deviation over the final 10% of updates;
- mean absolute relative per-update change over the final 100 updates;
- progress/safety/jerk changes and weighted contributions;
- final headway range;
- runtime and failure details.

Raw unweighted objective components must not be compared as if their scales or
weights were equal.

## Files To Create

- `configs/milestone3_small_model.yaml`
  - authoritative executable settings for architecture, seeds, horizons,
    optimizer grid, budget candidates, cadence, diagnostic checkpoints,
    dtype/device, and tolerances.
- `src/differential_sim/small_model_training.py`
  - MLP construction and state hashing;
  - four-way paired initialization;
  - training and evaluation;
  - LR and budget selection;
  - failure handling;
  - convergence aggregation;
  - gradient-field diagnostics;
  - H2 classification.
- `scripts/run_milestone3.py`
  - explicit stages:
    `smoke`, `execution-check`, `calibrate-lr`, `calibrate-budget`, `full`,
    and `summarize`;
  - artifact and metadata writing.
- `tests/test_small_model_training.py`
  - architecture, pairing, determinism, update ordering, shared policy,
    selection rules, held-out isolation, failure handling, execution
    equivalence, result schema, gradient diagnostics, and H2 classification.

Generated artifacts go under:

`reports/milestone3/small_model_training/`

The approved Phase E objective-weight sensitivity group writes separate
artifacts under:

`reports/milestone3/small_model_training_weight_sensitivity_aggressive/`

It must not modify or replace the default-objective artifact folder.

## Phase E Objective-Weight Sensitivity Amendment

Status: Approved by user feedback in Phase E.

Run one separate full Milestone 3 experiment group that changes only objective
weights:

```text
progress_weight = 1.2
safety_weight = 0.4
jerk_weight = 15.0
```

All other scientific and execution semantics remain fixed:

- same simulator, scenarios, held-out isolation, normalization, architecture,
  initialization protocol, model seeds, horizons, batching mode, dtype, device
  policy, logging cadence, diagnostics, and H2 classification rule;
- same LR candidate grid and update-budget candidate grid;
- fresh LR calibration and fresh update-budget calibration under the new
  objective weights;
- fresh main runs from the canonical paired initial MLP states;
- separate result folder and metadata identifying the objective profile;
- no replacement or reinterpretation of the existing default-objective run.

This sensitivity group is not a search over objective weights. It tests whether
the horizon-response curve and H2 classification depend on a deliberately
aggressive change in progress/safety/jerk tradeoff. It may support, challenge,
or leave unresolved the robustness of the default-objective conclusion, but it
does not by itself choose a better objective.

Commands for the sensitivity group:

```bash
/home/zpz/miniconda3/envs/differential_sim/bin/python scripts/run_milestone3.py --objective-profile aggressive_progress_comfort --output-dir reports/milestone3/small_model_training_weight_sensitivity_aggressive --stage smoke
nvidia-smi
/home/zpz/miniconda3/envs/differential_sim/bin/python scripts/run_milestone3.py --objective-profile aggressive_progress_comfort --output-dir reports/milestone3/small_model_training_weight_sensitivity_aggressive --stage execution-check
/home/zpz/miniconda3/envs/differential_sim/bin/python scripts/run_milestone3.py --objective-profile aggressive_progress_comfort --output-dir reports/milestone3/small_model_training_weight_sensitivity_aggressive --stage calibrate-lr
/home/zpz/miniconda3/envs/differential_sim/bin/python scripts/run_milestone3.py --objective-profile aggressive_progress_comfort --output-dir reports/milestone3/small_model_training_weight_sensitivity_aggressive --stage calibrate-budget
/home/zpz/miniconda3/envs/differential_sim/bin/python scripts/run_milestone3.py --objective-profile aggressive_progress_comfort --output-dir reports/milestone3/small_model_training_weight_sensitivity_aggressive --stage full
/home/zpz/miniconda3/envs/differential_sim/bin/python scripts/run_milestone3.py --objective-profile aggressive_progress_comfort --output-dir reports/milestone3/small_model_training_weight_sensitivity_aggressive --stage summarize
```

## Files That May Be Modified Narrowly

- `src/differential_sim/controllers.py`
  - add the bounded MLP and, if needed, a generic controller protocol;
  - preserve structured-controller behavior.
- `src/differential_sim/temporal_gradients.py`
  - controller typing or protocol generalization only;
  - no detach or forward semantic change.
- `src/differential_sim/batched_temporal_gradients.py`
  - controller typing or protocol generalization only;
  - no batching, objective, reduction, or detach semantic change.
- `src/differential_sim/structured_optimization.py`
  - preferably unchanged;
  - helper extraction only when mechanical and covered by Milestone 2
    regression tests.

Do not modify dependency or environment files.

## Implementation Steps

1. Re-read `AGENTS.md`, `PROJECT_CONTEXT.md`, `PLANS.md`, this approved plan,
   and the closed Milestone 2 plans and handoff.
2. Run the full existing test suite before editing.
3. Add the fixed MLP and paired initialization/state-hash utilities.
4. Generalize controller typing only as required to use existing batched and
   unbatched rollouts.
5. Implement the Milestone 3 configuration, one-run training, no-grad
   evaluation, logging, failure handling, and aggregation.
6. Implement LR calibration using the approved training-only rule.
7. Implement update-budget calibration using the approved training-only rule.
8. Implement the full-gradient-reference gradient-field diagnostic.
9. Implement the CLI stages and artifact writing.
10. Add focused tests.
11. Run the CPU execution-equivalence check and stop if it fails.
12. Run brief CPU/CUDA timing. Do not promote CUDA to main evidence.
13. Run LR calibration and stop if no candidate is eligible.
14. Run budget calibration and stop if no candidate qualifies.
15. Freeze the selected LR and budget in result metadata.
16. Run all 24 main `(seed,K)` experiments.
17. Run the gradient-field diagnostic using saved `K=80` states.
18. Summarize and classify H2 using the approved rule.
19. Run the full test suite.
20. Write the Phase E report and stop for user review.

Do not start Phase F closure or another milestone automatically.

## Required Tests

1. The MLP has one width-16 `tanh` hidden layer and exactly 81 parameters.
2. Output shapes are correct and headways remain within physical bounds.
3. Normalization equals the approved training-only statistics and is
   independent of held-out scenarios, seed, and horizon.
4. Same-seed initialization states and hashes are identical; different seeds
   differ.
5. All four horizon runs for a seed load identical initial states.
6. Optimizer state is independent across runs.
7. Forward trajectories, outputs, and objectives are identical across horizons
   at identical parameters.
8. The full MLP gradient passes a central finite-difference directional check
   on a small deterministic float64 case. Accept relative error `<1e-3` for a
   stable epsilon or absolute error `<1e-6`.
9. Detachment tests confirm only intended recurrent future dependencies are
   cut and local same-step model gradients remain.
10. Scenario-batched and unbatched trajectories, per-scenario components,
    aggregate objectives, and flattened gradients pass approved tolerances.
11. Held-out evaluation is no-grad, does not populate gradients, and is absent
    from LR and budget selection inputs.
12. Repeated CPU smoke runs are deterministic.
13. Shared-policy validation rejects horizon-specific settings.
14. Non-finite runs are retained and not retried.
15. The CPU update-0 and 10-update execution-equivalence probe passes.
16. Shared-state diagnostics evaluate every horizon at identical parameter
    vectors and do not mutate model or optimizer state.
17. H2 classification follows the approved full-versus-best-truncated rule.
18. The full Milestone 0–2 regression suite remains passing.

## Commands

```bash
/home/zpz/miniconda3/envs/differential_sim/bin/python -m pytest -p no:cacheprovider
/home/zpz/miniconda3/envs/differential_sim/bin/python -m pytest -p no:cacheprovider tests/test_small_model_training.py
/home/zpz/miniconda3/envs/differential_sim/bin/python scripts/run_milestone3.py --stage smoke
nvidia-smi
/home/zpz/miniconda3/envs/differential_sim/bin/python scripts/run_milestone3.py --stage execution-check
/home/zpz/miniconda3/envs/differential_sim/bin/python scripts/run_milestone3.py --stage calibrate-lr
/home/zpz/miniconda3/envs/differential_sim/bin/python scripts/run_milestone3.py --stage calibrate-budget
/home/zpz/miniconda3/envs/differential_sim/bin/python scripts/run_milestone3.py --stage full
/home/zpz/miniconda3/envs/differential_sim/bin/python scripts/run_milestone3.py --stage summarize
/home/zpz/miniconda3/envs/differential_sim/bin/python -m pytest -p no:cacheprovider
```

CUDA timing commands require escalation when sandboxing hides the host GPU.

## Result Artifacts

Machine-readable:

- `initializations.json`;
- `equivalence.jsonl`;
- `timing.jsonl`;
- `gradient_field_diagnostics.jsonl`;
- `lr_calibration.jsonl`;
- `lr_calibration_summary.json`;
- `budget_calibration.jsonl`;
- `budget_calibration_summary.json`;
- `training.jsonl`;
- `summary.json`;
- initial and final state files under `models/`.

Human-readable:

- `lr_calibration_summary.md`;
- `budget_calibration_summary.md`;
- `summary.md`;
- `observations.md`;
- `phase_e_report.md`.

Artifacts must record the Git commit, configuration hash, requested and actual
device, dtype, execution mode, Python, PyTorch, CUDA availability, GPU metadata,
`diffidm` version, architecture, parameter count, model seed, initialization
hash, horizon, optimizer policy, selected LR, selected budget, scenario counts,
and failure state.

## Acceptance Criteria

Milestone 3 passes Phase E only if:

1. Architecture, seeds, horizons, optimizer policy, budget, and cadence match
   this plan and the saved configuration.
2. Every six-horizon seed group starts from identical hashed weights.
3. All horizons share optimizer family and parameters, LR, budget, scenarios,
   objective, dtype, device, execution mode, and evaluation cadence.
4. Normalization is training-only and fixed.
5. Forward values are identical across horizons at identical parameters.
6. Full-gradient finite-difference and detachment tests pass.
7. Scenario-batched/unbatched MLP equivalence passes.
8. Held-out evaluation is always no-grad and never affects calibration,
   stopping, budget selection, or checkpoint selection.
9. CPU execution equivalence passes before calibration.
10. CUDA remains timing-only.
11. Every nonfailed main run completes the selected budget.
12. Failed runs remain in denominators and receive no altered policy.
13. Reports retain total, progress, safety, jerk, weighted contributions,
    minimum gap/speed, headway range, convergence, stability, failures, runtime,
    and held-out metrics.
14. Raw and human-readable artifacts are complete and reproducible.
15. The full regression suite passes without weakening earlier tolerances.
16. H2 is classified as supported, contradicted, or unresolved using the
    approved rule.
17. Gradient-field diagnostics follow the approved shared-state protocol and
    do not alter optimization.

## Stop Conditions

Stop before making a conflicting change and request a plan amendment if:

- the MLP requires changing forward simulator semantics;
- forward values differ by horizon;
- the full finite-difference or detachment checks fail after implementation
  errors are ruled out;
- batching changes per-scenario weighting or gradient semantics;
- CPU execution equivalence fails;
- no LR candidate is eligible;
- no update budget qualifies, including failure of the update-1200 stability
  requirement;
- more than one main run fails per horizon or failures are systemic;
- held-out data appears necessary for LR, budget, stopping, or checkpoint
  selection;
- CUDA is proposed as main evidence;
- architecture, width, activation, seeds, objective beyond the approved Phase E
  sensitivity profile, scenarios, normalization, optimizer family, candidate
  grids, cadence, diagnostic protocol, or classification rule must change;
- per-horizon tuning, clipping, scheduling, early stopping, recovery,
  stochastic batching, or seed batching is proposed;
- Python/environment drift causes regression or numerical failure;
- implementation requires broader gradient-field analysis, a larger model, or
  a later milestone.

Small implementation details that do not affect scientific meaning may be
resolved conservatively and documented in the Phase E report.

## Unresolved Items

No implementation-blocking scientific or engineering choice remains open in
this approved plan.

Any new evidence requiring a material decision must trigger the stop-and-amend
procedure before implementation continues.

## Deferred Items

- Width or activation sensitivity.
- Additional model seeds or a second dataset.
- Batching across model seeds or horizons.
- Gradient-field diagnostics beyond the approved `K=80` reference trace.
- Horizons beyond `[6,10,20,35,50,80]`.
- Objective-weight sensitivity beyond the approved Phase E group, or any
  normalization sensitivity.
- Larger models or platoons.
- Other simulators, policy gradients, RL, SUMO, or hard-simulator transfer.
- Statistical generalization beyond this fixed deterministic experiment.

## Assumptions And Risks

- The existing training-only normalization remains appropriate because model
  inputs and scenario split are unchanged.
- Six paired seeds are expected to provide descriptive evidence; high neural
  variability may still produce an unresolved H2 result.
- Timing estimates are workload- and host-specific.
- A finite and stable training curve may still converge to a biased solution.
- Training and held-out objectives may disagree.
- The new `K=20`, `K=35`, and `K=50` horizons have no prior Milestone 2 rank
  and are interpreted from the Milestone 3 curve only.
- The conclusion is descriptive for one deterministic dataset, architecture,
  optimizer family, selected shared policy, and update budget.

## Plans Superseded

This approved plan freezes and supersedes:

`temp_content/temp_plan/milestone_3_phase_a_proposal.md`

It supplements, and does not supersede or reopen:

- `docs/milestones/milestone_1_local_gradient_utility.md`;
- `docs/milestones/milestone_2_cpu_gpu_infrastructure.md`;
- `docs/milestones/milestone_2_gpu_scenario_batching.md`;
- `docs/milestones/milestone_2_iterative_structured_optimization.md`.

The closed Milestone 2 result and handoff remain authoritative historical
evidence. This file is authoritative for the Milestone 3 implementation pass.

## Phase D Boundary

Phase D has not started.

The next task, only when explicitly requested, must re-read the governing
documents and implement only this approved plan. This Phase C task stops after
freezing and verifying the plan.
