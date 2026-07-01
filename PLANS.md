# PLANS.md

## Milestone 0 — minimal reproduction

### Goal

Establish a trustworthy deterministic IDM rollout and reproduce one synthetic
parameter-fitting case.

### Required work

1. Inspect the available `diffidm` package or source.
2. Record:
   - installed version or repository commit;
   - available API;
   - supported dtypes and devices;
   - whether it supplies only a one-step IDM layer or a complete rollout.
3. Implement a plain Python/PyTorch reference IDM acceleration function.
4. Implement one differentiable IDM step using the selected library interface.
5. Implement a deterministic leader-profile generator.
6. Implement one leader–one follower rollout.
7. Generate a synthetic reference trajectory from known IDM parameters.
8. Fit only 2–3 parameters, initially selected from:
   - desired time headway;
   - desired speed;
   - minimum gap.
9. Add minimal unit and smoke tests.
10. Save a concise reproduction report.

### Acceptance criteria

Milestone 0 is complete only if:

- the plain and differentiable IDM step agree within a documented tolerance;
- the rollout is deterministic for a fixed configuration;
- fitting reduces trajectory loss substantially from initialization;
- fitted parameters move toward the known synthetic values;
- gradients are finite;
- a finite-difference directional check is consistent with the full autograd
  gradient on a small case;
- all commands and environment details required to reproduce the result are
  documented.

### Stop conditions

Stop and ask before proceeding if:

- `diffidm` uses equations or units incompatible with the intended formulation;
- collision handling or hidden clipping changes the scientific semantics;
- a package API cannot support double precision or required autograd behavior;
- a design choice materially changes the state update or fitting objective.

## Milestone 1 — local gradient utility

Status: Closed as passing on 2026-06-23. See
`docs/milestones/milestone_1_local_gradient_utility.md` for the approved plan,
amendments, results handoff, and Phase F closure decisions.

### Goal

Compare local, short, and full temporal gradient horizons using one-step descent.

### Required work

1. Add the four-parameter bounded headway controller defined in
   `PROJECT_CONTEXT.md`.
2. Implement gradient modes:
   - `K = 1`;
   - short horizon, default `K = 10`;
   - full horizon `K = T`.
3. Prove by test that all modes produce identical forward trajectories.
4. Implement normalized one-step descent.
5. Add random-direction controls.
6. Run across a small deterministic scenario set.
7. Report improvement probability, relative improvement, gradient norm, runtime,
   and memory.

### Acceptance criteria

- forward outputs are numerically identical across gradient modes;
- detachment cuts only intended backward dependencies;
- full gradient passes a directional derivative check;
- at least one gradient mode improves the objective more often than random
  directions;
- all modes use identical forward and evaluation settings.

### Initial decision gate

If no gradient mode beats random directions:

- do not proceed to direct optimization;
- inspect objective scaling, parameter identifiability, and detach placement;
- add only the diagnostics needed to identify the failure.

## Milestone 2 — direct structured optimization

Status: Closed as passing on 2026-06-24. See
`docs/milestones/milestone_2_iterative_structured_optimization.md` for the
approved plan and Phase F closure, and
`reports/milestone2/closure_and_milestone3_handoff.md` for the concise evidence
and Milestone 3 handoff.

The original required-work list below records the pre-planning milestone
intent. The later approved D.1b/D.2 plans superseded its provisional device and
batching assumptions: D.2 used validated scenario-batched CPU execution and did
not run a redundant full CUDA experiment.

### Goal

Determine whether Stage 1 local utility predicts iterative optimization of the
four-parameter controller.

### Required work

1. Use normalized controller inputs only.
2. Compare the approved horizons:
   - `K = 1`;
   - `K = 3`;
   - `K = 6`;
   - `K = 10`;
   - `K = T`.
3. Use the six approved controller initializations:
   - `T_init = [0.9, 1.2, 1.4, 1.6, 1.9, 2.2]`.
4. Use identical initialization, optimizer family, optimizer parameters,
   learning-rate policy, scenarios, objective, evaluation schedule, and update
   budget for every gradient mode.
5. Choose the exact optimizer family, learning-rate/search procedure, optimizer
   parameters, optimization budget, evaluation frequency, logging frequency, and
   CPU/GPU parity tolerances during Milestone 2 planning.
6. Run CPU/GPU parity checks before using CUDA results as main evidence.
7. If parity checks pass, run full CPU and CUDA structured-optimization
   comparisons with identical settings; the CPU run must duplicate the full GPU
   run as a faithful comparison.
8. Evaluate periodically on held-out leader profiles under no-grad evaluation.
9. Store total and component losses, final parameters, failure flags, device
   metadata, runtime, and enough per-initialization detail to compare
   variability.
10. Compare Stage 1 ranking with final optimization ranking.
11. Do not implement batching across scenarios or initializations as first-hand
   scope. Add batching only if preliminary timing evidence justifies it and the
   user approves that scope.

### Acceptance criteria

- reproducible optimization curves;
- held-out evaluation separated from training;
- no per-mode objective or hyperparameter tuning;
- normalized-input-only comparison; SI-unit evaluation is not part of Milestone
  2 unless a later approved plan opens a separate parameterization study;
- CPU/GPU parity check passes within approved tolerances before CUDA results are
  used as main evidence, or GPU execution is reported as blocked and CPU remains
  the valid path;
- CPU and CUDA runs use identical optimizer/scenario/initialization/evaluation
  settings when both are executed;
- reports include requested device, actual device, dtype, PyTorch version, CUDA
  availability, GPU name when applicable, and other dependency metadata;
- reports include total objective and progress/safety/jerk components for
  training and held-out evaluation;
- a clear result on whether one-step descent predicts iterative optimization.

### Decision gate

The decision gate passed:

- all five modes learned finite stable controllers;
- `K=80` and `K=10` improved held-out objective for all six initializations;
- the Stage 2 ranking exactly matched Stage 1;
- the remaining H2 question requires a learned model.

Milestone 2 closure selected `K=80` and `K=10` as the evidence-backed pair for
Milestone 3 planning. This selection does not start or approve Milestone 3.

## Milestone 3 — small-model training

Status: Implemented through Phase E for the base objective and approved
aggressive objective-weight sensitivity run; awaiting or subject to Phase F
review/closure. See `docs/milestones/milestone_3_small_model_training.md` and
`reports/milestone3/`.

### Goal

Test whether the gradient-horizon result transfers from a structured controller
to a small MLP.

### Required work

1. Implement one hidden layer with 16–32 units.
2. Preserve the same inputs, output bounds, simulator, objective, and scenario
   split.
3. Compare only:
   - full gradient;
   - the best non-full gradient from Milestone 2.
4. Use identical initialization protocol and training budget.
5. Evaluate on held-out profile families.

### Milestone 2 handoff defaults

Use these as Phase A starting points, not as already approved Milestone 3
settings:

- compare `K=80` and `K=10`;
- preserve normalized inputs, bounded headway output, simulator, objective,
  scenario split, and held-out no-grad isolation;
- pair identical initial MLP weights across horizons and use multiple fixed
  model seeds;
- select one shared training-only optimizer/LR policy with no per-horizon
  tuning;
- determine the update budget from MLP-specific convergence evidence;
- remeasure CPU/CUDA performance for the fixed MLP architecture and complete
  parity checks before using CUDA results as main evidence;
- retain component, safety, convergence, failure, and held-out reporting;
- do not treat more scenario samples as an automatic correction for temporal
  truncation bias.

Open choices include architecture width, seed count, optimizer/LR grid, update
budget, logging cadence, device policy, batching scope, and optional
gradient-alignment diagnostics.

### Acceptance criteria

- training is reproducible;
- held-out results include total and component objectives;
- comparison isolates gradient horizon rather than architecture or tuning;
- the report states whether H2 is supported, contradicted, or unresolved.

### Result handoff

Milestone 3 supports H2 for the approved deterministic small-MLP experiments.
Both the base objective and aggressive objective-weight sensitivity run produced
the same held-out median ranking:

```text
K=80 > K=50 > K=35 > K=20 > K=10 > K=6
```

Full temporal gradients remained best, `K=50` was the best truncated horizon,
and short horizons remained substantially weaker. The next project direction is
therefore sparse-gradient methodology: preserving useful long-horizon temporal
credit with lower backward temporal resolution while keeping forward rollouts
unchanged.

## SG1 — sparse long-horizon gradient resolution

Status: Approved for Phase D planning handoff on 2026-07-01. See
`docs/milestones/milestone_sg1_sparse_gradient.md` for the authoritative
implementation and validation plan. Do not implement before a separate Phase D
task.

SG1 means "sparse gradient 1." It corresponds to the discussed Stage 0 and
Stage 1 sparse-gradient methodology.

### Goal

Test whether sparse checkpoint-level backward connectivity can preserve useful
long-horizon temporal gradient information while keeping the forward IDM rollout
and full-resolution objective unchanged.

### Conceptual required work

1. Preserve the exact Milestone 3 forward rollout, objective evaluation,
   scenario split, bounded MLP, normalization, and held-out no-grad evaluation.
2. Define a sparse full-horizon gradient method that lowers only backward
   temporal resolution through checkpoint-to-checkpoint span sensitivities.
3. Run gradient-only admission diagnostics before full training.
4. Compare sparse full-horizon gradients against dense baselines:
   - dense `K=80`;
   - dense `K=50`;
   - dense `K=10`.
5. Report held-out total and component objectives, convergence, gradient
   alignment, gradient norms, failure counts, runtime, and memory where
   available.
6. Treat sparse truncated gradients such as `K=50` with sparse stride as
   conditional secondary scope after sparse full-horizon behavior is
   informative.
7. Treat `T=160` as a conditional stress test after the primary `T=80`
   comparison shows nontrivial sparse-gradient promise.

### Phase C decisions

The approved SG1 plan fixes:

- sparse VJP/adjoint accumulation with a B1 anchored macro-step surrogate;
- checkpoint state `[x_t, v_t]`;
- sparse strides `m=[2,4,6,8]`;
- complete-span-only `m=6` behavior;
- Stage 0 admission before training;
- all six Milestone 3 model seeds for Stage 1;
- base objective only;
- Milestone 3 base LR/update policy first;
- dense `K=80`, `K=50`, and `K=10` rerun as SG1 references.

### Acceptance criteria

SG1 acceptance criteria are frozen in
`docs/milestones/milestone_sg1_sparse_gradient.md`. At minimum, they require:

- exact forward-value equivalence to dense rollout;
- finite sparse gradients;
- confirmed sparse backward connectivity rather than dense reverse-mode through
  every micro-step;
- objective normalization matching dense baselines;
- no held-out leakage into training, calibration, stopping, or checkpoint
  selection;
- a clear downstream result on whether sparse full-horizon gradients preserve
  useful long-horizon signal.

### Decision gate

Proceed to SG2 only after SG1 Phase E evidence is reviewed and the user
explicitly decides that sparse long-horizon gradients carry enough useful signal
to justify hybrid-method planning.

## SG2 — hybrid dense-short plus sparse-long gradients

Status: Conditional future milestone. Do not plan or implement until SG1 is
reviewed and explicitly judged sufficient to continue.

SG2 means "sparse gradient 2." It corresponds to the discussed Stage 2 hybrid
method.

### Goal

Test whether dense full-resolution gradients over a recent short window can be
combined with sparse long-range checkpoint gradients to retain local control
fidelity and long-horizon bias-correction utility.

### Conceptual required work

1. Use SG1 evidence to choose a sparse long-horizon component.
2. Define a dense recent temporal window and sparse older temporal window.
3. Combine dense-short and sparse-long contributions with a predeclared
   objective-consistent rule.
4. Compare the hybrid method against:
   - dense `K=80`;
   - dense `K=50`;
   - dense `K=10`;
   - SG1 sparse full-horizon method.
5. Preserve the same forward rollout, objective, scenarios, initialization,
   optimizer policy, and held-out evaluation controls used for SG1 unless an
   approved SG2 plan explicitly changes them.

### Open decisions for SG2 planning

- dense recent-window length;
- sparse older-window extent and stride;
- whether to include sparse truncated variants;
- whether to include `T=160`;
- exact hybrid combination rule;
- whether any blend weights are allowed, and if so how they are fixed without
  per-method tuning;
- acceptance criteria for hybrid utility versus SG1 and dense baselines.

### Non-goals unless separately approved

- no learned or tuned gradient-resolution schedule;
- no per-method objective or loss-weight tuning;
- no policy-gradient, derivative-free, hard-simulator, platoon, or stochastic
  traffic comparison;
- no change to forward simulator semantics.

## Minimal module plan

The exact repository layout may adapt to existing code, but responsibilities must
remain separated:

- `scenarios`: deterministic leader profiles and train/evaluation splits;
- `idm`: reference IDM equation and library wrapper;
- `rollout`: temporal integration and trajectory outputs;
- `controllers`: constant, structured, and later MLP controllers;
- `objectives`: shared loss and component reporting;
- `gradient_modes`: full, truncated, and later approved sparse backward
  connectivity;
- `diagnostics`: one-step descent and limited numerical checks;
- `optimize`: direct controller optimization;
- `train`: later small-model training;
- `evaluate`: held-out evaluation without gradient tracking;
- `tests`: numerical, forward-equivalence, and detachment tests.

## Initial environment defaults

These are defaults, not yet fixed scientific decisions:

- Ubuntu 24.04;
- Python 3.11;
- PyTorch;
- `diffidm` pinned to an inspected version or commit;
- NumPy;
- SciPy;
- Pandas;
- Matplotlib;
- PyYAML;
- pytest;
- TensorBoard only if useful after the smoke test.

Do not introduce SUMO, JAX, RL libraries, distributed frameworks, or container
orchestration for the initial milestones.

## First Codex task

When starting implementation, the user should ask Codex to:

1. read `AGENTS.md`, `PROJECT_CONTEXT.md`, and `PLANS.md`;
2. inspect the repository and installed `diffidm` interface;
3. propose the smallest Milestone 0 file plan;
4. implement only Milestone 0;
5. run its acceptance tests;
6. stop and report before beginning Milestone 1.
