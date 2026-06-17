# PLANS.md

## Plan status

Current mainline:

> Gradient-horizon utility transfer in a deterministic differentiable IDM task.

Current milestone: **Milestone 0 — minimal simulator reproduction**.

Do not implement later milestones until Milestone 0 acceptance criteria pass and
the user explicitly requests continuation.

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

### Goal

Determine whether Stage 1 local utility predicts iterative optimization of the
four-parameter controller.

### Required work

1. Use identical initialization and optimizer settings for every gradient mode.
2. Run a fixed optimization budget.
3. Evaluate periodically on held-out leader profiles.
4. Store total and component losses.
5. Compare Stage 1 ranking with final optimization ranking.

### Acceptance criteria

- reproducible optimization curves;
- held-out evaluation separated from training;
- no per-mode objective or hyperparameter tuning;
- a clear result on whether one-step descent predicts iterative optimization.

### Decision gate

Proceed to Milestone 3 only if:

- at least one mode learns a nontrivial controller;
- the Stage 2 outcome is stable across a small number of seeds or initializations;
- the remaining question requires a learned model rather than further direct
  optimization analysis.

## Milestone 3 — small-model training

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

### Acceptance criteria

- training is reproducible;
- held-out results include total and component objectives;
- comparison isolates gradient horizon rather than architecture or tuning;
- the report states whether H2 is supported, contradicted, or unresolved.

## Minimal module plan

The exact repository layout may adapt to existing code, but responsibilities must
remain separated:

- `scenarios`: deterministic leader profiles and train/evaluation splits;
- `idm`: reference IDM equation and library wrapper;
- `rollout`: temporal integration and trajectory outputs;
- `controllers`: constant, structured, and later MLP controllers;
- `objectives`: shared loss and component reporting;
- `gradient_modes`: full and truncated backward connectivity;
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
