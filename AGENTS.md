# AGENTS.md

## Purpose

This repository studies whether gradients from a differentiable traffic simulator
are useful for downstream optimization and model training.

The first mainline is deliberately narrow:

> Compare full and truncated temporal gradients through a deterministic
> differentiable IDM rollout, while keeping the forward simulation identical.

Before planning or editing code, read:

1. `PROJECT_CONTEXT.md`
2. `PLANS.md`
3. the existing source, tests, and environment files

`PROJECT_CONTEXT.md` defines scientific intent and invariants.
`PLANS.md` defines the current implementation milestone and acceptance criteria.
Code and tests define current implemented behavior.

## Scientific invariants

These rules must not be violated without explicit user approval.

1. The initial simulator is deterministic, single-lane, and leader–follower.
2. Vehicle ordering and leader identity are fixed.
3. There is no lane changing, route choice, stochastic arrival, discrete vehicle
   generation, signal logic, reinforcement learning, or SUMO integration.
4. Every gradient mode must use the same forward rollout, state values, scenarios,
   objective, initialization, and evaluation procedure.
5. Gradient truncation may alter only backward graph connectivity. It must not
   change forward values.
6. Evaluation scenarios are held out and must not affect training, optimization,
   normalization, or hyperparameter selection.
7. Controller outputs and fitted IDM parameters must remain physically bounded.
8. The first study compares temporal gradient horizons only. Do not add unrelated
   gradient estimators, simulators, models, or baselines without approval.
9. Do not claim that a gradient is correct or useful solely because reconstruction
   loss decreases. Downstream utility must be measured directly.
10. Do not silently replace the scientific task with generic IDM fitting,
    trajectory prediction, traffic-signal control, or neural policy learning.

## Decision status vocabulary

Interpret project statements using these labels:

- **Fixed**: must be implemented as written.
- **Default**: use unless evidence or an API constraint requires reconsideration.
- **Hypothesis**: an empirical claim to test, not an assumed truth.
- **Open**: requires a documented decision before implementation.
- **Deferred**: outside the current mainline.

Do not convert hypotheses or defaults into fixed scientific requirements.

## Operating rules

1. Inspect before editing.
2. Verify third-party APIs from the installed package or source. Do not invent
   `diffidm` interfaces from memory.
3. Before a choice that can change experimental meaning, stop and report:
   - the missing decision;
   - the smallest viable options;
   - the expected scientific effect of each option.
4. Prefer the smallest implementation that satisfies the current milestone.
5. Implement and validate one milestone before beginning the next.
6. Do not add dependencies unless the current milestone requires them.
7. Do not add fallback clipping, safety branches, smoothing, or regularization
   that changes simulator semantics without documenting and obtaining approval.
8. Do not tune separate objectives or loss weights for different gradient modes.
9. Keep experiment settings configurable rather than duplicated in scripts.
10. Set and record random seeds even when the current simulation is deterministic.
11. Use double precision for numerical gradient checks unless there is a documented
    reason not to.
12. Run relevant tests after each material change.
13. Never place secrets, credentials, or private data in the repository.
14. Do not modify or delete user data or existing results unless explicitly asked.

## Required validation discipline

For changes to the simulator or gradient modes, verify at minimum:

1. A plain reference IDM step and the differentiable step agree numerically.
2. Forward trajectories are identical across full and truncated gradient modes.
3. The full gradient passes a central finite-difference directional check on a
   small deterministic case.
4. Detachment tests confirm that only the intended temporal dependencies are cut.
5. Held-out evaluation runs under `torch.no_grad()` or equivalent.

If a validation fails, report it rather than weakening the test.

## Implementation style

- Language: Python.
- Primary framework: PyTorch.
- Target platform: Ubuntu 24.04.
- Prefer typed, small, testable functions.
- Keep scenario generation, simulation, objectives, gradient modes, optimization,
  and evaluation in separate modules.
- Use explicit tensor shapes, units, dtypes, and device handling.
- Avoid hidden global mutable state.
- Save machine-readable results in CSV or JSONL and human-readable summaries in
  Markdown or plain text.
- Do not use notebooks as the only implementation of core experiments.

## Work report

At the end of each Codex task, report:

- files changed;
- assumptions made;
- commands run;
- tests and results;
- unresolved scientific or engineering risks;
- the next milestone, without starting it unless requested.
