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


## Mandatory milestone workflow

Every scientifically distinct milestone must follow the same gated workflow.
Planning and implementation are separate Codex tasks. Do not combine them unless
the user explicitly waives the gate for a narrowly mechanical change.

You should store the temporal plan in a file at a specific folder: 
"./temp_content/" because stating the plan in the console is not human-readable and bad for plan editting. You should choose the appropriate subfolder to put the plan.
The file name and file formats are free to choose since it is only for demonstrative and discussion purposes.
The user may provide feedback under ./temp_plan/feedbacks.md.

### Phase A — inspect and propose

For the current milestone:

1. Read `AGENTS.md`, `PROJECT_CONTEXT.md`, `PLANS.md`, applicable approved
   milestone plans, and relevant code/tests/configuration.
2. Inspect actual third-party APIs and current repository behavior.
3. Do not edit project files, install packages, or run environment-modifying
   commands unless the user explicitly authorizes this planning task to do so.
4. Produce a milestone proposal containing:
   - verified current state and dependency behavior;
   - proposed scope and explicit non-goals;
   - files to create or modify and their responsibilities;
   - data flow, equations, units, tensor shapes, dtypes, and update ordering;
   - tests, numerical tolerances, commands, and acceptance criteria;
   - unresolved scientific and engineering decisions;
   - risks, assumptions, and expected consequences of each unresolved option.
5. Stop after the proposal. Do not implement.

### Phase B — discuss and revise

1. Treat the proposal as unapproved until the user explicitly accepts it.
2. Answer questions and revise the proposal without starting implementation.
3. Do not silently resolve choices that can change experimental meaning.
4. When alternatives exist, present the smallest viable options and their
   scientific consequences.
5. Continue until scope, semantics, validation, and stop conditions are explicit.

### Phase C — freeze the approved milestone plan

After explicit approval:

1. Save the accepted plan under:

   `docs/milestones/milestone_<N>_<short_name>.md`

2. Include at least:
   - status: `Approved`;
   - approval date if known;
   - milestone scope;
   - explicit non-goals;
   - implementation steps;
   - acceptance criteria;
   - stop conditions;
   - unresolved or deferred items;
   - plans superseded, if any.
3. The approved milestone file is authoritative for that implementation pass.
4. Do not overwrite high-level project methodology in `PROJECT_CONTEXT.md` or the
   milestone sequence in `PLANS.md` with low-level implementation details.
5. If the approved plan conflicts with `AGENTS.md` or fixed scientific invariants,
   stop and report the conflict before implementation.

### Phase D — implement only the approved plan

Implementation must be a separate task after plan approval.

1. Re-read `AGENTS.md`, `PROJECT_CONTEXT.md`, `PLANS.md`, and the approved
   milestone plan.
2. Implement only the approved scope.
3. Do not redesign the experiment, add features, broaden dependencies, or begin a
   later milestone.
4. If new evidence requires a material deviation:
   - stop before making the conflicting change;
   - report the evidence and affected assumptions;
   - propose the smallest plan amendment;
   - wait for explicit approval;
   - update the milestone plan before continuing.
5. Small implementation details that do not affect scientific meaning may be
   resolved conservatively and documented in the final report.
6. Phase D from different milestones should be clearly separated.

### Phase E — validate and report

1. Run all tests and acceptance checks specified by the approved plan.
2. Do not weaken, remove, or reinterpret a failed acceptance check merely to
   complete the milestone.
3. Report:
   - files changed;
   - commands run;
   - tests and quantitative results;
   - acceptance criteria status, item by item;
   - deviations and approved amendments;
   - assumptions made;
   - unresolved scientific or engineering risks.
4. Stop after reporting. Do not start the next milestone.

### Phase F — user review and closure

The milestone is not complete merely because code was produced.

1. Wait for the user to review the diff, evidence, and acceptance results.
2. The user decides whether the milestone:
   - passes;
   - needs a corrective implementation pass;
   - requires a plan amendment;
   - should be abandoned or deferred.
3. Plan the next milestone only after explicit closure of the current one.

### When the full gate may be skipped

The user may explicitly waive the planning gate for a narrowly mechanical task,
such as:

- correcting a typo;
- applying an already approved rename;
- adding a precisely specified assertion;
- rerunning an approved experiment configuration;
- making a local refactor with no behavioral or scientific effect.

Simulator equations, state updates, gradient construction, objectives, data
splits, controller parameterization, experiment comparisons, dependency changes,
and acceptance tests always require the full workflow unless the user explicitly
states otherwise.

## Operating rules

1. Inspect before editing.
2. Verify third-party APIs from the installed package or source. Do not invent
   `diffidm` interfaces from memory.
3. Before a choice that can change experimental meaning, stop and report:
   - the missing decision;
   - the smallest viable options;
   - the expected scientific effect of each option.
4. Prefer the smallest implementation that satisfies the approved milestone plan.
5. Follow the mandatory milestone workflow and never begin the next milestone automatically.
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
15. If a command is likely to fail because of sandboxing or restricted access
    rather than true project/environment state, request escalation for explicit
    user approval and rerun the same diagnostic or action before drawing
    conclusions. This includes GPU/driver visibility, network access,
    filesystem permissions outside the workspace, and other host resources.

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
