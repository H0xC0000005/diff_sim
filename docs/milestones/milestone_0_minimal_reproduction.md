# Milestone 0: Minimal Simulator Reproduction

Status: Approved
Approval date: 2026-06-17

## Purpose

Establish a trustworthy deterministic one-leader, one-follower IDM rollout and reproduce one synthetic parameter-fitting case. This milestone is the replication and simulator trust-building stage before any temporal gradient-horizon comparison.

The implementation must follow `AGENTS.md`, `PROJECT_CONTEXT.md`, and `PLANS.md`. If this approved plan conflicts with those documents or with fixed scientific invariants, stop and report the conflict before implementation.

## Verified Repository State At Approval

- Repository root: `/home/zpz/repos/differential_sim`.
- Current branch at inspection: `main`, HEAD `d2fa471 init`.
- Existing tracked project files at inspection: `AGENTS.md`, `PROJECT_CONTEXT.md`, `PLANS.md`, and `.gitignore`.
- No source modules, tests, reports, or prior approved milestone plans existed at Phase A inspection.
- Existing user-provided worktree context at inspection:
  - modified `AGENTS.md`;
  - untracked `environment.yml`.
- Treat uncommitted files not managed by Codex as user-provided context. Do not revert them unless explicitly instructed.

## Verified Environment And Dependency State

Use the project conda environment for implementation and validation:

```bash
/home/zpz/miniconda3/envs/differential_sim/bin/python
```

Verified package state in that environment:

- Python `3.11.15`;
- `torch 2.5.1`;
- `numpy 2.4.6`;
- `scipy 1.17.1`;
- `pandas 3.0.3`;
- `matplotlib 3.11.0`;
- `PyYAML 6.0.3`;
- `pytest 9.1.0`;
- `diffidm 0.0.3`.

CUDA was not available during inspection. The base shell Python was `3.13.13` and did not have the declared dependencies installed; do not use it for milestone validation unless the environment has been explicitly changed and re-verified.

`environment.yml` specifies Python `3.11`, PyTorch CPU, NumPy, SciPy, Pandas, Matplotlib, PyYAML, pytest, `diffidm==0.0.3`, and `MPLCONFIGDIR=/tmp/matplotlib`.

## Verified `diffidm` API And Behavior

Installed package:

- version: `diffidm==0.0.3`;
- installed path: `/home/zpz/miniconda3/envs/differential_sim/lib/python3.11/site-packages/diffidm`;
- distribution homepage: `https://github.com/SonSang/diffidm`;
- installed wheel metadata does not expose a repository commit.

Public API:

- `diffidm.__init__` exports `IDMLayer`;
- source files are `diffidm/__init__.py` and `diffidm/layer.py`;
- `diffidm` supplies a one-step IDM acceleration layer only, not a temporal rollout.

`IDMLayer` is a class with static methods using ordinary PyTorch operations. It is not a `torch.nn.Module` and does not subclass `torch.autograd.Function`.

Primary call:

```python
IDMLayer.apply(
    a_max,
    a_min,
    a_pref,
    v_curr,
    v_target,
    pos_delta,
    vel_delta,
    min_space,
    time_pref,
    delta_time,
    prevent_negative_speed=True,
)
```

The package documents each tensor argument as shape `[N]`. CPU `float32` and `float64` calls were verified to preserve output dtype and produce finite autograd gradients for representative inputs.

Unclamped helper equations:

```text
s_star = min_space + v_curr * time_pref
       + v_curr * vel_delta / (2 * sqrt(a_max * a_pref))

acc = a_max * (1 - (v_curr / v_target)^4 - (s_star / pos_delta)^2)
```

`IDMLayer.apply` then applies smooth lower bounds:

```text
s_star = soft_min_clamp(s_star, 0.0)
acc = soft_min_clamp(acc, acc_lb)
```

where:

```text
soft_min_clamp(x, min_val) = min_val + softplus(x - min_val)
```

If `prevent_negative_speed=True`, then:

```text
acc_lb = max(-v_curr / delta_time, a_min)
```

Otherwise:

```text
acc_lb = a_min
```

This smooth lower bound is not an exact clamp and changes values even above the bound. Verified examples:

- `soft_min_clamp(0, 0) = 0.69314718056`;
- `soft_min_clamp(1, 0) = 1.31326168752`;
- `soft_min_clamp(10, 0) = 10.0000453989`;
- `soft_min_clamp(-10, 0) = 0.0000453988992169`.

The smooth-clamp behavior is approved as essential to faithful `diffidm` replication. It must be preserved and documented rather than treated as accidental hidden clipping.

## Scope

Implement the smallest reusable Milestone 0 system that:

1. Adds a small Python package for deterministic IDM components.
2. Implements a plain PyTorch textbook IDM acceleration helper.
3. Implements a `diffidm` one-step wrapper with documented smooth-clamped semantics.
4. Implements deterministic exogenous leader profile generators.
5. Implements one leader-one follower rollout with explicit temporal integration.
6. Generates synthetic reference trajectories from known IDM parameters using the selected Milestone 0 rollout semantics.
7. Fits configurable 2-parameter and 3-parameter IDM settings against synthetic trajectories using full autograd through the rollout.
8. Runs the 2-parameter fit as the primary smoke experiment.
9. Adds unit and smoke tests for step agreement, determinism, finite gradients, fitting movement, and finite-difference consistency.
10. Saves a concise reproduction report with both useful summary findings and raw machine-readable results.

## Explicit Non-Goals

- No gradient horizon comparison.
- No truncated-gradient or detachment modes.
- No bounded headway controller from Milestone 1.
- No progress/safety/comfort objective from later milestones.
- No held-out evaluation protocol beyond documenting that it is deferred.
- No lane changes, route choice, stochastic arrivals, signal logic, reinforcement learning, SUMO, or multi-vehicle platoons.
- No dependency additions unless an implementation blocker appears and a plan amendment is approved.
- No full reproduction of `diffidm` paper experiments.
- No paper-replication experiment beyond optional very quick package-verification checks that fit Milestone 0 without new dependencies or scope expansion.
- No follower-speed clamp in the rollout update. If the accepted scenario or initialization produces negative follower speed, stop and revise the plan rather than adding a safety branch.

## Approved File Plan

Use a minimal `src` layout:

- `pyproject.toml`
  - project metadata;
  - pytest configuration;
  - package discovery for `src/differential_sim`.
- `src/differential_sim/__init__.py`
  - package marker and optional version constant.
- `src/differential_sim/idm.py`
  - `IDMParameters` dataclass;
  - plain textbook IDM acceleration helper;
  - smooth-clamped `diffidm` wrapper;
  - parameter packing and bounded transforms if bounded fitting is selected.
- `src/differential_sim/scenarios.py`
  - deterministic leader profile generation for constant-speed, braking-and-recovery, and sinusoidal profiles;
  - scenario config dataclass.
- `src/differential_sim/rollout.py`
  - explicit one-leader, one-follower rollout;
  - trajectory output dataclass or typed dictionary.
- `src/differential_sim/fit.py`
  - synthetic trajectory generation;
  - small parameter-fitting routine;
  - finite-difference directional derivative helper.
- `scripts/run_milestone0.py`
  - reproducible smoke run;
  - writes machine-readable and human-readable report outputs.
- `tests/test_idm.py`
  - plain equation tests;
  - `diffidm` wrapper characterization;
  - step agreement tests.
- `tests/test_rollout.py`
  - deterministic rollout and shape tests.
- `tests/test_fit.py`
  - finite gradients;
  - fitting reduces trajectory loss;
  - fitted parameters move toward known synthetic values;
  - central finite-difference directional check.
- `reports/milestone0/`
  - generated during implementation validation;
  - contains both a useful summary report and raw machine-readable results.

## Equations, Units, Shapes, Dtype, And Update Ordering

Units:

- position `x`: meters;
- speed `v`: meters/second;
- acceleration `a`: meters/second squared;
- time `t`, `dt`, `time_pref`: seconds;
- gap `s`: meters;
- relative speed `delta_v`: meters/second.

Default trajectory shapes:

- `leader_x`: `[T + 1]`;
- `leader_v`: `[T + 1]`;
- `follower_x`: `[T + 1]`;
- `follower_v`: `[T + 1]`;
- `follower_a`: `[T]`;
- `gap`: `[T + 1]`;
- `delta_v`: `[T + 1]`.

Internally, scalar tensors may be represented as shape `()` or `[1]`, but public rollout outputs must be consistently time-major.

Dtype and device:

- Use `torch.float64` for numerical checks and default Milestone 0 tests.
- Keep device explicit, default `cpu`.
- The `diffidm` wrapper accepts tensors already on the requested device.

Textbook IDM helper:

```text
delta_v_t = v_follower_t - v_leader_t
s_t = x_leader_t - x_follower_t - leader_length
s_star_t = s0 + v_t * T_headway
         + v_t * delta_v_t / (2 * sqrt(a_max * b_comfort))
a_t = a_max * (1 - (v_t / v0)^delta - (s_star_t / s_t)^2)
```

Approved defaults and semantics:

- `delta = 4`;
- `leader_length = 5.0 m`;
- physically valid positive gap scenarios only;
- no collision handling;
- no fallback clipping;
- no stochastic noise;
- preserve `diffidm` smooth lower bounds in the `diffidm` replication path.

Mapping to `diffidm`:

- `a_max` -> `a_max`;
- `a_min` -> negative deceleration lower bound;
- `a_pref` -> `b_comfort`;
- `v_curr` -> follower speed;
- `v_target` -> desired speed `v0`;
- `pos_delta` -> gap `s`;
- `vel_delta` -> `delta_v = v_follower - v_leader`;
- `min_space` -> `s0`;
- `time_pref` -> desired time headway `T_headway`;
- `delta_time` -> rollout `dt`.

Update ordering:

1. At time index `t`, read follower `x_t`, `v_t` and leader `x_leader_t`, `v_leader_t`.
2. Compute `s_t` and `delta_v_t`.
3. Compute `a_t` from IDM.
4. Update speed explicitly:

```text
v_{t+1} = v_t + dt * a_t
```

5. Update position from the updated speed:

```text
x_{t+1} = x_t + dt * v_{t+1}
```

6. Recompute `s_{t+1}` and `delta_v_{t+1}` from the exogenous leader trajectory at `t + 1`.

## Approved `diffidm` Semantics Decision

Implement both paths if feasible:

- a textbook helper path for equation transparency and unclamped helper comparison;
- a smooth-clamped `diffidm.IDMLayer.apply` replication path for the primary differentiable step.

If both paths cannot be maintained cleanly within Milestone 0, fall back to the smooth-clamped `diffidm.IDMLayer.apply` semantics as the primary implementation. Faithfulness of replication has priority for Milestone 0.

The report must explicitly state which path generated the synthetic fitting trajectory and which path is only a transparency check.

## Synthetic Fitting Task

Leader profiles:

- implement constant speed;
- implement braking-and-recovery pulse;
- implement sinusoidal speed disturbance;
- use braking-and-recovery as the primary acceptance profile for fitting;
- test that all three profiles generate deterministic, finite trajectories.

Synthetic data:

- Generate follower trajectory from known IDM parameters.
- Use identical rollout code for synthetic generation and fitting, except fitted parameters are initialized away from truth.
- Use conservative documented defaults for synthetic truth, initialization, bounds, and optimization budget.
- If conservative defaults fail identifiability and fixing that requires materially different scenario severity, parameters, objective, or acceptance criteria, stop and request a plan amendment.

Primary fitted parameters:

- Fit desired time headway `T_headway`.
- Fit desired speed `v0`.

Fixed in the primary run:

- maximum acceleration `a_max`;
- comfortable deceleration `b_comfort`;
- minimum gap `s0`;
- leader length;
- initial follower state.

Secondary default fitted parameters:

- Fit desired time headway `T_headway`.
- Fit desired speed `v0`.
- Fit minimum gap `s0`.

The implementation must make the fitted parameter subset configurable and reusable for later milestones.

Approved loss:

```text
trajectory_loss = mean((x_pred - x_ref)^2 / x_scale^2)
                + mean((v_pred - v_ref)^2 / v_scale^2)
```

Approved scale defaults:

- `x_scale = 10.0 m`;
- `v_scale = 5.0 m/s`.

Optimization:

- use full autograd through the rollout;
- use `torch.optim.Adam`;
- set and record a fixed seed even though the current simulation is deterministic;
- use a conservative fixed smoke-test budget, approximately 200-500 steps, and document the exact value;
- do not add gradient-mode comparison.

Physical bounds:

- Use unconstrained train variables transformed into documented physical ranges.
- Approved default bounds:
  - `T_headway in [0.5, 3.0] s`;
  - `v0 in [10.0, 40.0] m/s`;
  - optional `s0 in [0.5, 10.0] m`.

## Tests, Tolerances, And Commands

Primary validation commands:

```bash
/home/zpz/miniconda3/envs/differential_sim/bin/python -m pytest
/home/zpz/miniconda3/envs/differential_sim/bin/python scripts/run_milestone0.py
```

Step agreement tests:

- Compare textbook helper implementation against `diffidm.compute_optimal_spacing` and `diffidm.compute_acceleration` before clamps.
- Compare smooth-clamped reference implementation against `IDMLayer.apply`.
- Use `torch.float64`.
- Use `rtol=1e-12`, `atol=1e-12` for these direct equation comparisons unless implementation evidence shows this is infeasible; if infeasible, stop and report rather than weakening silently.

Rollout determinism tests:

- Re-running the same fixed configuration should reproduce outputs exactly on CPU in the same dtype and operation order.
- Use `torch.equal` where practical; otherwise use `rtol=0`, `atol=0`.

Finite-difference directional derivative check:

- Use central finite difference.
- Use `torch.float64`.
- Check multiple epsilon values, proposed `1e-4`, `3e-5`, and `1e-5`.
- Accept relative error `< 1e-3` for a stable epsilon or absolute error `< 1e-6` when derivative magnitude is small.

Fitting smoke checks:

- final loss at least 50% lower than initial loss;
- fitted parameters move closer to synthetic truth in normalized parameter space;
- all gradients finite.

## Acceptance Criteria

Milestone 0 is accepted only if:

1. Installed environment and `diffidm` version/API are documented in the report.
2. The approved plain/textbook and smooth-clamped differentiable IDM step comparisons pass within documented tolerance.
3. The rollout is deterministic for a fixed configuration.
4. Synthetic fitting substantially reduces trajectory loss from initialization.
5. Fitted parameters move toward known synthetic values.
6. Gradients are finite during the fitting run.
7. A central finite-difference directional check is consistent with the full autograd gradient on a small deterministic case.
8. Commands and environment details required to reproduce the result are documented.
9. Result summary and raw machine-readable results are saved in the same milestone report directory.

## Stop Conditions

Stop before making the conflicting change and report if:

- `diffidm` smooth-clamped semantics cannot be preserved in the replication path.
- The selected `diffidm` comparison cannot satisfy the documented tolerance without redefining the reference equation.
- The fitting parameters are not identifiable enough for the primary smoke task under the approved braking-and-recovery profile.
- Collision handling, follower-speed clipping, hidden safety branches, extra smoothing, or fallback regularization would be needed to keep the rollout finite.
- Double precision or required autograd behavior fails for the selected step.
- Any proposed change would broaden Milestone 0 into gradient-horizon comparison, downstream controller optimization, or full paper-experiment replication.
- A quick paper-documented verification would require new dependencies, network access, nontrivial experiment infrastructure, or a broader experiment target.
- Conservative default constants fail and the needed adjustment would materially change scenario severity, fitted parameters, objective, or acceptance criteria.

If a stop condition is hit, propose the smallest plan amendment and wait for explicit approval before continuing.

## Deferred Items

- Temporal gradient horizons `K = 1`, short horizon, and full horizon.
- Detachment-mode tests.
- One-step descent utility diagnostics.
- Structured bounded headway controller.
- Direct structured optimization.
- Held-out evaluation protocol.
- Small MLP training.
- Full reproduction of `diffidm` paper experiments.
- Plots, unless needed to diagnose Milestone 0 fitting failure.

## Superseded Plans

This approved milestone plan supersedes the temporary Phase A/Phase B discussion plan:

- `temp_plan/milestone_0_phase_a_proposal.md`

The temporary file is retained for discussion history but is not authoritative for implementation.

## Phase D Instructions

When implementation begins in a separate task:

1. Re-read `AGENTS.md`, `PROJECT_CONTEXT.md`, `PLANS.md`, and this file.
2. Implement only this approved Milestone 0 scope.
3. Do not begin Milestone 1.
4. Use `/home/zpz/miniconda3/envs/differential_sim/bin/python` unless the environment is explicitly changed and re-verified.
5. Run the approved tests and smoke script.
6. Report files changed, commands run, quantitative results, acceptance criteria status item by item, deviations, assumptions, and unresolved risks.
7. Stop after reporting and wait for user review.

