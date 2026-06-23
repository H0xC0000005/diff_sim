# Milestone 1: Local Gradient Utility

Status: Approved
Approval date: 2026-06-19
Amended: 2026-06-22

## Purpose

Measure local downstream utility of temporal simulator gradients in the existing
deterministic one-leader, one-follower differentiable IDM rollout. The diagnostic
compares one-step descent directions from temporal horizons `K=[1,3,6,10,T]`
against normalized random directions while keeping the forward rollout,
controller initialization, scenarios, objective, and evaluation procedure
identical.

This milestone tests Stage 1 of H1 only: whether a gradient mode has higher
one-step objective-improvement probability than random parameter-space
directions. It does not test iterative optimization or model-training transfer.

## Scope

- Use the Milestone 0 smooth-clamped `diffidm` semantics and explicit
  one-leader, one-follower update order.
- Add the bounded four-parameter structured headway controller.
- Add full and truncated temporal gradient modes for `K=[1,3,6,10,T]`, with
  `T=80`.
- Expand the deterministic scenario set beyond elementary profiles with
  approved random braking, multi-pulse braking, chirp sinusoid, and mixed-regime
  leader profiles.
- Run a normalized-input primary one-step diagnostic and a full-parity SI-unit
  input validation diagnostic.
- Report gradient cosine similarity to the full-gradient direction.
- Save machine-readable raw outputs and human-readable summaries.

## Non-Goals

- No iterative controller optimization.
- No MLP training.
- No reinforcement learning, SUMO, lane changes, route choice, stochastic
  arrivals, signals, second simulator, or additional gradient estimator.
- No per-gradient-mode tuning of objective weights, scenario sets,
  initialization, normalization, alpha grid, or evaluation.
- No change to Milestone 0 smooth-clamped forward simulator semantics.
- No CPU peak-memory metric; report runtime and qualitative graph-size
  expectations only.
- No sawtooth/noisy emergency-brake stress profile in this amended Milestone 1
  pass.

## Fixed Simulator Semantics

Use Milestone 0 `diffidm` acceleration semantics:

```text
delta_v_t = v_follower_t - v_leader_t
s_star_t = s0 + v_t * T_t + v_t * delta_v_t / (2 * sqrt(a_max * b_comfort))
s_star_t = soft_min_clamp(s_star_t, 0)
a_raw_t = a_max * (1 - (v_t / v0)^4 - (s_star_t / gap_t)^2)
a_lb_t = max(-v_t / dt, a_min)    when prevent_negative_speed=True
a_t = soft_min_clamp(a_raw_t, a_lb_t)
```

where `soft_min_clamp(x, m) = m + softplus(x - m)`.

Update order:

1. At time `t`, read follower `x_t`, `v_t` and exogenous leader `x_leader_t`,
   `v_leader_t`.
2. Compute `gap_t = x_leader_t - x_t - leader_length`.
3. Compute controller headway `T_t`.
4. Compute IDM acceleration `a_t`.
5. Update speed `v_{t+1} = v_t + dt * a_t`.
6. Update position `x_{t+1} = x_t + dt * v_{t+1}`.
7. Recompute output `gap` and `delta_v` from resulting trajectories.

Units: meters, seconds, m/s, m/s^2. Default dtype for checks and diagnostics is
`torch.float64` on CPU.

Fixed diagnostic rollout values are configurable but must default to:

- `steps=80`, `dt=0.2`;
- `leader_length=5.0`;
- `prevent_negative_speed=True`;
- initial follower state `position=-22.0`, `speed=16.0`;
- fixed IDM parameters except controller-selected headway:
  `a_max=1.4`, `b_comfort=2.0`, `v0=28.0`, `s0=2.0`, `a_min=-8.0`.

Record these values in every result artifact.

## Controller

Primary controller:

```text
T_t = T_min + (T_max - T_min) * sigmoid(
    beta0 + beta1 * z_v_t + beta2 * z_delta_v_t + beta3 * z_gap_t
)
```

Inputs:

- `v_t`: follower speed, m/s;
- `delta_v_t = v_follower_t - v_leader_t`, m/s;
- `gap_t`, m.

Bounds:

- `T_min=0.5 s`;
- `T_max=3.0 s`.

Primary input parameterization:

- compute `mu` and `sigma` from diagnostic scenarios only;
- use the noiseless constant-headway centers listed below for initialization;
- do not use held-out scenarios;
- use `z = (x - mu) / sigma`;
- apply a `sigma` floor of `1e-12` only for numerical degeneracy and report if
  used.

Secondary validation:

- rerun full one-step diagnostics with natural SI-unit inputs
  `[v_t, delta_v_t, gap_t]`;
- use the same scenarios, initializations, objective, horizons, alpha grid,
  random directions, and evaluation;
- interpret this only as parameterization sensitivity, not the primary result.

Initialization:

- centers: `T_init in [0.9, 1.2, 1.4, 1.6, 1.9, 2.2] s`;
- center beta:
  `beta_center = [logit((T_init - T_min) / (T_max - T_min)), 0, 0, 0]`;
- add `0.01 * xi` to all beta components;
- generate one standard-normal vector `xi` per center with
  `torch.Generator` seeds `0`, `1`, `2`, `3`, `4`, and `5`;
- serialize generated beta vectors in the report.

## Objective

Compute the same whole-rollout objective for every gradient mode:

```text
J = mean_scenarios mean_t [
    1.0 * progress_error_t
  + 0.7 * safety_penalty_t
  + 5.0 * jerk_penalty_t
]
```

Use equal scenario weighting: compute each scenario objective as a time mean,
then average named scenarios equally.

Components:

```text
progress_error_t = ((v_follower_t - v_leader_t) / 5.0)^2
safety_penalty_t = softplus((s_safe_t - gap_t) / 5.0)^2
jerk_penalty_t = ((a_t - a_{t-1}) / 2.0)^2, averaged over t=1..T-1
```

Safety target:

```text
s_safe_t = s0 + v_t * T_safe + v_t * delta_v_t / (2 * sqrt(a_max * b_comfort))
s_safe_t = soft_min_clamp(s_safe_t, 0)
```

Use fixed safety parameters `s0=2.0 m`, `T_safe=1.0 s`,
`a_max=1.4 m/s^2`, `b_comfort=2.0 m/s^2`. Do not use controller-selected
`T_t` inside `s_safe_t`.

Report total objective, unweighted component magnitudes, weighted component
contributions, and diagnostic-only inverse-mean-magnitude semantic weights
computed from unweighted component means over all diagnostic scenarios and all
six approved noisy initializations before any one-step update. These suggested
semantic weights must not be used to retune this amended Milestone 1 result.

## Scenarios

Use deterministic scenario configs. Held-out scenarios must not affect
normalization, gradient construction, objective-weight selection, alpha
selection, or random-direction generation.

Diagnostic scenarios, all with `steps=80`, `dt=0.2`, initial leader position
`0.0 m`, default initial follower state:

- `constant_16`: `kind="constant"`, `initial_speed=16.0`;
- `constant_20`: `kind="constant"`, `initial_speed=20.0`;
- `brake_mild`: `kind="braking_recovery"`, `initial_speed=18.0`,
  `brake_delta_v=3.0`, `brake_start=4.0`, `brake_duration=2.0`,
  `recovery_duration=3.0`;
- `brake_stronger`: `kind="braking_recovery"`, `initial_speed=18.0`,
  `brake_delta_v=5.0`, `brake_start=4.0`, `brake_duration=2.0`,
  `recovery_duration=3.0`;
- `sinusoidal_low`: `kind="sinusoidal"`, `initial_speed=18.0`,
  `sinusoid_amplitude=1.5`, `sinusoid_period=8.0`;
- `sinusoidal_high`: `kind="sinusoidal"`, `initial_speed=18.0`,
  `sinusoid_amplitude=2.5`, `sinusoid_period=10.0`;
- `random_brake_0`, `random_brake_1`, `random_brake_2`:
  `kind="random_braking_cycles"`, seeds `[101,102,103]`,
  `initial_speed_range=[16,20] m/s`,
  `upper_target_speed_range=[18,24] m/s`,
  `post_brake_speed_range=[10,18] m/s`,
  `minimum_speed_drop=3 m/s`,
  `acceleration_magnitude_range=[0.4,1.2] m/s^2`,
  `braking_magnitude_range=[0.8,2.2] m/s^2`,
  `hold_duration_range=[1.0,3.0] s`,
  `post_brake_hold_duration_range=[0.6,1.6] s`,
  `speed_floor=6 m/s`, `speed_ceiling=26 m/s`;
- `multipulse_mild`: `kind="multi_pulse_braking"`,
  `initial_speed=18.0`, `pulse_starts=[2.5,6.5,10.5] s`,
  `brake_delta_v=[2.0,3.0,2.5] m/s`,
  `brake_duration_range=[1.0,1.4] s`,
  `recovery_duration_range=[1.5,2.4] s`;
- `multipulse_varied`: `kind="multi_pulse_braking"`,
  `initial_speed=18.0`, `pulse_starts=[2.0,5.8,11.0] s`,
  `brake_delta_v=[2.5,4.0,3.0] m/s`,
  `brake_duration_range=[1.0,1.4] s`,
  `recovery_duration_range=[1.5,2.4] s`;
- `chirp_low`: `kind="chirp_sinusoidal"`, `base_speed=18.0`,
  `sinusoid_amplitude=1.5`, `period_start=10.0`,
  `period_end=4.0`, `speed_floor=8.0`, `speed_ceiling=26.0`;
- `chirp_high`: `kind="chirp_sinusoidal"`, `base_speed=18.0`,
  `sinusoid_amplitude=2.5`, `period_start=10.0`,
  `period_end=4.0`, `speed_floor=8.0`, `speed_ceiling=26.0`;
- `mixed_regime`: `kind="mixed_regime"`, `base_speed=18.0`,
  regimes:
  `0.0-3.0 s` constant-speed approach,
  `3.0-5.0 s` mild braking pulse,
  `5.0-7.0 s` recovery and hold,
  `7.0-11.0 s` low-amplitude sinusoidal disturbance,
  `11.0-13.0 s` stronger bounded braking pulse,
  `13.0-16.0 s` recovery/constant-speed finish, with
  `first_brake_delta_v=2.5 m/s`, `first_brake_duration=1.4 s`,
  `first_recovery_duration=1.6 s`,
  `sinusoid_amplitude=1.2 m/s`, `sinusoid_period=3.5 s`,
  `second_brake_delta_v=4.0 m/s`,
  `second_brake_duration=1.2 s`,
  `second_recovery_duration=2.0 s`,
  `speed_floor=8.0 m/s`, `speed_ceiling=24.0 m/s`.

Held-out reporting scenarios:

- `constant_18`: `kind="constant"`, `initial_speed=18.0`;
- `brake_shifted`: `kind="braking_recovery"`, `initial_speed=18.0`,
  `brake_delta_v=4.0`, `brake_start=5.0`, `brake_duration=2.0`,
  `recovery_duration=3.0`;
- `sinusoidal_period6`: `kind="sinusoidal"`, `initial_speed=18.0`,
  `sinusoid_amplitude=2.0`, `sinusoid_period=6.0`;
- `heldout_random_brake_0`, `heldout_random_brake_1`:
  `kind="random_braking_cycles"`, seeds `[201,202]`,
  `initial_speed_range=[15,21] m/s`,
  `upper_target_speed_range=[19,25] m/s`,
  `post_brake_speed_range=[9,17] m/s`,
  `minimum_speed_drop=4 m/s`,
  `acceleration_magnitude_range=[0.5,1.4] m/s^2`,
  `braking_magnitude_range=[1.0,2.5] m/s^2`,
  `hold_duration_range=[0.8,2.5] s`,
  `post_brake_hold_duration_range=[0.5,1.4] s`,
  `speed_floor=6 m/s`, `speed_ceiling=27 m/s`;
- `heldout_multipulse_shifted`: `kind="multi_pulse_braking"`,
  `initial_speed=18.0`, `pulse_starts=[3.0,7.2,12.0] s`,
  `brake_delta_v=[3.0,3.5,4.0] m/s`,
  `brake_duration=[1.2,1.2,1.5] s`,
  `recovery_duration=[1.8,2.0,2.2] s`;
- `heldout_chirp_shifted`: `kind="chirp_sinusoidal"`,
  `base_speed=18.0`, `sinusoid_amplitude=2.0`,
  `period_start=8.0`, `period_end=3.5`,
  `speed_floor=8.0`, `speed_ceiling=26.0`;
- `heldout_mixed_regime_shifted`: `kind="mixed_regime"`,
  `base_speed=19.0`,
  `first_brake_delta_v=3.0 m/s`, `first_brake_duration=1.2 s`,
  `first_recovery_duration=1.8 s`,
  `sinusoid_amplitude=1.5 m/s`, `sinusoid_period=3.0 s`,
  `second_brake_delta_v=3.5 m/s`,
  `second_brake_duration=1.4 s`,
  `second_recovery_duration=2.2 s`,
  `speed_floor=8.0 m/s`, `speed_ceiling=25.0 m/s`.

Diagnostics must consume scenario lists rather than hard-code these families
inside controller, objective, or gradient-mode logic.

The amended scenario set contains `14` diagnostic scenarios and `8` held-out
reporting scenarios.

## Gradient Modes

Use horizons `K=[1,3,6,10,T]` with `T=80`.

Let `retain_start = max(0, T_steps - K)`.

Implementation contract:

- all modes construct identical forward trajectories and objective values;
- `K=T` performs no temporal state detachment and must match no-detach full
  autograd;
- for truncated `K`, detach recurrent state passed to later transitions before
  the retained window;
- compute local objective contributions inside the loop before detaching
  recurrent state, so pre-window same-step local gradients are retained;
- store detached/reporting copies as needed for trajectories and summaries;
- do not detach controller parameters, leader tensors, or constants.

This preserves whole-rollout objective semantics while truncating only future
temporal credit assignment through simulator state recurrence.

## One-Step Protocol

For each input parameterization, initialization, scenario set, and gradient
mode:

```text
g_K = grad_beta J(beta)
d_K = -g_K / (||g_K||_2 + 1e-12)
beta' = beta + alpha * d_K
```

Alpha grid:

```text
[0.001, 0.002, 0.004, 0.01, 0.02, 0.04, 0.1, 0.3, 1.0]
```

Primary alpha: `0.1`.

Random controls:

- smoke tests: 8 random directions;
- diagnostic run: 32 random directions;
- use one base seed, `random_direction_seed=10000`, and deterministic
  per-initialization/per-parameterization sequence derivation;
- serialize generated random direction vectors in summary output;
- normalize in the same raw beta parameter space as simulator gradients.

Zero gradients:

- if `||g|| <= 1e-12`, treat the update as no-op with `improved=false`;
- include it in improvement-probability denominators;
- report zero-gradient count, proportion, and gradient-norm distribution by
  mode and input parameterization.

Unsafe or non-finite candidate updates:

- treat as non-improvements at that alpha;
- include them in denominators;
- report counts by mode, alpha, initialization, scenario aggregate, and input
  parameterization;
- do not shrink alpha adaptively.

Evaluation:

- use the unchanged forward simulator and same objective;
- run held-out reporting under `torch.no_grad()`;
- held-out results are reporting-only, not selection or tuning.

"Beats random" rule:

- descriptive only, no significance claim;
- a simulator-gradient mode beats random if its primary-alpha improvement
  probability is strictly greater than the random-direction improvement
  probability aggregated over the matched diagnostic set;
- report exact counts, probabilities, and relative objective changes;
- treat small margins as weak evidence in the report.

Gradient cosine similarity:

- for each initialization and input parameterization, compute cosine similarity
  between each truncated gradient and the full-gradient direction:
  `cos(g_K, g_T)` for `K in [1,3,6,10]`;
- compute cosine in raw beta space before direction normalization;
- report undefined cosine values explicitly if either compared gradient norm is
  zero;
- cosine similarities are diagnostic summaries only and must not alter the
  one-step update rule.

## Phase D Files

Expected files to create:

- `src/differential_sim/controllers.py`: structured controller, input
  normalization, beta initialization, flattening helpers.
- `src/differential_sim/objectives.py`: objective components, total objective,
  component reports, configurable weights, semantic-weight diagnostics.
- `src/differential_sim/gradient_modes.py`: controller rollout with temporal
  detachment and forward-equivalence helpers.
- `src/differential_sim/diagnostics.py`: one-step descent, random directions,
  aggregation, zero/non-finite handling, cosine similarity, normalized and
  SI-unit diagnostics.
- `scripts/run_milestone1.py`: reproducible diagnostic run and summaries.
- `tests/test_controller.py`: bounds, initialization, normalization.
- `tests/test_objectives.py`: objective components, finite values, jerk
  convention, configurable weights, held-out isolation.
- `tests/test_gradient_modes.py`: forward equivalence, finite-difference full
  gradient, detachment, `K=T`, `K=1` local-dependence checks.
- `tests/test_diagnostics.py`: one-step normalization, random controls,
  zero/non-finite handling, output separation.

Expected files to modify only if needed:

- `src/differential_sim/scenarios.py` for approved leader-profile families;
- `src/differential_sim/__init__.py`;
- `pyproject.toml` for pytest configuration only.

Do not add dependencies without an approved plan amendment.

## Tests

Required validation:

1. Existing Milestone 0 tests still pass.
2. Controller outputs remain within `[0.5, 3.0]` for representative and extreme
   inputs.
3. Forward trajectories and objective values are identical across
   `K=1,3,6,10,T` for identical inputs.
4. `K=T` gradient agrees with no-detach full autograd.
5. Full gradient passes central finite-difference directional check on a small
   deterministic case using `torch.float64` and epsilons `1e-4`, `3e-5`,
   `1e-5`; accept relative error `<1e-3` for a stable epsilon or absolute error
   `<1e-6` for small derivatives.
6. Detachment tests confirm only intended temporal dependencies are cut.
7. `K=1` retains local same-step objective gradients while removing earlier
   future-state credit assignment.
8. Held-out evaluation runs under `torch.no_grad()` and does not affect
   gradients, normalization, or selection.
9. Random directions use the same raw beta-space L2 normalization as simulator
   gradients.
10. Normalized and SI-unit diagnostics are separated in output and summary.
11. Scenario generation is deterministic for all approved profile families and
    serializes any sampled random-braking parameters.
12. Cosine similarity reports match direct raw-beta-space calculations and
    handle zero-gradient cases explicitly.

## Commands

Use the project conda environment:

```bash
/home/zpz/miniconda3/envs/differential_sim/bin/python -m pytest -p no:cacheprovider
/home/zpz/miniconda3/envs/differential_sim/bin/python -m pytest -p no:cacheprovider tests/test_gradient_modes.py
/home/zpz/miniconda3/envs/differential_sim/bin/python -m pytest -p no:cacheprovider tests/test_diagnostics.py
/home/zpz/miniconda3/envs/differential_sim/bin/python scripts/run_milestone1.py
```

The full diagnostic experiment is a Phase D/E validation command, not a
planning command.

## Result Artifacts

Write under `reports/milestone1/`:

- `normalized/diagnostics.jsonl`;
- `normalized/summary.json`;
- `normalized/summary.md`;
- `si_units/diagnostics.jsonl`;
- `si_units/summary.json`;
- `si_units/summary.md`;
- top-level `summary.md` labeling normalized-input results as primary and
  SI-unit results as secondary validation.

Rows should include at least:

- seed and generated beta/random direction identifiers;
- input parameterization;
- mode and `K`;
- initialization id;
- direction type;
- alpha;
- objective before/after;
- relative objective change;
- improvement flag;
- total and component objectives before/after;
- gradient norm;
- gradient cosine similarity to full gradient where applicable;
- finite/zero-gradient flags;
- unsafe/non-finite update flag;
- runtime;
- minimum gap and speed;
- scenario aggregate metadata.

Human-readable summaries must include all-alpha results, not only primary-alpha
tables. Primary alpha remains the descriptive "beats random" comparison point.

## Acceptance Criteria

Milestone 1 passes only if:

1. Required tests pass without weakening tolerances.
2. All gradient modes have identical forward trajectories and objective values.
3. Detachment tests verify the intended graph cuts.
4. Full gradient finite-difference check passes.
5. Controller bounds hold.
6. Held-out scenarios are isolated from gradient construction, normalization,
   and selection.
7. Random directions use matched parameter-space normalization.
8. Both normalized and SI-unit diagnostics run, or SI-unit validation is
   explicitly reported as blocked by a numerical issue.
9. At least one simulator-gradient mode beats random directions under the
   descriptive primary-alpha rule, or failure is reported and Milestone 2 is not
   started.
10. Reports include total/component objectives, unweighted magnitudes, weighted
    contributions under `progress=1.0`, `safety=0.7`, `jerk=5.0`,
    diagnostic-only semantic weights, improvement probabilities, relative
    changes, gradient norms, cosine similarities to full gradient,
    zero-gradient metrics, unsafe/non-finite counts, runtime, seeds, scenario
    configs, dependency versions, and fixed rollout/controller settings.
11. Human-readable summaries include an interpretation tied to H1 Stage 1 and
    state whether richer scenarios preserve, weaken, or overturn the previous
    elementary-scenario observation.

## Stop Conditions

Stop Phase D and request a plan amendment if:

- implementing truncation would change forward values;
- local same-step gradients cannot be retained while detaching recurrent state;
- Milestone 0 smooth-clamped semantics cannot be preserved;
- approved scenarios or alpha grid produce pervasive non-finite states such that the
  diagnostic cannot run as specified;
- objective components are locally insensitive to controller parameters across
  all modes and initializations;
- full finite-difference check fails after implementation errors are ruled out;
- any change would broaden the task into iterative optimization, model training,
  simulator redesign, or scenario-family development.

## Deferred Items

- Milestone 2 iterative structured optimization.
- Milestone 3 MLP training.
- CPU or CUDA memory study.
- Sawtooth/noisy emergency-brake stress profiles.
- Any further objective reweighting based on diagnostic-only semantic weights.
- Any change to Milestone 0 simulator semantics.

## Amendment History

- 2026-06-22: Phase C amendment expands scenario diversity, confirms objective
  weights `progress=1.0`, `safety=0.7`, `jerk=5.0`, adds gradient cosine
  similarity reporting, requires richer human-readable interpretation, and
  requires revised Phase E to rerun both normalized and SI-unit diagnostics.
- 2026-06-22: Phase D/E feedback amendment expands initialization centers to
  `T_init=[0.9,1.2,1.4,1.6,1.9,2.2]` with seeds `0..5`, and requires
  human-readable all-alpha result reporting.

## Superseded Draft

This approved plan supersedes the discussion draft:

- `temp_content/temp_plan/milestone_1_phase_a_proposal.md`

## Phase D Instructions

When implementation begins in a separate task:

1. Re-read `AGENTS.md`, `PROJECT_CONTEXT.md`, `PLANS.md`, Milestone 0 plan, and
   this file.
2. Implement only this approved scope.
3. Do not begin Milestone 2.
4. Run the required tests and diagnostic command.
5. Report changed files, commands, quantitative results, acceptance criteria
   status item by item, deviations, assumptions, unresolved risks, and stop.
