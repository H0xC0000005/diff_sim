# SG1: Sparse Long-Horizon Gradient Resolution

Status: Approved

Approval date: 2026-07-01

## Authority And Scope

This is the authoritative Phase C plan for SG1 implementation and validation.
It freezes the accepted Phase A/B proposal for sparse long-horizon gradients on
the Milestone 3 bounded-MLP IDM task.

SG1 tests H3: whether lower-resolution checkpoint-level backward connectivity
can preserve useful long-horizon temporal gradient signal while the forward
rollout, objective values, scenarios, model, initialization, optimizer policy,
and held-out evaluation remain unchanged.

Implementation must occur in a separate Phase D task. This Phase C task does
not authorize implementation, experiments, or SG2 work.

## Research Question

> Can sparse full-horizon checkpoint gradients preserve useful long-horizon
> temporal credit for the Milestone 3 MLP task while keeping the exact
> full-resolution `T=80` forward IDM rollout and objective unchanged?

SG1 may support, contradict, or leave unresolved H3. No downstream result is
assumed.

## Fixed Scientific Semantics

SG1 must preserve the Milestone 3 base setting:

- deterministic one-lane, one-leader/one-follower simulation;
- fixed vehicle ordering and leader identity;
- rollout length `T=80` and step size `dt=0.2 s`;
- existing smooth-clamped `diffidm==0.0.3` acceleration semantics;
- existing speed-then-position update ordering;
- fixed IDM parameters other than model-selected desired headway;
- width-16 bounded MLP, normalized inputs, and headway bounds `(0.5,3.0)` s;
- the approved 14 training and 8 held-out scenarios;
- equal scenario weighting within each split;
- base objective weights only:

  ```text
  progress_weight = 1.0
  safety_weight = 0.7
  jerk_weight = 5.0
  ```

- Milestone 3 model seeds:

  ```text
  [1000,1001,1002,1003,1004,1005]
  ```

- Milestone 3 base optimizer policy for the first SG1 pass:

  ```text
  Adam, lr = 0.001, budget = 1200 updates
  ```

- `torch.float64` for validation and main evidence;
- held-out evaluation under `torch.no_grad()`;
- no held-out use for training, normalization, calibration, stopping, or
  checkpoint selection.

Every compared method must use identical forward values at identical model
parameters. Sparse gradients may alter only backward graph connectivity and
the backward sensitivity approximation.

## Approved Methods

Dense references must be rerun in the SG1 result group with the SG1 runner and
artifact layout:

```text
dense K=80
dense K=50
dense K=10
```

The sparse full-horizon methods are:

```text
sparse full horizon, m=2
sparse full horizon, m=4
sparse full horizon, m=6
sparse full horizon, m=8
```

`K=80` is the dense full-gradient reference. `K=50` is the strongest completed
Milestone 3 dense truncated reference. `K=10` is the short-horizon weak
reference. Sparse methods use the full `T=80` forward rollout and lower only
backward temporal resolution.

## Explicit Non-Goals

- No SG2 hybrid dense-short plus sparse-long method.
- No sparse truncated gradients such as sparse `K=50`.
- No `T=160` experiment.
- No aggressive-objective SG1 run.
- No learned, tuned, decreasing, or continuously varying gradient-resolution
  schedule.
- No per-method LR, optimizer, budget, scheduler, clipping, regularization,
  recovery, early stopping, or checkpoint selection.
- No per-method objective or loss-weight tuning.
- No change to simulator equations, integration order, scenario split,
  normalization, model architecture, or headway bounds.
- No stochastic mini-batching, scenario resampling, curriculum, or online data
  generation.
- No new traffic setting, platoon, lane changing, route choice, traffic signal,
  SUMO integration, policy gradient, RL, hard-simulator transfer, or
  derivative-free baseline.
- No SG2 planning or implementation.

Sparse truncated gradients and `T=160` are deferred for both SG1 and SG2. They
may be opened only by a later explicit user decision after SG1 evidence exists.

## Sparse Gradient Definition

### Checkpoint State

Use the existing minimal recurrent state:

```text
z_t = [x_t, v_t]
shape [S,2]
dtype torch.float64
```

`gap_t`, `delta_v_t`, model inputs, acceleration, and losses are derived from
`z_t`, the exogenous leader trajectory, and the existing simulator/objective
code. They are not additional recurrent checkpoint state.

### Span Function

For a complete sparse span `[a,b]`, define:

```text
(z_b, L_ab) = F_ab(z_a, theta; leader_ab, fixed_config)
```

where:

- `z_a = [x_a, v_a]`, shape `[S,2]`;
- `z_b = [x_b, v_b]`, shape `[S,2]`;
- `L_ab` is the span contribution to the same global scenario/time mean used by
  the dense objective;
- `theta` is the flattened 81-parameter MLP state;
- `leader_ab` and simulator/objective settings are fixed constants.

The formal sparse method uses sparse VJP/adjoint accumulation over checkpoint
spans. Parameter gradients must be collected for one scalar sparse objective,
followed by one Adam update. Do not update parameters span by span, checkpoint
by checkpoint, or coordinate by coordinate.

### B1 Anchored Macro-Step Surrogate

The approved span approximation is B1, the anchored macro-step surrogate.

For each complete span with stride `m`, let:

```text
Delta = m * dt
z_a_star = exact full-resolution checkpoint state at a
z_b_star = exact full-resolution checkpoint state at b
L_ab_star = exact full-resolution span loss contribution
```

Construct one differentiable coarse transition `G_m` using the same controller,
IDM acceleration formula, and speed-then-position update order, but with
`Delta` instead of `dt`:

```text
headway_a = model(features(z_a, leader_a), theta)
accel_a = IDM(z_a, leader_a, headway_a, fixed_IDM_parameters)
v_b_macro = v_a + Delta * accel_a
x_b_macro = x_a + Delta * v_b_macro
G_m(z_a, theta) = [x_b_macro, v_b_macro]
```

`leader_a` is the exogenous leader state at the span start. Loss sensitivities
use a single coarse proxy `Q_m` evaluated from the same macro transition and
the existing objective component formulas, weighted as the span's contribution
to the global dense scenario/time mean.

Anchor the sparse values to the exact full-resolution forward rollout:

```text
z_b_sparse =
    stopgrad(z_b_star)
  + G_m(z_a, theta)
  - stopgrad(G_m(z_a_star, theta_star))

L_ab_sparse =
    stopgrad(L_ab_star)
  + Q_m(z_a, theta)
  - stopgrad(Q_m(z_a_star, theta_star))
```

`stopgrad` means `.detach()` or an equivalent boundary that cuts gradient flow.
The anchor preserves exact forward endpoint and objective values while exposing
only the approved coarse span derivative in the backward graph.

Exact span VJP through all internal micro-steps is allowed only as a
correctness/reference fallback. It is not the formal SG1 sparse method. If B1
cannot be implemented soundly, stop for a plan amendment before substituting a
different formal method.

### `m=6` Remainder

For `m=6`, use only complete spans:

```text
[0,6], [6,12], ..., [72,78]
```

Do not create `[78,80]`, do not merge `[78,80]` into the previous span, and do
not add a special short-span VJP. The exact forward rollout and dense objective
still cover all `T=80` steps. The final two-step remainder contributes to the
reported forward objective value as a stop-gradient constant and has no sparse
recurrent checkpoint link.

The implementation report must explicitly state this treatment.

### Parameter-Gradient Accumulation

Use an autograd-backed sparse surrogate first:

1. run the exact full-resolution forward rollout;
2. build all approved sparse span anchors;
3. aggregate one scalar sparse objective;
4. call `.backward()` once;
5. apply one Adam update.

The implementation must keep the code boundary between span sensitivity
construction and sparse surrogate accumulation clear. If this route fails for
an implementation reason without changing scientific meaning, manual adjoint
accumulation may be proposed as an amendment, but do not switch silently.

## Stage 0 Admission Diagnostics

Run Stage 0 before sparse training. Use:

- all six canonical Milestone 3 model seeds;
- all 14 training scenarios;
- update-0 model states only;
- dense `K=80`, dense `K=50`, dense `K=10`, and all four sparse strides;
- no held-out evaluation.

For every method and seed, report:

- forward objective and component values;
- maximum absolute trajectory difference from dense `K=80`;
- finite objective and gradient flag;
- raw flattened gradient norm;
- cosine to dense `K=80`;
- cosine to dense `K=50`;
- one-step normalized descent utility;
- runtime and memory where available.

Use the normalized one-step candidate:

```text
theta' = theta - 0.001 * g / (||g||_2 + 1e-12)
```

Evaluate `theta'` on the same 14 training scenarios only.

Each sparse stride is admitted to Stage 1 if all of the following hold:

1. forward trajectories and objective values are equivalent to dense `K=80`
   within the validation tolerances below;
2. all six sparse gradients are finite;
3. all six sparse gradient norms are greater than `1e-12`;
4. either:
   - its median one-step relative training-objective change is no worse than
     dense `K=10` by more than `0.01` absolute relative-change units; or
   - its median cosine to dense `K=80` is greater than the median cosine of
     dense `K=10` to dense `K=80`.

Do not require a sparse method to beat dense `K=50` in Stage 0. If no sparse
stride is admitted, stop after the Stage 0 report and do not run Stage 1.

## Stage 1 Training Comparison

Run Stage 1 only for admitted sparse strides plus the dense references.

For every `(seed, method)`:

1. load the canonical Milestone 3 initial MLP state for that seed;
2. construct a fresh Adam optimizer with `lr=0.001`;
3. run exactly 1200 updates unless the run fails;
4. use all 14 training scenarios per update;
5. call backward once per update;
6. apply one optimizer step per update;
7. evaluate held-out metrics only under `torch.no_grad()`.

Dense references must be executed in the SG1 result folder rather than only
cited from old Milestone 3 artifacts. The SG1 report may also compare against
completed Milestone 3 summaries as context, but the primary SG1 tables use the
locally rerun dense references.

## Logging And Artifacts

Use this result root:

```text
reports/sg1_sparse_gradient/
```

Required machine-readable artifacts:

- `config.json`;
- `environment.json`;
- `stage0_admission.csv`;
- `stage0_gradients.jsonl`;
- `training_metrics.jsonl`;
- `heldout_metrics.jsonl`;
- `gradient_diagnostics.jsonl`;
- `failures.jsonl`;
- initial and final model state files for each run;
- a manifest listing all artifact files and hashes.

Required human-readable artifacts:

- `summary.md`;
- `observations.md`;
- `acceptance.md`.

At minimum, log aggregate training metrics every update. Log held-out aggregate
and per-scenario metrics at update 0, every 30 updates, and update 1200. The
final update must be logged even if a future amendment changes the budget to a
number not divisible by 30.

The report must include:

- final held-out total objective and progress/safety/jerk components;
- final training total objective and components;
- per-seed results;
- median, mean, minimum, maximum, and IQR by method;
- paired differences against dense `K=80`, dense `K=50`, and dense `K=10`;
- convergence curves or tabular convergence summaries;
- gradient norms and cosine diagnostics;
- failure counts and failure reasons;
- runtime and memory where available;
- explicit statement on whether SG1 supports, contradicts, or leaves unresolved
  H3.

## Validation Requirements

Add focused SG1 tests before the main run.

Required tests:

1. Sparse and dense methods produce identical forward trajectories and
   objective values at identical parameters.
2. The `m=6` non-divisor case preserves full forward objective values while
   using no `[78,80]` sparse checkpoint link.
3. Sparse gradients are finite and nonzero on deterministic admission cases.
4. Sparse objective normalization matches the dense global scenario/time mean.
5. Sparse backward connectivity does not retain dense micro-step recurrent
   graph connectivity.
6. Dense `K=80` finite-difference directional check remains passing.
7. Dense-equivalent VJP check passes on `mixed_regime` and `brake_stronger`.
8. Held-out evaluation runs under `torch.no_grad()`.

Validation tolerances:

```text
trajectory max absolute difference:        <= 1e-10
objective/component absolute difference:   <= 1e-10
objective/component relative difference:   <= 1e-8
gradient finite check:                     all finite
nonzero gradient norm:                     > 1e-12
dense-equivalent VJP relative difference:  <= 0.01% of dense K=80 norm
held-out grad-enabled flag:                False
```

For sparse-connectivity tests, prove both:

- gradients can flow through approved checkpoint links; and
- no gradient path remains from a later sparse span to an internal
  full-resolution micro-step state that is not part of the approved B1
  macro-step surrogate.

Use `torch.float64` for numerical checks.

## Acceptance Criteria

SG1 Phase E passes only if:

1. all required validation tests pass without weakening tolerances;
2. Stage 0 is completed and reported for all dense references and all four
   sparse strides;
3. Stage 1 is run only for admitted sparse strides plus dense references, or
   Stage 1 is skipped because no sparse stride is admitted;
4. all compared methods use identical forward rollout, objective, scenarios,
   model initialization protocol, optimizer family, LR, update budget, dtype,
   device policy, and evaluation cadence;
5. held-out evaluation is isolated from training and all admission decisions;
6. dense SG1 references are present in the SG1 result folder;
7. artifacts are complete enough to reproduce the reported rankings and
   diagnostics;
8. the report states whether H3 is supported, contradicted, or unresolved.

Classify H3 as supported when at least one sparse full-horizon stride:

- passes Stage 0;
- completes Stage 1 with at most one failed seed;
- has negative median held-out relative change; and
- materially beats dense `K=10` while remaining operationally competitive with
  dense `K=50`.

Use `0.01` absolute relative-change units as the operational-equivalence band
for final held-out relative change. Material pairwise advantage requires at
least four of six paired-seed wins.

Classify H3 as contradicted when finite evidence is sufficient and all admitted
sparse strides are materially worse than dense `K=10` or fail to improve
held-out median while dense references improve.

Classify H3 as unresolved for excessive failures, no admitted sparse stride,
validation failure, held-out-isolation failure, or mixed evidence that does not
meet the support or contradiction criteria.

## Stop Conditions

Stop before continuing and request a plan amendment if:

- B1 cannot be implemented without differentiating internal micro-steps in the
  formal sparse method;
- any sparse mode changes forward trajectory or objective values;
- dense-equivalent VJP differs from dense `K=80` beyond tolerance;
- sparse-connectivity tests show unintended dense recurrent graph paths;
- all sparse gradients are zero, nonfinite, or rejected by Stage 0;
- more than one Stage 1 run fails for any method;
- held-out evaluation is found to affect training, admission, stopping, or
  checkpoint selection;
- implementation evidence requires changing objective weights, LR policy,
  budget, scenarios, model architecture, simulator semantics, or compared
  methods.

## Deferred Items

- SG2 hybrid dense-short plus sparse-long gradients.
- Sparse truncated gradients.
- `T=160` stress testing.
- Aggressive-objective sparse-gradient sensitivity run.
- Manual adjoint accumulation as the formal implementation.
- Exact internal span VJP as a formal sparse training method.
- B2 checkpoint-local first-order hold as a formal method.
- Any learned or tuned gradient-resolution schedule.

## Plans Superseded

This plan supersedes the SG1 Phase A/B draft:

```text
temp_content/temp_plan/sg1_sparse_gradient_phase_a_b_proposal.md
```

The aftermath discussion files under `temp_content/temp_impl/aftermath/` remain
historical discussion references only.
