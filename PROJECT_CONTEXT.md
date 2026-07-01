# PROJECT_CONTEXT.md

## 1. Research motivation

Differentiable traffic simulators are commonly validated through reconstruction,
calibration, or direct optimization. A smaller set of studies has also trained
learned controllers through differentiable simulation. What remains unclear is
which properties of a simulator gradient make it useful, neutral, or harmful for
downstream optimization and model training.

This project does not initially build a new traffic simulator. It starts from a
simple, reconstruction-friendly differentiable Intelligent Driver Model (IDM)
and studies one controlled factor: temporal gradient horizon.

## 2. Primary research question

> How much temporal simulator gradient is required for useful downstream
> optimization and small-model training in a deterministic IDM task?

The project compares gradients that use the same forward trajectory but different
backward temporal connectivity.

Let:

- `T` be the full rollout length;
- `K` be the number of final time steps retained in the backward graph;
- `K = T` mean full backpropagation through the rollout;
- smaller `K` mean earlier simulator states are detached from the backward graph.

Initial gradient modes:

- `K = 1`: immediate/local temporal gradient;
- `K = 10`: short-horizon gradient;
- `K = T`: full temporal gradient.

The exact numeric horizon is a default and may be adjusted only after the basic
rollout length is fixed. The conceptual comparison—local, short, and full—must
remain unchanged.

## 3. Main hypotheses

### H1 — local utility predicts direct optimization

A gradient mode with a higher probability of producing one-step objective
improvement will tend to perform better during iterative optimization of a small,
structured controller.

### H2 — optimization utility transfers to model training

The gradient mode that performs best, or remains competitive, during direct
structured optimization will also be useful for training a small state-dependent
model on held-out traffic scenarios.

These are hypotheses to test. Failure of either hypothesis is scientifically
informative.

### H3 — sparse long-horizon gradients can preserve useful temporal signal

After H1 and H2, the next methodological question is whether the useful
long-horizon signal from full temporal gradients can be retained with lower
backward temporal resolution.

A sparse long-horizon gradient keeps the forward rollout and objective at the
original time resolution, but exposes only selected checkpoint-to-checkpoint
dependencies in the backward graph. This is a hypothesis about gradient
construction, not a proposal to simulate fewer forward steps.

### H4 — hybrid dense-short and sparse-long gradients can combine local and long-range utility

If sparse long-horizon gradients retain useful signal, a later hybrid method may
combine dense full-resolution gradients over a recent short window with sparse
longer-range checkpoint gradients. This tests whether short-term local effects
and long-term bias-correction effects can be represented at different backward
resolutions.

H3 and H4 are hypotheses, not assumed conclusions. They require held-out
downstream validation and must preserve the same forward values as the dense
baseline.

## 4. Intended contribution

The intended contribution is not:

- another differentiable IDM implementation;
- proof that differentiable simulation can fit trajectories;
- proof that a neural controller can be trained through a simulator;
- a broad benchmark covering all differentiable traffic methods.

The intended contribution is a controlled empirical link between:

1. local gradient utility;
2. iterative downstream optimization;
3. small-model training utility.

A useful result may show that:

- full gradients are consistently superior;
- moderate truncation is equally useful at lower cost;
- long temporal gradients reduce utility;
- local descent predicts optimization but not model training;
- reconstruction-valid gradients do not transfer to the downstream objective.

## 5. Why IDM is selected

The initial IDM setting avoids the structural difficulties that would confound
the study:

- continuous state updates;
- fixed interaction topology;
- fixed leader identity;
- deterministic dynamics;
- no route or lane-choice sampling;
- no branching over vehicle generation or signal phases;
- low-dimensional, physically interpretable states and parameters;
- easy construction of deterministic synthetic ground truth.

This allows the experiment to isolate temporal gradient usefulness rather than
arbitrary branching, stochastic estimators, path explosion, or multi-agent
combinatorics.

## 6. Initial scenario

Use a one-lane leader–follower system.

The leader trajectory is exogenous and is not differentiated. Start with one
leader and one follower. A short follower platoon is deferred until the basic
result is stable.

Leader-profile families may include:

- constant speed;
- one braking pulse;
- braking followed by recovery;
- sinusoidal disturbance;
- stop-and-go disturbance.

Each generated scenario is deterministic after its profile parameters are fixed.

Training and evaluation profiles must differ by profile parameters and should
eventually include held-out profile shapes or disturbance regimes.

## 7. Simulator state and update

A minimal follower state contains:

- position `x_t`;
- speed `v_t`;
- acceleration `a_t`;
- gap to leader `s_t`;
- relative speed `Δv_t`.

The follower acceleration is computed by IDM:

`a_t = F_IDM(v_t, Δv_t, s_t, T_t, fixed_IDM_parameters)`

where `T_t` is the controller-selected desired time headway.

The initial integration scheme is explicit and fixed across all experiments:

- update speed from acceleration;
- update position from speed;
- recompute gap and relative speed from the leader trajectory.

The exact update ordering and units must be documented and tested. It must not
change between gradient modes.

## 8. Downstream controller

### First downstream model: structured controller

Use a bounded, interpretable controller:

`T_t = T_min + (T_max - T_min) * sigmoid(
    β0 + β1 * v_t + β2 * Δv_t + β3 * s_t
)`

where:

- `β0...β3` are trainable parameters;
- `T_min` and `T_max` are fixed physical bounds;
- inputs should be normalized using statistics fixed independently of gradient
  mode.

This four-parameter model is the first downstream optimization object.

### Later model: small MLP

Only after the structured controller works, replace it with a one-hidden-layer
MLP using the same inputs and bounded output. Use approximately 16–32 hidden
units. Compare only the most informative gradient modes from the earlier stage.

## 9. Objective

The task balances progress, safety, and comfort:

`J = mean_t [
    w_v * progress_error_t
  + w_s * safety_penalty_t
  + w_j * jerk_penalty_t
]`

Recommended forms:

- progress error: squared relative-speed or desired-speed error;
- safety penalty: squared `softplus(s_safe(v_t) - s_t)`;
- jerk penalty: squared acceleration difference.

Requirements:

1. The objective is identical across gradient modes.
2. Loss components are normalized once using a documented reference procedure.
3. Weights are fixed before comparing gradient modes.
4. No gradient mode receives separately tuned weights.
5. Final evaluation reports individual components as well as the total objective.

The mainline objective was fixed before Milestone 1 comparison:

```text
progress_weight = 1.0
safety_weight = 0.7
jerk_weight = 5.0
```

The implemented normalized component scales and safe-gap definition are
documented in `src/differential_sim/objectives.py` and the closed milestone
plans. They must remain identical across gradient horizons. Any objective
reweighting is a separate scientific sensitivity study and requires approval.

## 10. Primary measurements

### Stage 1 primary diagnostic: one-step descent

For controller parameters `φ` and gradient estimate `g_K`:

`φ' = φ - α * g_K / (||g_K|| + ε)`

Evaluate the same forward objective at `φ'`.

Primary outputs:

- probability of objective improvement;
- relative objective change;
- gradient norm;
- runtime and peak memory.

Use a small shared set of normalized step sizes. Random normalized directions are
the negative control.

A central finite-difference directional derivative is a correctness check on a
small subset, not the main experiment.

### Stage 2: direct optimization

Run the same optimizer, initialization, scenarios, objective, and update budget
for each gradient horizon. Evaluate on held-out leader profiles.

After Milestone 1 Phase F, the structured-controller direct optimization path
uses normalized controller inputs only. The SI-unit Milestone 1 validation was
weak at the primary alpha and should be treated as parameterization-sensitivity
evidence, not as a Milestone 2 branch, unless a later approved plan opens a
separate parameterization study.

Milestone 2 closed as passing on 2026-06-24. Under the approved shared Adam
procedure, the held-out ranking exactly matched Milestone 1:

`K=80 > K=10 > K=6 > K=3 > K=1`

The descriptive Spearman rank correlation was `1.0`. Full gradient `K=80` was
best, and `K=10` was the best non-full horizon. H1 is supported for the fixed
structured-controller experiment.

The closed result also shows that truncated modes converge reproducibly to
horizon-specific inferior controllers. The most supported interpretation is
structural bias from omitted recurrent state-mediated temporal sensitivity,
not numerical failure or insufficient updates. More scenario samples may
reduce variance but do not generally restore omitted temporal derivatives.

### Stage 3: small-model training

Train the same small MLP with selected gradient modes. Compare held-out objective,
individual loss components, convergence, and stability.

Milestone 3 base and aggressive objective-weight sensitivity runs support H2
for the approved deterministic small-MLP experiments. In both objective
profiles, the held-out ranking was:

`K=80 > K=50 > K=35 > K=20 > K=10 > K=6`

The full temporal gradient remained best by held-out median, `K=50` was the
best truncated horizon, and the short horizons remained substantially weaker.
The aggressive objective profile changed the magnitude and shape of the
horizon-response curve but not the qualitative ranking.

The result supports moving from "does temporal horizon matter?" to "can useful
long-horizon temporal information be preserved more efficiently?"

### Stage SG1: sparse long-horizon gradient resolution

SG1 tests H3. It should preserve the exact full-resolution forward rollout and
objective while lowering only the backward temporal resolution through
checkpoint-level sparse gradient connectivity.

Conceptually:

- run the same full-resolution forward rollout;
- record checkpoint states and span losses;
- construct a sparse checkpoint-to-checkpoint backward surrogate;
- compare sparse full-horizon gradients against dense `K=80`, dense `K=50`,
  and dense `K=10` baselines;
- use gradient-only admission before full training;
- evaluate downstream held-out utility, component tradeoffs, gradient alignment,
  runtime, and memory.

SG1 should use `T=80` as the primary comparison because it preserves
comparability with Milestone 3. Sparse truncated gradients such as `K=50` with
sparse stride, and longer `T=160` stress tests, are conditional add-ons after
sparse full-horizon behavior at `T=80` is informative.

Open SG1 decisions include span sensitivity computation, checkpoint state
definition, stride values, admission criteria, optimizer/LR policy, budget,
validation tolerances, and exact reporting layout.

### Stage SG2: hybrid dense-short plus sparse-long gradients

SG2 tests H4 and should start only if SG1 shows that sparse long-horizon
connectivity carries useful signal.

Conceptually:

- use dense full-resolution gradients for a recent short temporal window;
- use sparse checkpoint gradients for older long-range dependencies;
- combine them by a predeclared objective-consistent rule;
- compare against dense `K=80`, dense `K=50`, dense `K=10`, and the SG1 sparse
  full-horizon method.

The default scientific intent is to test whether local short-term effects and
long-range bias-correction effects can be represented at different backward
resolutions without changing forward values. Tuned blend weights or per-method
objectives would be separate scientific choices and require approval.

## 11. Scientific controls

To make the comparison interpretable:

- same forward rollout for every gradient horizon;
- same initial controller parameters;
- same scenario order and batches;
- same optimizer and learning-rate policy;
- same number of updates and objective evaluations;
- same parameter bounds;
- same train/evaluation split;
- same numerical precision unless explicitly studied;
- if CUDA/GPU execution is used, CPU/GPU parity must be checked within approved
  tolerances before GPU results are treated as main evidence;
- device, dtype, PyTorch, CUDA availability, and GPU metadata must be recorded
  in result artifacts when device choice is part of the experiment;
- report failures and unsafe trajectories instead of silently clipping them.

## 12. Deferred topics

The following are outside the initial mainline:

- lane changes;
- changing leader identity;
- stochastic arrivals or route choice;
- traffic signals;
- SUMO;
- reinforcement learning or PPO;
- broad gradient-corruption studies;
- cross-simulator correction;
- amortized IDM parameter inference;
- hard-simulator transfer;
- probabilistic driver populations;
- large multi-agent networks;
- broad benchmark packaging.

They may be revisited only after the main hypotheses or approved sparse-gradient
follow-ups have been tested. Fixed-order multi-follower platoons, hard-simulator
validation, and matched sample-based comparisons are recognized as plausible
later methodological studies, but they are not SG1 or SG2 scope unless a later
approved plan says otherwise.


## 13. Final Notes 
If the user override the project defaults in interactive session, report the override and accept the user override by default.
