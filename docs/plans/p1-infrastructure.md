# Plan: P-I infrastructure (queue item 3)

Author-approved queue of 4 Oct 2026; branch `quadrotor-sim`. Everything is built and
tested on synthetic problems. Nothing runs on quadrotor training data, and every P-I
stage that would (generate, train, analyse, hover) refuses to run unless the annotated
tag `prereg-p1` exists. The one exception is a runtime benchmark on random targets.

## Library (reusable, `lens/`, `control/`)

- `lens/models.py`: the two surrogate architectures, in float64 JAX with the weight
  layout (out, in), so the first layer is theory.md's E ∈ R^{H×k}.
  - (a) **Pre-norm residual block.** h = E z + b, then per block
    h ← h + W2 GELU(W1 (γ LN(h) + β) + b1) + b2, and output y = Wo h + bo.
    This is `lens/core.py`'s block structure, generalised to vector outputs.
  - (b) **TD-MPC2-style NormedLinear stack.** n NormedLinear layers (Linear → LN with
    affine → Mish), then a Linear to the output.
  - Shared settings: initialisations `zero_bias` and `torch_default` (the initial-lens
    check's conventions); ε = 1e-5; H = 128; 1 or 3 blocks.
  - Functions: `first_layer()` gives (E, b) for `lens/geometry.py`, and `trace()` gives
    each block's pre-affine LayerNorm output (for attenuation).
- `lens/train.py`: Adam with full batch by default, chunked with `lax.scan` like
  `core.train`.
  - Early stopping: patience 10% of max steps, tolerance 1% on held-out one-step MSE,
    restoring the best parameters.
  - Lens logging every `log_every` steps: z*, κ, widths, r*(d), r_eff(d), D/r_eff, S.
  - Seeds; history and metadata returned.
- `lens/analysis.py`, the analysis of trained models:
  - Jacobians by autodiff, and Jacobian error per state;
  - error against lens distance (10% nearest vs 50% farthest by ρ, or ρ_eff for a
    degenerate lens);
  - coverage and sharpness from the weights along any direction;
  - the Corollary 1 identity along arbitrary lines;
  - attenuation through the stack.
- `control/lqr.py`: continuous and discrete LQR (scipy's ARE solvers), closed-loop
  eigenvalues, the spectral abscissa and spectral radius.

## Experiment (`experiments/p1_quadrotor/`)

- `config.yaml`: the sampling design (all of its choices open, listed in the p1
  draft), the grid (2 architectures × 2 initialisations × H 128 × {1, 3} blocks ×
  5 seeds), training, logging, analysis and hover-check settings, and the benchmark.
- `p1_data.py`:
  - sampling of states and inputs around hover, and the target
    y = (x_{t+1} − x_t)/dt with RK4 at dt;
  - standardisation from the training split, and ground-truth Jacobians of the
    standardised target map by autodiff;
  - `.npz` storage outside the repo (`data/p1/`), with a committed SHA-256 manifest.
- `p1_hover.py`, the hover linearisation check: the surrogate's (A, B) at hover in
  physical units against the truth's. Reported:
  - relative Frobenius errors and sign agreement;
  - the LQR gains (fixed Q, R) and the closed-loop eigenvalues on the true
    linearisation.
- `run.py`: stages `generate`, `train`, `analyse` and `hover`, all gated on
  `prereg-p1`, and `benchmark`, which is not gated (random targets only).

## Tests (synthetic only)

- Architectures: shapes, the initialisation distributions, and agreement with a
  hand-written forward pass.
- Training: early stopping on a constructed validation curve; the lens log; fitting a
  learnable synthetic target.
- Analysis: zero error for a surrogate equal to the true map; the near/far split on
  constructed data; Corollary 1 on a trained-like random layer; attenuation on a
  constructed stack.
- Hover check: zero error and identical gains when the surrogate equals the true map;
  known LQR cases; sign agreement.
- Data: shapes, standardisation, Jacobians against finite differences, and the
  manifest round trip.
