# Experiment plan (v3, 23 Sep 2026)

What the code must produce for the TMLR submission. The full plan (dates, venue
strategy, risks) lives in the "TMLR J2C plan v3" doc; this file keeps only what
matters for the code. Target: paper submitted 17 Nov 2026; main-text experiments
frozen 1 Nov 2026.

## Tiers

Only the core tier must be done before submission. Stretch items run afterwards and
go into the journal revision.

| Test | What it is | Tier |
| --- | --- | --- |
| Initial-lens check | 1,000 initialisations per (H, k): distribution of r* and z* | Core, week 1 |
| R6 | Lens of TD-MPC2's first encoder layer vs the states the robot visits | Core, week 1 |
| P-I | 40 quadrotor surrogates: lens and Jacobian error vs lens distance | Core |
| Hover check | Surrogate (A, B) at hover vs truth; LQR gain; closed-loop eigenvalues | Core |
| P-III (hover) | Gradient-based MPC through the surrogate from 100 starts near hover | Core |
| P-II | Six fixes compared at equal accuracy | Core |
| R1a, R1e, R2b | Toy experiments on existing plants | Core |
| P-III tracking, P-I at H = 256, R6 dynamics layer, Jacobian penalty, DyT, R1b | | Stretch |

## Decision gates

- G1 (4 Oct): is the TD-MPC2 encoder lens sharp and inside its data, by the
  pre-registered criterion in `prereg/r6.md`?
- G2 (18 Oct): do zero-bias quadrotor surrogates concentrate Jacobian error at hover?
- G3 (13 Nov): is every main-text claim backed?

## Common rules

- float64 throughout. 5 seeds for comparisons, 3 for mechanism experiments.
- Early stopping: patience 10% of steps, tolerance 1% on held-out one-step MSE.
  Converged budget 100k steps. rel-MSE = MSE / Var(y).
- Every Jacobian error is reported in the units fixed by the pre-registration.

## Initial-lens check

H = 128, k = 16 (add other (H, k) pairs if cheap). Two initialisations:
(a) zero bias with PyTorch's default weight init; (b) PyTorch's default nn.Linear init
for weights and bias. 1,000 draws each. Report the distributions listed in
docs/theory.md, implementation convention 4, and compare with r* ≈ √(1 − (k+1)/H).

## R6: TD-MPC2 (evaluation only; no training)

1. Only after the git tag `prereg-r6` exists.
2. Read TD-MPC2's weight-initialisation code and record it exactly.
3. Load the single-task checkpoints in `experiments/r6_tdmpc2/config.yaml` with
   PyTorch on CPU. Extract the encoder's first NormedLinear: weight E, bias b, and the
   LayerNorm ε. Save as .npz.
4. Compute the lens with `lens/geometry.py`: z*, principal widths, Σ.
5. Collect observations by running each released agent in its environment
   (episodes and seeds in config.yaml). Record which normalisation, if any, TD-MPC2
   applies to observations before the encoder.
6. Compute coverage, sharpness and lens distances; apply the criterion in
   `prereg/r6.md`. Also check the Corollary 1 identity on the real layer along lines
   through the data.
7. Stretch: the dynamics model's first layer (degenerate regime): distance of replayed
   (latent, action) pairs to the singular set.

## P-I: quadrotor surrogates

- Plant: 12-state, 4-input quadrotor (plants/quadrotor.py), exact Jacobians by
  autodiff; inputs standardised.
- Surrogate target: the scaled increment (x_{t+1} − x_t)/Δt in standardised
  coordinates. Main-text Jacobian errors are for this map; discrete-map values go to
  the appendix.
- Grid: 2 architectures (pre-norm residual block; TD-MPC2-style NormedLinear stack)
  × {zero bias, PyTorch default} × H = 128 × {1 block, 3-block stack} × 5 seeds
  = 40 surrogates. k = 16.
- Log z*, r* and the susceptibility S every few hundred steps during training.
- Measure: the lens, coverage and sharpness from weights; Jacobian error against lens
  distance; the Corollary 1 identity on trained models.
- Predictions are in `prereg/p1.md` (to be written after the initial-lens check).

## Hover linearisation check

For each surrogate, compare its (A, B) at hover with the simulator's (both by
autodiff, same coordinates): relative Frobenius error of A and B; sign agreement on
entries above a fixed magnitude threshold; LQR gain from the surrogate with fixed
Q, R vs the true gain; closed-loop eigenvalues of that gain on the true
linearisation (stable or not, spectral abscissa). Repeat at steady-flight trims
spanning lens distance where possible.

## P-III: closed loop

Hover regulation from 100 perturbed initial states; gradient-based shooting MPC or
iLQR in JAX through the surrogate. Arms: zero bias, PyTorch default, best P-II fix,
Sobolev oracle. A run fails if tilt > 60°, position error > 2 m, actuator limits are
violated beyond solver tolerance, or cost > 2× the oracle's from the same start.
Test whether failures cluster near the lens centre. Validate the MPC on the true
simulator first.

## P-II: fixes

Targets: linear and 2-state plants (mechanism), quadrotor (scale). Arms: zero bias;
PyTorch default; projection-matched initialisation (Corollary 4); degree-one
normalisation (static, or large ε with γ initialised at √ε); RMSNorm; Sobolev
training (oracle). One shared log grid per penalty at the early-stopping budget;
select by minimum J-error with MSE within 20% of baseline; then 5 seeds at both
budgets. Report J-error at matched rel-MSE, lens diagnostics, closed-loop cost and a
compute-cost column.

## Toy experiments (existing plants)

- R1a: one-input block Linear(1→H) → LayerNorm → branch → skip → decoder; lens width
  fixed at 8 values from 100× narrower to 30× wider than the domain; targets SINE with
  X ∈ {1, 3, 6, 12} and a single bump; H = 16 and 64. 240 runs.
- R1e: pre-registered replay of SINE X = 12: predict from the lens alone how many
  periods the model reproduces, then check.
- R2b: net step vs zero-net bump; prediction that the step is removed ≥ 10× faster.
