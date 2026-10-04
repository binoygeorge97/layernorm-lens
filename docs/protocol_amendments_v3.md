# Protocol amendments v3

These amend the TMLR plan: theory & protocols doc of 22 September. Where they
differ, these win; everything not listed stands.

## Pre-registration items

Every item below is committed to the repository with a date and hash before G1
on 4 October. The thresholds are proposals; confirm them after the initial-lens
sampling check, never after seeing trained quadrotor models.

1. Surrogate target and units (new). The surrogate predicts the scaled
   increment (x_{t+1} - x_t)/dt in standardised coordinates. All main-text
   Jacobian errors are for this map; discrete-map values go to the appendix.
   Reason: a next-state Jacobian is about I + dt*A, which compresses errors by
   1/dt (100 at dt = 0.01, per the envelope taxonomy notes).
2. R6 sharp-lens criterion (new). Final version in prereg/r6.md. A lens counts
   as sharp and in the data if z* lies inside the data's 95% Mahalanobis
   ellipsoid, z* is no farther from its nearest observation than the 95th
   percentile of the data's own nearest-neighbour distances, and D/r_eff >= 5
   along the data's principal direction. The earlier "5% of samples within
   lens distance 1" test was dropped: in 16 or more dimensions almost no data
   falls within one lens width in every direction at once, even for a harmless
   lens.
3. P-I predictions, with numbers.
   1. Zero bias: the lens centre sits within 0.1 standardised units of the data
      mean (hover). Median Jacobian error over the 10% of test states nearest
      the lens (smallest rho) is at least 3x the median over the 50% farthest.
      Fixed rho cut-offs were dropped for the same high-dimension reason as in
      item 2.
   2. PyTorch default: the initial r* lies within a factor of 2 of the
      sampling-check estimate, and the error ratio in (1) is smaller than under
      zero bias.
   3. Coverage and sharpness computed from the weights rank the 40 surrogates
      by affected-data fraction with Spearman rho >= 0.7.
   4. The stack's first-block feature is attenuated, with E1b's median ratio
      of about 0.2 as the point prediction.
4. Display rule. Planning-gradient panels show every seed whose peak slope
   error exceeds the median, and report the fraction of seeds with a sign flip.
5. P-III failure (new). A run fails if tilt exceeds 60 deg, position error
   exceeds 2 m, actuator limits are violated beyond solver tolerance, or cost
   exceeds 2x the Sobolev oracle's from the same initial state.
6. Unchanged. Early stopping (patience 10% of steps, tolerance 1% on held-out
   one-step MSE), the P-II selection rule, and 5 seeds for comparisons.

## P-I amended

The grid drops from 120 surrogates to 40. Projection-matched initialisation
moves to P-II, where it is compared as a fix; H = 256 becomes a stretch item for
the revision.

| Field | v2 | v3 |
| --- | --- | --- |
| Plant | 12-state, 4-input quadrotor, exact Jacobians by autodiff | Unchanged |
| Surrogate target | Not specified | Scaled increment, standardised (item 1) |
| Architectures | Pre-norm residual block; NormedLinear stack | Unchanged |
| Initialisations | Zero, PyTorch uniform, projection-matched | Zero, PyTorch uniform |
| Width | H in {128, 256} | H = 128 |
| Depth | 1 block and a 3-4-block stack | 1 block and a 3-block stack |
| Seeds | 5 | 5 |
| Total | 120 surrogates | 40 surrogates |
| Measurements | Lens (z*, Sigma), coverage, sharpness; Jacobian error against lens distance; Corollary 1 identity | Unchanged, plus the hover linearisation check |

## P-II, the hover check and P-III amended

P-II keeps six arms in the core; P-III starts with hover regulation, and a
linearisation check is added as its cheap, always-available half.

P-II. Targets: the linear and 2-state plants for mechanism, the quadrotor for
scale.

| Arm | Tier |
| --- | --- |
| Zero bias; PyTorch uniform; projection-matched | Core |
| Degree-one normalisation; RMSNorm (predicted to keep the kink) | Core |
| Sobolev training, labelled as an oracle | Core |
| Generic Jacobian penalty; DyT | Stretch, revision |
| <m,q> penalty | Appendix, existing toy runs only |
| TaperNorm | Dropped |

The selection rule, budgets and headline (Jacobian error at matched rel-MSE,
with the cost column) are unchanged.

Hover linearisation check (new). For each surrogate, compare its (A, B) at hover
with the simulator's, both by autodiff and in the same coordinates.
- Relative Frobenius error of A and of B.
- Sign agreement on every entry whose true magnitude exceeds a fixed threshold.
- LQR gain from the surrogate's (A, B) with fixed Q and R, against the true gain.
- Closed-loop eigenvalues of the surrogate's gain on the true linearisation:
  stable or not, and the spectral abscissa.
- Repeat at steady-flight trims chosen to span lens distance where the flight
  envelope allows.

P-III. Hover regulation from 100 perturbed initial states, using gradient-based
shooting MPC or iLQR in JAX through the surrogate. Compare zero and PyTorch
initialisation, the best P-II fix and the Sobolev oracle on cost, failures
(pre-registration item 5) and constraint violations. Test whether failures
cluster near the lens centre. The tracking task moves to the revision.

## Toy experiments and the claims ledger

Three toy experiments stay in the core: R1a, R1e and R2b. Their protocols are
unchanged; everything else moves as below.

| Experiment | v3 status |
| --- | --- |
| R1a lens sweep (240 runs); R1e SINE X = 12; R2b two timescales | Core, protocol unchanged |
| R1b tail blindness | Stretch, revision |
| R1d; R2a | Appendix, evaluation only |
| R1c widening and migration; R2c annealing law | Thesis chapter |
| R4a-R4c on the toys | Thesis; superseded by the hover check and P-III |
| Silverbox or Wiener-Hammerstein | Dropped |

Ledger changes. Nothing marked pending enters the main text, as before.

| Claim | v3 change |
| --- | --- |
| The lens decides where the model can fit | Main text limited to R1a and R1e; widening and migration go to the thesis |
| Annealing time scales with delta_0 | Stated as a conjecture in the discussion, not a claim |
| Planning gradients flip sign and closed-loop cost rises near the lens centre | Evidence moves from the toys to the hover check and P-III |
| The <m,q> penalty fails by trading width for peak | Appendix |
| The lens theory applies to deployed world models | Encoder layer is core; the dynamics layer is stretch |
| New: under zero bias, the surrogate's hover linearisation is wrong in sign or stability | Prediction, pending the hover check |
