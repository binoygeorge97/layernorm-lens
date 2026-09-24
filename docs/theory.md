# Theory reference

Definitions and results the code implements. Source: "TMLR plan: theory & protocols"
(22 Sep 2026), §1–2. The Sections 2–4 draft (v2) is the authoritative statement of
the theory; where it differs, see "Differences in the v2 draft" at the end.

## Setting

- Encoder input z ∈ ℝᵏ, counting every input the encoder sees (state and control).
- Pre-norm activation h = E z + b, with E ∈ ℝ^{H×k}, b ∈ ℝ^H.
- Centring projector P = I − (1/H) 𝟙𝟙ᵀ. A = P E, with rank k ≤ H − 1.
- LayerNorm without affine parameters: ĥ = r / σ, with r = P h and
  σ = √(‖r‖²/H + ε).
- Closest-approach point: z* = −(AᵀA)⁻¹ Aᵀ P b.
- Offset: c⊥ = P b + A z*, so Aᵀ c⊥ = 0 and r = A (z − z*) + c⊥.
- ĉ = c⊥ / ‖c⊥‖,  κ = H ε / ‖c⊥‖²,  φ = κ / (1 + κ).

The results hold for the first normalisation after an affine map, with or without a
residual connection. This covers TD-MPC2-style NormedLinear layers
(Linear → LayerNorm → Mish).

## Theorem 1 (LayerNorm is a gnomonic projection)

If c⊥ ≠ 0, then with u = A (z − z*) / ‖c⊥‖,

    ĥ(z) = √H · (ĉ + u) / √(1 + κ + ‖u‖²).

The direction of ĥ is the inverse gnomonic projection of u from the tangent plane at
ĉ, for every ε. The floor only shrinks the radius, to
√H · √((1 + ‖u‖²) / (1 + κ + ‖u‖²)). Proof: r = ‖c⊥‖(ĉ + u) with u ⟂ ĉ, so
‖r‖² = ‖c⊥‖² (1 + ‖u‖²); substitute into r/σ.

## Corollary 1 (one angle per line; exact Jacobian)

On a line z = z₀ + s·d, let s* be the line's own closest approach,
r*_ℓ = ‖c⊥,ℓ‖ / ‖A d‖ its width and κ_ℓ its floor ratio. Then

    θ(s) = arctan((s − s*) / r*_ℓ)
    ĥ(s) = √H (cos θ ĉ_ℓ + sin θ q̂) / √(1 + κ_ℓ cos² θ)

The branch is exactly B(s) = g̃(θ(s)) for a function g̃ on (−π/2, π/2) fixed by the
branch weights and κ_ℓ, so, exactly, for every ε:

    J_branch(s) = g̃′(θ) · r*_ℓ / (r*_ℓ² + (s − s*)²)

The Lorentzian factor comes from the encoder alone; g̃′ is the only learned factor.
Link to the older form: g̃′(θ) = ⟨m, dĥ/dθ⟩, so g̃′(0)/r* = ⟨m, q⟩/δ at ε = 0.

## Corollary 2 (the invariants are arc lengths)

At κ = 0 the path of ĥ along a line is half a great circle of radius √H: total
variation π√H, peak speed √H/r*, half-width r*, so peak × width = √H. As κ → ∞ the
path becomes a diameter of length 2√H; √H·I(φ) interpolates between the two.

## Theorem 2 (scaling identities)

Hold fixed the branch weights, γ, β, the sweep direction q, the direction ĉ and the
centre s*; set ε = 0 and vary only λ = ‖c⊥‖, so r* = λ/‖q‖ and δ = λ/√H. If the
branch has no net step, g̃(π/2) = g̃(−π/2) =: g̃∞, then exactly

    ∫ (B − g̃∞)² ds = r* · K₁,   K₁ = ∫ (g̃(θ) − g̃∞)² sec² θ dθ
    ∫ B′(s)² ds    = K₂ / r*,   K₂ = ∫ g̃′(θ)² cos² θ dθ
    sup |B′|       ≥ |g̃′(0)| / r*

Squared-L² footprint Θ(δ), squared H¹ seminorm Θ(δ⁻¹), peak slope Ω(δ⁻¹). On a
bounded domain of half-width D containing the core, the first identity holds up to
O(r*²/D). For ε > 0 the same holds while λ² ≫ Hε.

## Corollary 3 (net step versus bump; a prediction)

If g̃(π/2) ≠ g̃(−π/2), the branch carries a net step Δ that no affine skip can absorb;
it costs Θ(Δ²) in MSE regardless of δ. A zero-net bump costs only Θ(δ). Prediction:
training removes the net step fast and the bump slowly, leaving lobes of opposite
sign. Tested in R2b.

## Proposition 3 (the Cauchy lens)

Sphere area seen per unit input volume is proportional to

    (1 + (z − z*)ᵀ M (z − z*) / δ₀²)^(−(k+1)/2),  M = AᵀA / H,  δ₀² = ‖c⊥‖² / H

a multivariate Cauchy centred at z* with scale matrix Σ = δ₀² M⁻¹, independent of ε.
In one dimension it is the Lorentzian, and half of the angular budget π falls within
|s − s*| < r*. Proof: the inverse gnomonic map has Jacobian determinant
(1 + ‖u‖²)^(−(k+1)/2); pull it back through u = A (z − z*) / ‖c⊥‖.

## Corollary 4 (the optimal lens for a domain)

In one dimension, the width maximising worst-case resolution r*/(r*² + s²) over
|s − s*| ≤ D is r* = D. It spends half the angular budget on the domain, and
resolution varies only twofold across it. Proved for k-dimensional boxes: r*ᵢ = Dᵢ.

## Remarks

- **Remark 1 (zero bias).** With zero encoder bias (Flax's default), c⊥ = 0: the lens
  has zero width at z = 0, which is the data mean when inputs are standardised. The
  branch starts as a step smoothed only by ε.
- **Remark 1b / Remark 2 (degenerate case).** When rank A = rank P (k ≥ H − 1), the
  lens is degenerate for every bias: c⊥ can vanish on a set of dimension k − H + 1,
  giving an ε-dominated regime with an ε^(−1/2) law. This applies to TD-MPC2's
  dynamics input (k = 512 + T + A ≥ 511 = H − 1).
- **Scope.** Exact for the first normalisation in a block. Later blocks are
  second-order accurate near the tangent point (N3, exponent 1.88), within a measured
  radius of 0.35–0.69 r* (E1b), and usually attenuate the feature.

## Proposition (the lens at initialisation): estimate only

Zero bias gives c⊥ = 0: a zero-width lens at z = 0. Under PyTorch's default,
weights and bias are both uniform with variance 1/(3k) (that is, U(−1/√k, 1/√k) with
fan-in k). A rough calculation gives ‖c⊥‖² ≈ (H − 1 − k)/(3k) and
‖A d‖² ≈ H/(3k) for a unit direction d, so

    r* ≈ √(1 − (k + 1)/H)

about one input unit, with a centre whose coordinates are of order 1/√H. For
H = 128, k = 16: r* ≈ 0.93. These are estimates until the sampling check
(1,000 initialisations per (H, k)) confirms them.

## Diagnostics (all computable from the weights)

- Lens centre z* and width r* per input direction, from M.
- Sharpness D/r* relative to the data half-width D, and coverage (2/π)·arctan(D/r*).
- Readout gain |g̃′(0)| and susceptibility S = |g̃′(0)|/r*, exactly the peak branch
  slope at the centre.
- Total variation and J-error only where ground truth exists.

## Implementation conventions (confirmed 24 Sep 2026)

These follow from the definitions above but are not stated in the source; they fix
choices the code must make.

1. Width along a unit direction d through z*: r*(d) = ‖c⊥‖ / ‖A d‖. Principal widths:
   r*ᵢ = ‖c⊥‖ / sᵢ, with sᵢ the singular values of A.
2. Lens distance: ρ(z) = (z − z*)ᵀ Σ⁻¹ (z − z*) with Σ = δ₀² M⁻¹. This equals ‖u‖², so
   ρ = 1 is the 45° point of the projection.
3. Constant inputs fold into the bias. For a multi-task TD-MPC2 model with a fixed
   task embedding e, use b′ = b + E_task e and the state columns of E.
4. The initial-lens sampling check reports, for each initialisation, the distribution
   of the median of r*(d) over random unit d, the principal widths, and ‖z*‖. For
   degenerate (zero-bias) draws, where c⊥ = 0 and r* = 0, report the ε-limited widths
   √(Hε)/sᵢ instead of r* (and √(Hε)/‖A d‖ for the median over d).
5. ε is the value the model actually uses (PyTorch nn.LayerNorm defaults to 1e-5).
   Read it from kink_core.py or the checkpoint; never assume it.
6. lens/core.py's `r_star` is not r*_ℓ but r_eff = √(‖c⊥,ℓ‖² + Hε) / ‖A d‖ =
   r*_ℓ · √(1 + κ_ℓ) exactly (since ‖c⊥,ℓ‖²(1 + κ_ℓ) = ‖c⊥,ℓ‖² + Hε), the width of the
   normalised output's profile along the line; lens/geometry.py exposes it as `r_eff`.

## Differences in the v2 draft (paste the exact v2 statements here)

- Theorem 1 is stated as a central projection onto the half-spheroid
  (1 + κ)⟨v, ĉ⟩² + ‖Π_A v‖² = H, exact for every ε, with the gnomonic direction as
  part (iii).
- Corollary 1 describes each line's image as half an ellipse whose endpoints ±√H q̂
  depend only on the direction.
- Corollary 2 has the closed form 2√H E(φ), with E the complete elliptic integral of
  the second kind.
- New: Corollary 3 (slice lenses are the level structure of the plane lens) and
  Proposition 2 (the net step belongs to the direction and is shared by all parallel
  lines). Note the numbering clash with Corollary 3 above.
- Theorem 2 is an Lp ladder on the zero-net part of the branch, with the peak K∞/r*
  attained off-centre in general.
