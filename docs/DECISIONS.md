# Decision log

Choices that the specs leave open, made under the author's standing mandate (CLAUDE.md;
queue of 4 Oct 2026), for asynchronous review. Newest last. Each entry gives the date,
the task, the choice, the reasoning, and the alternatives rejected.

## 2026-10-04, queue item 2 (quadrotor simulator, `plants/quadrotor.py`)

1. **Frames.** The world frame is z-up with gravity −z; the body frame is FLU (x
   forward, y left, z up), with thrust along body +z.
   - Why: gym-pybullet-drones, the source of the approved parameters, uses a z-up
     world, so this keeps that source's sign conventions.
   - Rejected: NED/FRD, the aerospace convention, which would flip every sign
     relative to the source.

2. **Rotor numbering, positions and yaw-torque signs** come from the cf2x.urdf prop
   links and gym-pybullet-drones' CF2X `_dynamics`. Rotors 0–3 are front-right,
   rear-right, rear-left and front-left, at ±L/√2; rotors 0 and 2 give −z torque,
   1 and 3 give +z.
   - Why: the mixer is then exactly equivalent to the source, and a test checks it
     against the URDF positions.
   - Rejected: Bitcraze's M1–M4 numbering, and a "+" configuration.

3. **Inputs are not clipped inside `f`.** f_max is exposed as `params.f_max`, and
   limits are left to callers and controllers.
   - Why: clipping in `f` would put kinks into the dynamics and corrupt the exact
     Jacobians that P-I needs.
   - Rejected: clipping inside `f`, and a smooth saturation.

4. **Linear drag has no default coefficients.** `drag=True` requires explicit
   (Dx, Dy, Dz).
   - Why: no linear drag coefficient is in the approved parameter set, and the URDF's
     drag is rotor-speed-dependent.
   - Rejected: deriving a linear coefficient from the URDF's rotor drag at hover,
     which would give about 5.6e-3 N/(m/s) for xy. That would be a parameter beyond
     the approved set (see the questions in STATE.md).

5. **u is held constant over each RK4 step** (zero-order hold), and the Euler
   singularity at θ = ±π/2 is not handled.
   - Why: the sampling and tests keep |φ|, |θ| < 60°.
   - Rejected: quaternions. They would change the 12-state spec.

6. **`linearize_step` was added:** the Jacobians of the RK4 map, beside `linearize`'s
   continuous (A, B).
   - Why: plan.md wants discrete-map values in the appendix.

7. **The RK4 order test uses a near-hover manoeuvre** (5% differential thrust,
   moderate rates) with dt = T/20, T/40 and T/80 (T = 0.2 s), against a reference at
   T/2560.
   - Why: with Crazyflie inertias, random full-range thrusts give angular
     accelerations of order 100 rad/s². Those are not yet in RK4's asymptotic regime
     at dt = 0.01: the observed error ratios were 66 and then 12.6.

## 2026-10-04, queue item 3 (P-I infrastructure)

8. **The surrogates are new code, in `lens/models.py` and `lens/train.py`;
   `lens/core.py` is not used for them.**
   - Why: `core.py` is frozen, and its blocks are scalar-output, with GELU branches
     and Glorot initialisation. P-I needs 12 outputs, a NormedLinear arm and the
     initial-lens check's initialisations.
   - Rejected: wrapping `core.make_stack` (no vector head, wrong initialisation).

9. **The pre-norm block is `core.py`'s structure, generalised:**
   h ← h + W2 GELU(W1 (γ LN(h) + β) + b1) + b2, with the head read from the residual
   stream, y = Wo h + bo.
   - Branch width = H = 128 (configurable).
   - Why: same family as the R1 toys and Theorem 1's setting.
   - Rejected: a Mish branch (TD-MPC2's activation belongs to the other arm); widths
     of 2H or 4H; a final LayerNorm before the head.

10. **NormedLinear arm:** "1 block" is one NormedLinear (k → H) and a Linear head;
    "3 blocks" is three NormedLinear layers (k → H → H → H) and a Linear head.
    - Why: TD-MPC2's `mlp()` pattern. Regression outputs need a plain Linear last,
      not SimNorm.
    - Rejected: counting the head as a block, or ending with SimNorm.

11. **Initialisations follow the initial-lens check's conventions.**
    - `zero_bias`: PyTorch-default weights, U(±1/√fan_in), and every Linear bias = 0,
      in all layers. `torch_default`: weights and biases both U(±1/√fan_in).
    - The bias is always drawn, then zeroed for `zero_bias`, so the two arms of a
      seed share every weight: a paired comparison, like the initial-lens check's
      "same E for a given draw".
    - Rejected: zeroing only the first layer's bias; independent weight draws per arm.

12. **Training: full-batch Adam, lr 3e-3, (0.9, 0.999, 1e-8), as `core.train`;**
    evaluation and early-stopping checks every 500 steps.
    - The 1% tolerance is measured against the held-out MSE at the last counted
      improvement, and patience is 10% of `max_steps`.
    - The returned parameters are those with the lowest held-out MSE seen.
    - Rejected: minibatches by default (still available); counting patience in
      evaluations; returning the last parameters.

13. **The logged susceptibility is S(d) = ‖J(z*) d‖**, the output's slope at z*
    along d.
    - Why: by D6 (g)'s identity, S = ‖g′(0)‖/r*(d) = ‖J(0)‖. This is also finite for
      the degenerate zero-bias start, because ε > 0.
    - Rejected: computing S through g′ (undefined when r* = 0).

14. **Jacobian error is the per-state relative Frobenius error**
    ‖J_s − J‖_F / ‖J‖_F, in standardised coordinates.
    - The near/far comparison uses the median of the 10% nearest against the 50%
      farthest by ρ, or ρ_eff for a degenerate lens. Sets are ceil(fraction · n)
      states.
    - Listed as open in the p1 draft.
    - Rejected: the mean, absolute errors, physical units.

15. **Sharpness and coverage are computed along any direction:** D(d) as r6.md
    computes D along d₁, and coverage (2/π)·arctan(D/r*), using r_eff for a
    degenerate lens.
    - Both d₁ and u_min are reported, with d₁'s top-two eigenvalue ratio, because
      standardised i.i.d. inputs have a covariance close to I, so d₁ is poorly
      determined. This is flagged in the p1 draft.
    - Rejected: d₁ alone.

16. **Attenuation is measured on a grid uniform in t** along z* + t·d.
    - Each profile is summarised by peak/median of its speed: each block's pre-affine
      ĥ_j and the output.
    - It is reported relative to block 1's profile, which is the Corollary 1
      Lorentzian.
    - Rejected: a grid uniform in θ (it weights the median toward the centre);
      gradient-norm ratios at z* alone.

17. **The hover check compares the y-map's Jacobians**, A_y = (A_d − I)/dt and
    B_y = B_d/dt, with the surrogate's, both in physical units.
    - The LQR is continuous-time on (A_y, B_y), and the spectral abscissa is that of
      A_y,true − B_y,true K_s.
    - Q and R come from Bryson's rule on the sampling half-widths.
    - Signs are compared on entries above 1e-3 × max|truth|, per matrix.
    - A failed Riccati solve for the surrogate counts as unstable.
    - Rejected: the continuous f-Jacobian as the truth (not the map the surrogate
      learns), a discrete LQR, and Q = I, R = I (unit-dependent).

18. **Data.**
    - The plant is generic (a step function plus a trim), so tests use synthetic
      plants.
    - Sampling is i.i.d. uniform in a box around hover. Defaults: ±1 m, ±1 m/s,
      ±30° roll/pitch, ±180° yaw, ±2 rad/s, and thrust u0(1 ± 0.5), clipped. Sizes
      are 20,000 / 5,000 / 5,000.
    - The z-score comes from the training split, and Jacobians are stored for every
      split.
    - Files go to `data/p1/`, with the manifest at `results/p1/data_manifest.csv`;
      loading verifies every SHA-256.
    - All of these are open in the p1 draft.
    - Rejected: trajectory sampling as the default (configurable later).

19. **Gating:** `generate`, `train`, `analyse` and `hover` all require the annotated
    tag `prereg-p1`. `generate` is gated too, conservatively, so that no quadrotor
    data exists before the predictions are fixed. `benchmark` is ungated (random
    targets).
    - Rejected: leaving data generation ungated.

20. **The pipeline tests use a synthetic linear plant** with the quadrotor's
    dimensions. The quadrotor appears only in plant-level tests (sampling, the hover
    Jacobians of the RK4 map), never with a surrogate.
    - Why: stop condition 3 (no surrogate trained on quadrotor data before
      `prereg-p1`).

21. **The benchmark** (`run.py benchmark`, at 08d1deb) times each of the 8 grid
    configurations for 1,000 steps after a compile-and-warm-up call.
    - Data: random targets (Z ~ N(0, I₁₆), Y a fixed random tanh network), 20,000 and
      5,000 states.
    - Patience is disabled so every run takes all its steps.
    - It extrapolates to 40 models × 100k steps as an upper bound.
    - Rejected: timing on quadrotor data (stop condition 3), and timing with early
      stopping (its stopping point depends on the data).

## 2026-10-04, queue item 4 (P-I pre-registration draft)

22. **Predictions 1–4 were reconstructed from plan.md**, because the repository has
    neither the plan document's predictions nor the "protocol amendments":
    1. error concentration near the lens (G2);
    2. default dependence (R6's headline);
    3. hover linearisation damage;
    4. Corollary 1 on trained models.

    They are flagged in the draft and in the questions.
    - Rejected: leaving them blank, which would have blocked the draft.

23. **Prediction 5's thresholds** are κ ≤ 0.1 and 0.1 ≤ r_eff(d₁)/D ≤ 10.
    - They are anchored on values computed before any quadrotor data exists: the
      zero-bias start has r_eff/D median 0.0113 (p95 0.0126) for standardised i.i.d.
      inputs at H = 128, k = 16, ε = 1e-5.
    - And on R6's E3: trained TD-MPC2 lenses at 0.25–1.17, κ ≤ 0.0017.
    - Rejected: a relative-only criterion ("×10 from the start"), which says nothing
      about order 1; and thresholds taken from the TD-MPC2 values themselves, which
      would be too strict for a different model and data.

24. **The cell rule is "at least 4 of 5 seeds"**, with G2 proposed as prediction 1
    holding in at least 3 of 4 zero-bias cells, and accuracy matched at rel-MSE within
    20% (plan.md P-II's rule). All three are listed for decision.

## 2026-10-04, second queue: minibatch Adam, the long-budget subset, TACC (`docs/plans/p1-revision.md`)

25. **Linear drag is off for P-I** (the author's answer to question 1). The p1 draft
    records it as a modelling choice: `plants/quadrotor.py` keeps the flag, off, with no
    coefficients.

26. **Minibatch Adam is P-I's training default** (the author's answer to question 2):
    batch 2,048, lr 3e-3 and the Adam constants unchanged.
    - Each step draws 2,048 distinct training rows afresh, from
      `fold_in(PRNGKey(seed), step)`. There are no epochs.
    - The held-out one-step MSE is evaluated on the full validation set every 500 steps,
      and so is the training MSE, on the full training set. Early stopping is unchanged.
    - `max_steps` stays 100,000, plan.md's converged budget, now counted in minibatch
      steps (about 10,240 epochs of 20,000 states).
    - The default lives in `experiments/p1_quadrotor/config.yaml`. `lens/train.py`'s
      library default stays full batch, because synthetic tests use fewer than 2,048
      rows. `train` refuses a batch larger than the training set.
    - Why: the batch stream depends only on (seed, step). The paired arms of a seed
      therefore see the same batches as well as the same weights, and any step can be
      reproduced without carrying sampler state.
    - Rejected:
      - epoch-wise reshuffling (needs sampler state in the scan carry);
      - scaling lr with the batch size (no rule in plan.md, and 3e-3 is the full-batch
        value already in use);
      - counting the budget in epochs.

27. **u_min is recomputed from the current weights at every lens-log step**, along with
    D(u_min) from the training inputs. Logged: r*(u_min), r_eff(u_min), D(u_min), both
    ratios, S(u_min) and the vector u_min.
    - Why: prediction 5's primary quantity is along the lens's narrowest direction, which
      moves during training.
    - Rejected: a fixed u_min taken from the initial or final weights; it would track a
      direction the lens has left.

28. **The long-budget subset is a separate gated stage, `train_long`**: config
    `long_budget`; zero bias, 1 block, both architectures, seeds 0–4.
    - Early stopping is disabled (`patience_frac: null`). The lens is logged every 500
      steps, and parameter snapshots are saved every 10,000 steps.
    - Both the last parameters and the best held-out ones are kept.
    - The snapshots let `analyse_long` trace prediction 1's statistic over training (the
      race in the p1 draft).
    - The budget is set from the minibatch benchmark (entry 33).
    - Rejected: snapshots at every log step (about 15 MB per run at 500k steps, with
      little gain), and running the subset as an extension of the 40 runs (their early
      stopping would have to be switched off mid-run).

29. **One run per invocation.**
    - `--index i` runs member i of a stage's grid, and `--out-root DIR` moves data,
      checkpoints and results under DIR. The data manifest is always the committed one.
    - Every run, laptop or TACC, writes a run JSON (summary row, commit, dirty flag,
      versions, config, Slurm identity) and a SHA-256 manifest of its outputs.
    - `gather` verifies every member's files and requires one clean commit for all of
      them. It refuses on any gap or mismatch, then writes `train_summary.csv` and
      `outputs_manifest.csv`.
    - Why: array tasks must not write a shared summary, and a run on another machine
      needs the same provenance as a local one.
    - Rejected: one shared summary file with locking.

30. **TACC Slurm array scripts target Stampede3**, which supports job arrays (Stampede2
    did not): `slurm/p1_array.slurm`, `slurm/p1_gather.slurm`, `slurm/README.md`.
    - One run per array task and node. The repository is under `$WORK` and the outputs
      under `$SCRATCH`.
    - The commit is printed in the job log and recorded in every run JSON. `run.py`'s
      tag, clean-tree and data-manifest checks run first, and gather is a dependent job
      (`afterok`).
    - Queue, allocation and wall time are marked EDIT. The wall time is to be set from
      one test task measured against the laptop benchmark.
    - Rejected:
      - TACC's PyLauncher, which packs runs onto shared nodes and is more efficient, but
        the author asked for arrays;
      - GPU nodes (float64 with 128-wide layers gains little).

31. **Analysis quantities for the amendments' predictions 1–4** (`analyse_one`):
    - ‖z* − μ‖ with μ the training inputs' mean (prediction 1);
    - the initial median r*(d) over 64 random unit directions (rng seed 0), the
      initial-lens check's number per draw (prediction 2);
    - the affected-data fraction: test states whose Jacobian error exceeds 3× the far
      set's median, with Spearman correlations of the weight-based scores against it
      over the models (prediction 3);
    - attenuation along u_min as well as d₁ (prediction 4).

    Each is an operationalisation proposed in the p1 draft's decisions, not a choice
    already made.

32. **Prediction 5's rule is a pure function, `p5_run`, of a run's lens log**, applied
    by the gated stage `p5`.
    - It requires the end step to be logged: the P-I config logs at every evaluation.
    - The end is the returned (best held-out) parameters for the 40 runs, and the last
      step for the long runs. This is proposed in the draft.
    - Rejected: the best ratio at any time (it rewards a transient), which is reported as
      `ratio_max` instead.

33. **Minibatch benchmark** (`run.py benchmark` at d7b2cb3, `results/p1/benchmark_bs2048/`;
    random targets, the same design as entry 21). The full-batch benchmark of 08d1deb
    stays in `results/p1/benchmark/`.
    - Batch 2,048 takes 0.016–0.042 s per step on the laptop CPU, against 0.06–0.27 s
      full batch: 4–6× faster, not 10×, because the per-step row draw and the Adam
      update do not shrink with the batch.
    - One lens record (d₁ and u_min) takes 8–23 ms, under 1% of the time between
      records.
    - Upper bounds, every run to its full budget: 30.2 h for the 40-model grid at
      100,000 steps, and 24.5 h for the long-budget subset at 500,000 steps.
    - **Long budget: 500,000 steps**, 5× the 40 runs' maximum. Even a run that never
      stops early is then trained to 5× its early-stopping budget, at about 2.5 h per
      run and 24.5 h for the subset on the laptop.
    - Rejected:
      - 300,000 steps (3×; cheaper, but little room past a late stop);
      - 1,000,000 steps (10×; 49 h for the subset).
    - With both stages at about 55 h in the worst case, the laptop can finish the grid
      before G2 (18 Oct). TACC stays the fallback.

34. **Initial values come from an ungated stage, `init_lens`**: the initial first layer
    of seeds 0–999 of each initialisation, on 20,000 i.i.d. U(−√3, √3)¹⁶ inputs
    z-scored by their own mean and standard deviation.
    - It reports per seed:
      - r_eff/D along u_min and d₁;
      - κ, ‖z*‖ and ‖z* − μ‖;
      - prediction 2's initial median r* (64 directions, rng seed 0);
    - and quantiles over seeds, plus the grid's seeds 0–4.
    - Why:
      - The sampling design draws every input coordinate independently and uniformly,
        so its standardised inputs have this distribution up to sampling noise. No
        quadrotor state, target or surrogate is involved (stop condition 3).
      - The first layer is the first draw for both architectures and depths, so one
        lens per (initialisation, seed) covers all four cells (tested).
    - Rejected: drawing the inputs with `p1_data.sample` on the quadrotor's box (same
      distribution, but it would put P-I's own sampling code in front of a lens
      computation before the tag).

## 2026-10-04, second queue: the p1 draft, v2

35. **Predictions 1–4 are now the amendments' item 3, quoted exactly**
    (`docs/protocol_amendments_v3.md`, d6b822b). v1's reconstructions are withdrawn:
    - v1's prediction 3 (hover damage) becomes H1, the ledger's "wrong in sign or
      stability" claim, in the hover-check section;
    - its paired zero-bias against PyTorch-default comparison is reported;
    - v1's prediction 4 (Corollary 1 on trained models) becomes a pipeline check, as
      plan.md lists it under "Measure", and a failure is a code error.
    - Rejected: keeping v1's numbering, which would not match the numbering the
      amendments, and so the paper, use.

36. **Every open choice in the draft is a numbered decision (D1–D18), each with a
    recommendation, its reasoning and the alternatives.** They are not repeated here.
    The substantive proposals:
    - u_min is the primary direction (D6);
    - "affected" means an error above 3× the model's own far-set median (D9);
    - prediction 4's ratio is taken along u_min, with 0.2 checked within a factor of 2
      (D10);
    - prediction 5 is evaluated at the end, with the long-budget subset as the primary
      test (D11);
    - the race is reported, not tested (D13);
    - H1 is pre-registered, and the hover trims are 200 fixed no-drag trims (D16).

    The draft's operationalisations are already implemented in `run.py`. The exceptions
    wait for D16: H1's cell rule (its per-model quantities, sign agreement and
    stability, are already computed) and the trims.

## 2026-10-05, the author's decisions on the draft (`docs/plans/p1-prereg.md`)

37. **D3, the thrust range stays u0 (1 ± 0.5) = [0.5, 1.5] × hover thrust.**
    - It is symmetric about hover thrust and lies inside [0, f_max] = [0, 2.25 × hover
      thrust], so it is never clipped. Every box range is symmetric about hover, so the
      standardised mean is hover in every input dimension, up to sampling noise.
    - Rejected: the author's example [0, 2 × hover thrust]. It is symmetric too, but its
      zero-thrust corner is free fall, and it would double the input range for no gain
      in symmetry.

38. **D10: E1b's definition is not in the repository.**
    - Searched every commit for "e1b", "attenuation" and "local-affine": E1b appears
      only in theory.md's Scope remark. `lens/core.py` has `local_geometry` (the
      local-affine validity radius that E1b measured as 0.35–0.69 r*), but no
      attenuation ratio, and `notebooks/` holds only a README.
    - So the author's fallback is used: ratio = S_out/S_block1 along z* + t·u_min
      through block 1's lens centre, with S the slope ‖∂y/∂t‖ at t = 0.
    - S_block1 is read as the same slope for the model truncated after block 1:
      `models.trace(..., upto=1)`, with the head applied to block 1's output.
    - Rejected:
      - block 1's own ĥ slope, which is in different units from the output;
      - core.py's freeze ablation, which has no meaning for the NormedLinear stack
        (there is no skip connection).
    - The sharpness-ratio attenuation of entry 16 stays as a reported quantity.

39. **D16 (a), "no stabilising gain":** the Riccati solve for the surrogate's (A, B)
    fails, or its gain does not make A_s − B_s K_s Hurwitz. It is recorded per model,
    and counts as a failure for H1 (`h1_fail`). This replaces entry 17's "a failed
    solve counts as unstable", which missed a solve that returns a non-stabilising gain.

40. **D16 (b) and (c), the trims.**
    - 200 no-drag trims, from `default_rng(20261005)`: position, velocity and yaw
      uniform in the training box; roll, pitch and rates 0; thrust u0.
    - The stage `trims` (ungated, simulator only) checks that every trim is in the box
      and is a trim of the RK4 map (|step(x, u0) − x − dt·(v, 0, …)| ≤ 1e-12). It
      commits `results/p1/trims/` with a SHA-256 manifest before any training, and a
      rerun must reproduce the file byte for byte.
    - The truth depends on a trim only through yaw (tested to 1e-10). The author's
      premise that "the truth is nearly constant across the family" therefore holds
      exactly in position and velocity but not in yaw.
    - So the variation is reported as an error:
      ‖(J_s(trim) − J_s(hover)) − (J_t(trim) − J_t(hover))‖_F / ‖J_t(hover)‖_F.
      Per model, its median and its Spearman correlation with the trims' lens distance
      are reported, as are those of the A and B errors.
    - Rejected: fixing yaw at 0 (it would drop a free direction of the family), and
      the raw variation ‖J_s(trim) − J_s(hover)‖, which is wrong for yawed trims.

41. **Every rule now has code, in `p1_rules.py`**, applied by the gated stage
    `predictions`. Aggregations the draft left implicit:
    - prediction 2 holds if (a) holds for every PyTorch-default model and (b) holds in
      every cell;
    - prediction 4 holds if all four 3-block cells hold;
    - H1 holds if all four zero-bias cells hold;
    - prediction 5 holds if both architectures hold.

    Spearman ties get average ranks (scipy).

42. **The analysis stops on a Corollary 1 deviation above 1e-10** (`check_corollary1`;
    config `analysis.corollary1.tol`). It applies to a trained model with a
    non-degenerate lens, and a NaN deviation also stops it. Until now the deviation was
    only recorded, while the draft (D18) and `prereg/p1.md` say the stage stops.

## 2026-10-05, the author's review of `prereg/p1.md` at ade76d6

Source for entries 43–55: the author's message "Author review of prereg/p1.md at ade76d6:
not approved for tagging yet", items A1–A9 and P1–P5.

43. **A1, the sign test uses the vector field.**
    - The mask is |∂f| > 1e-3 × max|∂f| per matrix, from `plants/quadrotor.f`'s
      Jacobians at hover (`p1_hover.true_f_jacobians`). The signs of the surrogate's
      y-map (A, B) are compared with the true y-map's on that mask.
    - Before every comparison, at hover and at each trim, `mask_sign_check` verifies that
      the true y-map has ∂f's sign on the mask. It holds at hover and at all 200
      committed trims (tested).
    - At hover the mask has 8 entries in A and 16 in B. The y-map's own threshold would
      add 4 entries in A (∂y_p/∂θ = ±g·dt/2 = ±0.04905, exactly the predicted O(dt)
      term) and 12 in B (the attitude rows, through ∂ω̇/∂u·dt/2). The disagreement count
      on that unmasked set is reported without a rule.
    - `Plant` gained the field `f`.

44. **A2, prediction 4 is scoped to the stack:** the rule covers the two NormedLinear
    3-block cells, and the point prediction is the median over the five zero-bias
    NormedLinear 3-block models. Pre-norm 3-block cells are reported without a rule.

45. **A3, prediction 4's primary ratio is unit-free** (`analysis.attenuation_P`):
    - P = s(0)/median over 2,001 points uniform on [−D(u_min), D(u_min)], with s(0)
      evaluated at t = 0 exactly;
    - s_out = ‖∂y/∂t‖ (standardised output), and s_1 = ‖∂h₁/∂t‖ (`models.hidden`, block
      1's output in feature space);
    - ratio = P_out/P_1.

    The head-on-block-1 ratio of entry 38 stays, without a rule. A test shows that P is
    unchanged by scaling the head.

46. **A4, prediction 1's overall verdict:** it holds if it holds in every zero-bias cell
    (`p1_rules.p1_overall`). G2 is unchanged: 2 of 4 cells, a planning gate.

47. **A5.**
    - **ρ_eff everywhere:** within a model ρ = (1 + κ)·ρ_eff, a constant factor, so the
      ordering is identical (tested, including the stable-sort order). R, the affected
      fraction and the trims' correlations use ρ_eff, and the switch in
      `analysis.lens_distances` is gone.
    - **The lens log** records both r* and r_eff widths, without a switch.
    - **The numeric criterion:** theory.md defines degeneracy exactly, c⊥ = 0 (Remark 1;
      convention 4), with no numeric tolerance. `lens/geometry.py` (used unchanged by R6,
      so not changeable) treats ‖c⊥‖ ≤ 1e-12·‖b‖ as zero.
    - Per A5, I stopped before committing `prereg/p1.md` and proposed a criterion to the
      author. Nothing in the code depends on the choice except the degenerate flag and
      the Corollary 1 stop's applicability.
      - Prediction 5's κ ≤ 0.1 needs ‖c⊥‖² ≥ 10·Hε = 0.0128, so any lens under either
        criterion fails it.

48. **A6, failures and interruptions.**
    - A non-finite training or held-out MSE, or any non-finite parameter, at an
      evaluation stops the run, which is recorded as diverged (`train.py`).
    - A diverged run fails every rule it enters (`p1_rules`):
      - a per-seed rule does not count the seed;
      - a pooled rule (prediction 3) fails;
      - H1 does not count it;
      - the race tables count it in its own column.

      No seed is replaced.
    - Outputs are written atomically (temporary file, then `os.replace`), and the
      per-run manifest is written last. A run is complete exactly when its manifest
      verifies (`run_complete`). Complete runs are skipped, and any other run restarts
      from step 0 with its seed, after its stale manifest is removed.
    - Resuming from a snapshot is not implemented, so the byte-identity condition for
      resuming never arises.
    - `.npz` files are written with fixed zip timestamps (`save_npz`), so equal arrays
      give equal bytes. A test shows that a restart reproduces the files byte for byte.

50. **A8:**
    - (a) clause 2(a) is stated as carrying no evidential weight;
    - (b) within-initialisation Spearman correlations (n = 20 each) are reported in
      `p1_rules.p3`;
    - (c) ‖z* − hover_std‖ is added (`z_star_to_hover`).

51. **A9, yaw is not wrapped.** `plants/quadrotor.py` has no modulo on any state. A test
    draws 400 states from the box with yaw within 1° of ±π:
    - every target is far from 2π/dt = 628;
    - the Euler-rate targets are under 10 rad/s;
    - yaw leaves ±π continuously.

52. **P1, the pre-flight (`preflight.py`, `preflight_override.yaml`).**
    - It runs `run.py`'s own stage functions on a synthetic linear plant,
      ẋ = A(x − x0) + B(u − u0), with Euler steps (so the y-map is exactly (A, B)), the
      quadrotor's dimensions, trim and box, and the real config.
    - The override file cuts the budgets and redirects outputs to
      `results/p1_preflight/`; the tagged `config.yaml` doesn't contain it. The gate is
      not bypassed: `run.py`'s command line still refuses quadrotor stages.
    - `check_outputs` verifies every quantity p1.md promises. `preflight.py stops`
      injects the five stopping faults.

53. **Additions found while tracing p1.md to code (P2):**
    - the Jacobian error distribution (quantiles 5–99);
    - held-out rel-MSE on validation and test, which p1.md promises for prediction 2(b);
    - the final lens arrays (`analyse_lens.npz`).

    `run.py`'s analyse and hover loops became functions (`stage_analyse`,
    `stage_hover`) so the pre-flight and the tests call the same code as the stages.
