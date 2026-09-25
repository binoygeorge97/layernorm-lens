# Deviations from the R6 pre-registration, part 3

Date: 25 September 2026. Author: Binoy George.
Applies to `prereg/r6.md` (tag `prereg-r6`, tag object 086fbdd, commit 66b3a09),
`prereg/r6-deviations.md` (tag `prereg-r6-d1`, tag object 3a0bb10, commit e5060cf)
and `prereg/r6-deviations-2.md` (tag `prereg-r6-d2`, tag object df32edc, commit
e20f5a6), all unchanged. An earlier, untagged version of this file is commit 578f3e6;
this version replaces it before tagging. Decided before any R6 criterion quantity
(Inside, Populated, Sharp) or gate G1 was computed.

## What had been seen before D6

Recorded in `results/r6/`:

- D3 consistency on the policy-prior data (`consistency.csv`): identity was rejected
  for both pre-release checkpoints. cartpole-swingup seed 1: e/e₀ identity 0.036,
  symlog 0.020 (lowest), LayerNorm 0.046. humanoid-run seed 3: identity 0.989 (the
  random-pairing level), symlog 0.115 (lowest), LayerNorm 0.965. Public-layout
  calibration, identity: cheetah-run 0.014, walker-run 0.013, humanoid-run seed 1
  0.117.
- Policy-prior returns (`returns.csv`, D4): cartpole-swingup seed 1, acting with
  identity at position 0, reached 0.20 of its published return, while its public
  seeds 2 and 3 reached 1.00. humanoid-run reached at most 0.012, dog-run at most
  0.075, cheetah-run 0.38–0.48 of the published returns.
- Planner checks (`planner_check.csv`, `meta_planner_check.json`; public tdmpc2
  `TDMPC2` class at e9f59321, 5 episodes, Tesla T4): with `eval_mode=True`,
  cartpole-swingup seed 2 returned 883.2 (published 882.5, fraction 1.001) and
  humanoid-run seed 1 returned 672.8 (published 542.2, fraction 1.241); with
  `eval_mode=False`, 882.2 (1.000) and 616.8 (1.138). About 9 s (cartpole) and 13 s
  (humanoid) per episode on the T4, after a first episode of about 2 minutes that
  includes `torch.compile`.
- The key names and shapes of the two pre-release checkpoints
  (`prerelease_keys.json`, 4bc1a14).
- The lens from the weights, from `extract.py`'s lens stage: `results/r6/lens_summary.csv`
  and `results/r6/meta_lens.json` (with `lens/`, `weights/` and
  `meta_extract.json`), committed in 00603a8; the run is recorded at commit 4c3f129,
  clean (see `results/r6/PROVENANCE.md`). All 15 checkpoints are non-degenerate, with
  κ ≤ 0.00165. The r* principal widths are strongly anisotropic: width_max / width_min
  is 4.1–5.0 for cartpole-swingup, 93–142 for cheetah-run, 71–74 for walker-run,
  28–172 for humanoid-run and 9,040–11,000 for dog-run (ranges over the three seeds).
  dog-run's ‖z*‖ is 563–769. Dog-run's z* was also noted to lie outside the range of
  the observed states. That was an informal comparison of the lens with the states,
  related to Inside, for a task outside G1. The SHA-256s recorded in
  `meta_extract.json` match the provenance table below for all 15 checkpoints. For the
  two pre-release checkpoints these lens quantities are in the coordinates the layer
  receives, which under (b) are symlog coordinates. The lens is computed from the
  weights alone, in the coordinates the layer receives, so it doesn't depend on the
  position-0 reading; only its relation to raw observations does.
- The D4 policy-prior data (`meta_collect.json`; D3 consistency in
  `meta_consistency.json`, both recorded at 4c3f129) came from a different
  environment than the planner's pinned one: Python 3.13.15, torch 2.14.0+cpu, mujoco
  3.14.0, jax 0.10.2, against mujoco 3.1.2 in the planner's environment.

Scope: so far only the layer-1 lens has been measured from the weights (centre,
widths, κ). Nothing downstream of layer 1 has been measured, and no Inside,
Populated or Sharp quantity has been computed for any checkpoint.

Neither pre-release checkpoint has yet been loaded into the planner. The remap in (b)
was checked against key names and shapes only.

## Checkpoint provenance

Source: huggingface.co/nicklashansen/tdmpc2, revision
8fb2a82efb3bae96941da440128fe1332e4394fd (lastModified 2023-10-26T01:03:00Z).
SHA-256 from the Hugging Face LFS metadata. The two pre-release files (*) were also
hashed locally and match, so `prerelease_keys.json` (4bc1a14) was built from these
exact files.

| SHA-256 | File |
| --- | --- |
| e22aa7500efabafcb2b58f99134bb6e9fc42ca81ed032c6c90a921e393b3af4a | dmcontrol/cartpole-swingup-1.pt (*) |
| 4a81773b6136844c89f4ab0981e979758ead83aab163a64fa0151f142b004c0a | dmcontrol/cartpole-swingup-2.pt |
| 621b20b008a0b03b4266a118f9d7140707c8eadb52bca70984e77218575e06a3 | dmcontrol/cartpole-swingup-3.pt |
| 906924fee08bf43743b01f187db5cd9f355d717395c399f754c5ee4e4782b059 | dmcontrol/cheetah-run-1.pt |
| aa18d1ca1feaad68f503b267e6de7bdfe7b1cf46213c0aa90ed0985d1d1fdb67 | dmcontrol/cheetah-run-2.pt |
| 6ed9ba2730f75743ae27532a8e270c9de4769b2047855e32af565200fed74605 | dmcontrol/cheetah-run-3.pt |
| 7e825a2210ede7085a71f98893fe3f89a1acb9eedf234a7c43815c68ecc0150c | dmcontrol/walker-run-1.pt |
| 0c45b6a5c3f7387a30b5e3ce2c57b8c4101b33affaae582204c2dbe451ce0caf | dmcontrol/walker-run-2.pt |
| d8d0a635c5f49d951cd98a89125ee3ef205e01e631856a2d72a1288583d01089 | dmcontrol/walker-run-3.pt |
| 57f2c7e509d9efb677ae3f150e3e162c8f1a44ce2dfd34b7f4eb04a489e05c20 | dmcontrol/humanoid-run-1.pt |
| 7f6514a9519cbaec6f52f1319446fc74f74b1cda000ccf60026ce1663cd83db3 | dmcontrol/humanoid-run-2.pt |
| c262c177d9b748ef4f56397e334b8424cb0a1446c723f7b84526c09a8e106ffa | dmcontrol/humanoid-run-3.pt (*) |
| c5bc235f6aa605a0f4de99ae39848d9619b7a846527bd97152a5c3a39b249940 | dmcontrol/dog-run-1.pt |
| 25e1453434aeba04e523b332f99d54e542772eb272565ef0cca241c60df4b491 | dmcontrol/dog-run-2.pt |
| 348c9ce6f8c864a751ff5666443b216539e1d505ab77c8849df86621fd02e497 | dmcontrol/dog-run-3.pt |

The collector downloads each file from this revision (pinned in the URL) and refuses
any file whose SHA-256 does not match the table.

## D6. Planner data; symlog at pre-release position 0

### (a) Data for the 13 public-layout checkpoints

- Observations are collected with the public tdmpc2 planner at e9f59321, run
  unmodified: its own `TDMPC2` class, with the checkpoint loaded through
  `TDMPC2.load` (including `api_model_conversion`).
- The planner acts with `eval_mode=True`, its noise-free mean action. This departs
  from r6.md's literal "acting as in the repository's `evaluate.py`": `evaluate.py`
  calls `agent.act(obs, t0=t==0, task=task_idx)` without `eval_mode`, so the executed
  action includes the planner's exploration noise (tdmpc2 `tdmpc2.py:203`).
  `eval_mode=True` is how the published learning curves were produced
  (`trainer/online_trainer.py:37`) and how the deployed agent acts.
- The planner runs in float32 PyTorch, in tdmpc2's pinned environment
  (`docker/environment.yaml`, in its own Python 3.11 virtualenv), for acting only.
  The planner checks ran in it: `meta_planner_check.json` records Python 3.11.16,
  torch 2.7.1+cu126, tensordict 0.8.3, torchrl 0.8.1, mujoco 3.1.2, numpy 1.24.4 and
  gymnasium 0.29.1, the pinned versions (dm_control's version was not recorded,
  because the module has no `__version__`; the collector records it from the
  package metadata). The exact versions are recorded per run. All lens and
  criterion computations are float64, as CLAUDE.md requires.
- GPU execution with `torch.compile` is not bitwise deterministic, so a rerun
  reproduces the data in distribution, not exactly. The stored observations are the
  data of record.
- The published return of a checkpoint is the reward at the last logged step for
  the same seed in `results/tdmpc2/<task>.csv` of tdmpc2 e9f59321 (last changed in
  88095e7), as in D4; it is used for the 0.5 and 0.9 rules below.
- Any checkpoint whose planner return (mean over its 50 episodes) is below 0.5 of
  its published return is flagged. If this happens to the checkpoint that counts
  toward G1 for a task (seed 1, or seed 2 for cartpole-swingup if seed 1 ends up
  unidentified under (b)), that task leaves G1, and G1 then needs 2 of the
  remaining non-dog tasks.

### (b) The pre-release checkpoints: symlog at position 0

- For cartpole-swingup seed 1 and humanoid-run seed 3, encoder position 0 is taken
  to be symlog(o) = sign(o)·log(1 + |o|), elementwise. The layer then receives
  symlog(o), so for these two checkpoints every R6 coordinate (μ, C, D, z*, ρ, ρ_eff)
  is in symlog coordinates.
- Where symlog is applied: on the raw state observation (the flattened float32
  vector from tdmpc2's DMControl wrapper), before the first remapped `NormedLinear`
  of the encoder, and nowhere else. In tdmpc2 the observation enters the model only
  through `WorldModel.encode` (`tdmpc2.py:111, 115` and `_plan`, `tdmpc2.py:153`), so
  the collector passes symlog(o) to `agent.act` and nothing else changes; dynamics,
  reward, policy and Q-networks see only latents. The symlog is the lens code's own
  function, `layouts.input_candidates(...)["symlog"]`
  (`experiments/r6_tdmpc2/layouts.py:109`, `np.sign(o) * np.log1p(np.abs(o))`),
  evaluated in float64 and cast to float32 for the planner. The stored observations
  are the raw ones; symlog is applied again, in float64, for all lens and criterion
  computations.
- Public-layout checkpoints run through tdmpc2 unmodified, with no symlog. Loading a
  pre-release checkpoint through `TDMPC2.load` alone would give the identity reading
  at position 0, which D3 rejected.
- The two checkpoints are run through the same planner as (a) after a key remap to
  the public layout. The remap is fixed by the keys of both checkpoints
  (`results/r6/prerelease_keys.json`, 60 keys each). Each pre-release Linear becomes
  the public `NormedLinear`'s Linear and the LayerNorm-shaped module after it becomes
  that `NormedLinear`'s `ln`:

  | Pre-release key prefix | Public key prefix |
  | --- | --- |
  | `_encoder.state.1`, `.2` | `_encoder.state.0`, `.0.ln` |
  | `_encoder.state.4`, `.5` | `_encoder.state.1`, `.1.ln` |
  | `_dynamics.0.0`, `.0.1` | `_dynamics.0`, `.0.ln` |
  | `_dynamics.0.3`, `.0.4` | `_dynamics.1`, `.1.ln` |
  | `_dynamics.0.6`, `_dynamics.1` | `_dynamics.2`, `.2.ln` |
  | `_reward.0`, `.1` | `_reward.0`, `.0.ln` |
  | `_reward.3`, `.4` | `_reward.1`, `.1.ln` |
  | `_reward.6` | `_reward.2` (plain Linear, 101 bins) |
  | `_pi.0`, `.1` | `_pi.0`, `.0.ln` |
  | `_pi.3`, `.4` | `_pi.1`, `.1.ln` |
  | `_pi.6` | `_pi.2` (plain Linear, 2 × action dim) |

  These 40 keys (`.weight` and `.bias` of each prefix) match the public
  `WorldModel`'s parameter names and shapes exactly. The remaining 20 keys,
  `_Qs.params.0–9` and `_target_Qs.params.0–9`, are in the old public ensemble
  format and are converted by tdmpc2's own, unmodified `api_model_conversion`
  (parameter N → layer N // 4, weight/bias/ln.weight/ln.bias; 5 members, 101
  bins), which also supplies the absent `log_std_min` and `log_std_dif` buffers from
  the configuration. The checkpoints have no other keys. The remapped state dict is
  then loaded by `TDMPC2.load` like any public checkpoint, and the complete key
  mapping is written to the results. The parameter-free positions this implies are
  those of D3 (Mish after each hidden `NormedLinear`, SimNorm after the encoder's
  and the dynamics' last one), with ε = 1e-5, the same assumed stack as D3.
- Encoder agreement gate. Before any remapped planner run, a test compares the
  planner's float32 encoder (`WorldModel.encode` of the loaded `TDMPC2`) with the lens
  code's float64 encoder forward on the original, unremapped weights
  (`layouts.build_networks`), by the per-state relative error
  ‖z_planner − z_lens‖₂ / ‖z_lens‖₂ of the encoder output. Two state sets per
  checkpoint: (i) 1,000 random states, `numpy.random.default_rng(0).standard_normal
  ((1000, k))` scaled by the per-dimension standard deviation and shifted by the mean
  of that checkpoint's D4 policy-prior observations; (ii) all of that checkpoint's D4
  policy-prior observations. The random states are drawn in raw observation
  coordinates, and on both paths they pass through the same input as the data
  (symlog for the pre-release checkpoints).
- The gate and both controls run in the collector's environment (tdmpc2's pinned
  environment of (a)), on the same device (the T4), on the same loaded `TDMPC2`
  object that is then used for acting, with `torch.compile` configured as in (e). The
  gate tests the encoder that acts. In tdmpc2, `torch.compile` wraps `_plan`
  (`tdmpc2.py:50-51`), and `encode` runs inside it when acting; the gate calls
  `agent.model.encode` on that object directly, which runs the same modules and
  weights eagerly.
- Rule, on each state set: pass if the median relative error is ≤ 1e-5 and the
  maximum is ≤ 1e-3. Near the lens centre the LayerNorm Jacobian amplifies float32
  rounding, so a correct remap can exceed 1e-5 on a state close to z*, while a remap
  error gives errors of order one. Reported for each set: the median, 99th percentile
  and maximum relative error, and for the 5 worst states their distance to the first
  layer's lens centre, ‖x − z*‖₂ / r_eff(d₁), with x the layer input and d₁ from that
  checkpoint's D4 policy-prior observations, computed as in (g).
- Controls, run and reported before the pre-release gate:
  positive control: public cartpole-swingup seed 2, unmodified planner encoder versus
  the lens float64 public-layout forward pass, identity input; it must pass the same
  rule. Negative control: cartpole-swingup seed 1 remapped but read with identity at
  position 0 (no symlog on the planner side), against the lens float64 forward with
  symlog; it must fail by a wide margin, with maximum relative error > 1e-2. If either
  control does not behave as stated, stop and file a deviation.
- The gate itself: both pre-release checkpoints (remap + symlog) must pass the rule
  on both state sets. If either fails, stop and file a deviation; no remapped planner
  run happens until the gate passes. The gate checks the remap and the layer
  ordering, not the reading of the activations or the architecture; that is what the
  ≥ 0.9 × published condition below tests.
- If the remapped checkpoint cannot be made to load and run, the fallback is
  policy-prior acting (D4) under symlog, with the same two conditions below.
- Confirmation for cartpole-swingup seed 1 (gate): (i) planner return
  ≥ 0.9 × published under `eval_mode=True` (policy-prior return under the
  fallback), the return being the mean over the checkpoint's 50 episodes, as for
  the 0.5 flag in (a), AND (ii) on the new planner data, symlog gives the lowest e
  among the three D3 candidates, with e/e₀ ≤ 0.1 (D3's computation). Condition (ii)
  is binding on the planner data. If either condition fails, cartpole-swingup seed 1
  is unidentified and cartpole-swingup's G1 vote uses seed 2.
- Because the new data are collected with symlog driving the agent, condition (ii)
  is also reported on the earlier policy-prior data, collected with identity driving
  the agent. This is for the record only; the rule above is unchanged.
- humanoid-run seed 3 stays reported only, with no effect on G1. D3 and D5 showed
  that a correctly read humanoid-run network gives e/e₀ ≈ 0.117 (public seed 1), so
  the 0.1 threshold does not suit humanoid-run and is not applied to it. Its label:
  "confirmed" if condition (i) holds (planner return as the mean over its 50
  episodes) and symlog gives the lowest e on the planner data; "identified by
  consistency" if symlog gives the lowest e on the planner data but condition (i)
  fails, so that consistency alone (with the policy-prior D3 result) supports
  symlog; "unidentified" if symlog does not give the lowest e on the planner data.
  Its e/e₀ values are reported next to the humanoid-run seed 1 calibration value
  from (c).

### (c) Calibration on the planner data

The D3 computation (three candidates, e and e₀) is rerun on the planner data of
cheetah-run seed 1, walker-run seed 1, humanoid-run seed 1 and cartpole-swingup
seed 2. Reported only; it changes no rule or threshold.

### (d) The policy-prior data

The policy-prior data already collected (D4) is kept as a secondary dataset. It is
not used for G1.

### (e) Collection protocol

- All 15 checkpoints: the 13 public-layout ones under (a), the two pre-release ones
  under (b) once the encoder agreement gate has passed.
- 50 episodes per checkpoint: environment seeds 0–4, 10 consecutive episodes each.
  For environment seed s, tdmpc2's `make_env` is built with `seed = s` (DMControl
  `task_kwargs={'random': s}`) and tdmpc2's `set_seed(s)` is called once before its
  10 episodes. Each episode is 500 agent steps (tdmpc2's `Timeout`, action repeat 2).
- Kept per episode: all 501 raw observations (the reset observation and the
  observation after each agent step, float32 as returned by tdmpc2's environment),
  the 500 executed actions and the 500 rewards. This is r6.md's Data section
  unchanged (50 episodes, 10 per environment seed 0–4, every observation of every
  step kept): 25,050 observations per checkpoint.
- The criterion stage applies r6.md's definitions unchanged to each checkpoint's
  25,050 observations: μ, C, the kept dimensions, d₁ and D from all of them, and
  Populated's subsample `numpy.random.default_rng(0).choice(n, 5000, replace=False)`
  with n = 25,050.
- Device: an NVIDIA Tesla T4 on Colab (the GPU name is recorded; a different GPU is
  reported as such). `torch.compile` as in tdmpc2's configuration.
- Saved with every checkpoint's data: our configuration, tdmpc2's configuration, the
  git commit and dirty state, the prereg tags, the tdmpc2 commit, the package
  versions (torch, tensordict, torchrl, mujoco, dm_control, numpy), the GPU and CUDA
  version, the checkpoint's file name and SHA-256, and for the pre-release ones the
  key mapping and the encoder agreement results. Results are copied to Google Drive
  after each checkpoint. Observation data is never committed: it stays on Drive, and
  a manifest of its file names and SHA-256 hashes is written to `results/r6/` and
  committed. Because `.gitignore` ignores `results/`, every stage lists its output
  files with the exact `git add -f` command in its run summary, so no result file is
  left uncommitted.

### (f) G1

Otherwise unchanged, including humanoid-run.

### (g) End-to-end susceptibility (reported only)

Reported only: it changes no rule and does not affect G1. Specified now; computed at
the criterion stage, not before.

Rationale. The encoder is Linear → LayerNorm → Mish (layer 1, the layer analysed),
then Linear → LayerNorm → SimNorm (layer 2); SimNorm replaces only the last
activation. The observation (symlog(o) for the pre-release checkpoints) enters the
model only through layer 1, with no skip connection. By Corollary 1
(`docs/theory.md`), along a line x(t) = z* + t·d the layer-1 LayerNorm output depends
on t only through θ(t) = arctan(t / r*(d)): its normalised output before γ and β is
ĥ(t) = √H (cos θ ĉ + sin θ q̂) / √(1 + κ cos² θ), with ĉ = c⊥/‖c⊥‖ and
q̂ = A d/‖A d‖ (the line passes through z*, so its own closest approach is t = 0 and
its c⊥ and κ are the lens's). So for any downstream output y, exactly,
dy/dt = g′(θ) · r*(d) / (r*(d)² + t²): the layer-1 Lorentzian times a learned factor
g′. SimNorm and everything after it only shape g′.
This measures whether the lens reaches TD-MPC2's outputs. It does not show that those
slopes are wrong: the latent has no ground truth.

Specification.

- Data: each checkpoint's planner data (all 25,050 observations), in layer-1 input
  coordinates (symlog(o) for the two pre-release checkpoints, whatever label they end
  up with under (b)). Float64 throughout, with the lens code on the original weights.
  All 15 checkpoints, with any < 0.5 × published flag from (a) shown alongside.
- Directions (unit vectors, in layer-1 input coordinates): d₁ as defined in r6.md,
  "the unit eigenvector of C with the largest eigenvalue", with μ and C the sample
  mean and covariance of all observations of the checkpoint (numpy.cov, ddof = 1);
  d₂, defined here because r6.md defines only d₁: the unit eigenvector of C with the
  second-largest eigenvalue; and u_min, the lens's narrowest principal direction (the
  right singular vector of A for the width width_min).
- Kept dimensions: d₁ and d₂ are computed exactly as r6.md computes d₁. r6.md's
  kept-dimension rule is "Kept dimensions: eigenvectors of C whose eigenvalue is at
  least 1e-10 times the largest. m is their number. C⁺ is the pseudo-inverse of C
  restricted to them." It enters only m and C⁺ (Inside, Populated). r6.md has no
  kept-dimension step for d₁: d₁ is an eigenvector of the full C, already a vector in
  the full layer-1 input space, so no projection and no zero-padding is applied. d₂ is
  computed the same way. If d₂'s eigenvalue is below the kept-dimension threshold
  (1e-10 times the largest), d₂ is reported as degenerate for that checkpoint and its
  (g) quantities are not computed.
- Lines: x(t) = z* + t·d, with 2,001 points uniform in θ over
  |θ| ≤ arctan(T / r*(d)), where T = max(max_i |⟨x_i − z*, d⟩|, 3·r_eff(d)) over the
  data states x_i, so that the grid also covers |t| ≤ 3·r_eff(d).
- Outputs: (i) the encoder output after SimNorm; (ii) the dynamics prediction with
  the action held fixed at a = 0 (not differentiated). Slopes J(t) are
  Jacobian-vector products along d, by autodiff.
- Check of the Corollary 1 form: the layer-1 LayerNorm output before its affine
  parameters γ and β (theory.md's ĥ) along each line matches the closed form above to
  1e-10 relative (float64); the maximum deviation is reported. Then
  g′(θ) = J(t) · (r*(d)² + t²) / r*(d).
- Reported per checkpoint, direction and output:
  - ‖g′(0)‖;
  - S = ‖g′(0)‖ / r*(d), with the r_eff(d) version ‖g′(0)‖ / r_eff(d) beside it (they
    differ by a factor √(1 + κ), at most 0.1% since κ ≤ 0.0017);
  - the peak ‖J(t)‖ over the grid points with |t| ≤ 3·r_eff(d), and the t where it
    occurs;
  - the median over data states of ‖J(x_i) d‖ with a = 0, and a secondary version
    using each state's recorded executed action;
  - the ratio of the peak to that median (a = 0).
- dog-run: z* lies far outside the data (‖z*‖ 563–769); it is reported as is.
