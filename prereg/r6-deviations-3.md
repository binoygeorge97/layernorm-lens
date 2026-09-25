# Deviations from the R6 pre-registration, part 3

Date: 24 September 2026. Author: Binoy George.
Applies to `prereg/r6.md` (tag `prereg-r6`, tag object 086fbdd, commit 66b3a09),
`prereg/r6-deviations.md` (tag `prereg-r6-d1`, tag object 3a0bb10, commit e5060cf)
and `prereg/r6-deviations-2.md` (tag `prereg-r6-d2`, tag object df32edc, commit
e20f5a6), all unchanged. Decided after the D3 consistency test, the policy-prior
returns and the planner feasibility check below, and before any R6 criterion
quantity (Inside, Populated, Sharp) or gate G1 was computed.

## Evidence that led to D6

Recorded in `results/r6/`:

- D3 (`consistency.csv`): identity was rejected for both pre-release checkpoints.
  cartpole-swingup seed 1: e/e₀ identity 0.036, symlog 0.020 (lowest), LayerNorm
  0.046. humanoid-run seed 3: identity 0.989 (the random-pairing level), symlog
  0.115 (lowest), LayerNorm 0.965. Public-layout calibration, identity: cheetah-run
  0.014, walker-run 0.013, humanoid-run seed 1 0.117.
- Policy-prior returns (`returns.csv`, D4): cartpole-swingup seed 1, acting with
  identity at position 0, reached 0.20 of its published return, while its public
  seeds 2 and 3 reached 1.00. humanoid-run reached at most 0.012, dog-run at most
  0.075, cheetah-run 0.38–0.48 of the published returns.
- Planner feasibility (`planner_check.csv`, `meta_planner_check.json`; public
  tdmpc2 `TDMPC2` class at e9f59321, 5 episodes, Tesla T4): with `eval_mode=True`,
  cartpole-swingup seed 2 returned 883.2 (1.001 of the published 882.5) and
  humanoid-run seed 1 returned 672.8 (1.241 of 542.2); with `eval_mode=False`,
  882.2 (1.000) and 616.8 (1.138).
  About 9 s (cartpole) and 13 s (humanoid) per episode after a first episode of
  about 2 minutes that includes `torch.compile`.

## D6. Planner data; symlog at pre-release position 0

### (a) Data for the 13 public-layout checkpoints

- Observations are collected with the public tdmpc2 planner at e9f59321 (its own
  `TDMPC2` class, checkpoint loaded through `TDMPC2.load`, including
  `api_model_conversion`), with `eval_mode=True`: the planner's noise-free mean
  action.
- This departs from r6.md's literal "acting as in the repository's `evaluate.py`":
  `evaluate.py` calls `agent.act(obs, t0=t==0, task=task_idx)` without `eval_mode`,
  so the executed action includes the planner's exploration noise
  (tdmpc2 `tdmpc2.py:203`). `eval_mode=True` is how the published learning curves
  were produced (`trainer/online_trainer.py:37`) and how the deployed agent acts.
- 50 episodes per checkpoint, 10 with each environment seed 0–4 (the seed of
  `make_env` and of tdmpc2's `set_seed`), every observation of every step kept
  (501 per episode, reset included), with the public DMControl conventions of D4.
- The planner runs in float32 PyTorch, in tdmpc2's pinned environment
  (`docker/environment.yaml`), for acting only. All lens and criterion computations
  are float64, as CLAUDE.md requires.
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
- These checkpoints are run through the same planner as (a), with symlog applied
  to the observation before the encoder, after a key remap to the public layout.
  The remap is fixed by the keys of both checkpoints (`results/r6/prerelease_keys.json`,
  60 keys each). Each pre-release Linear becomes the public `NormedLinear`'s Linear
  and the LayerNorm-shaped module after it becomes that `NormedLinear`'s `ln`:

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
- If the remapped checkpoint cannot be made to load and run, the fallback is
  policy-prior acting (D4) under symlog, with the same two conditions below.
- Confirmation for cartpole-swingup seed 1 (gate): (i) planner return
  ≥ 0.9 × published under `eval_mode=True` (policy-prior return under the
  fallback), AND (ii) on the new data, symlog gives the lowest e among the three D3
  candidates, with e/e₀ ≤ 0.1 (D3's computation). If either fails, cartpole-swingup
  seed 1 is unidentified and cartpole-swingup's G1 vote uses seed 2.
- Because the new data are collected with symlog driving the agent, condition (ii)
  is also reported on the earlier policy-prior data, collected with identity
  driving the agent. This is for the record only; the rule above is unchanged.
- humanoid-run seed 3 stays reported only. It is marked "confirmed" if it also
  passes condition (i), otherwise "identified by consistency". Its consistency
  results on the new data are reported.

### (c) Calibration on the planner data

The D3 computation (three candidates, e and e₀) is rerun on the planner data of
cheetah-run, walker-run and humanoid-run seed 1 and of cartpole-swingup seed 2.
Reported only; it changes no rule or threshold.

### (d) The policy-prior data

The policy-prior data already collected (D4) is kept as a secondary dataset. It is
not used for G1.

### (e) G1

Otherwise unchanged, including humanoid-run.
