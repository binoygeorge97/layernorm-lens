# Deviations from the R6 pre-registration

Date: 24 September 2026. Author: Binoy George.
Applies to `prereg/r6.md` as tagged `prereg-r6` (tag object 086fbdd, commit 66b3a09),
which is unchanged. Each deviation below was decided before any observation was
collected and before any criterion quantity was computed.

## Background

The released single-task checkpoints (Hugging Face `nicklashansen/tdmpc2`, revision
8fb2a82; checkpoint metadata date 2023-06-03, `num_enc_layers=2`, `enc_dim=256`,
`mlp_dim=512`, `latent_dim=512`, `simnorm_dim=8`, `dropout=0.01`) use a module layout
that no public code builds or converts:

- `github.com/nicklashansen/tdmpc2` at e9f59321 (all 93 commits, all branches) and
  `github.com/tdmpc2/tdmpc2-eval` at ab776b65 (all 3 commits) build the encoder from
  `NormedLinear` (keys `_encoder.state.0.weight`, `_encoder.state.0.ln.*`), load with
  a strict `load_state_dict`, and convert only Q-ensemble keys
  (tdmpc2 `common/layers.py:167-221`). tdmpc2-eval's `load()` also requires a
  `model_target` entry, which the released files do not contain.
- The released encoder is `_encoder.state.{0..6}`: 0 no parameters or buffers;
  1 Linear (256, k); 2 LayerNorm-shaped (256,); 3 no parameters; 4 Linear
  (512, 256); 5 LayerNorm-shaped (512,); 6 no parameters.

## D1. The layer

Pre-registered: "the encoder's first NormedLinear".
Instead: E and b are `_encoder.state.1.weight` and `.bias`; the LayerNorm is
`_encoder.state.2` (its γ and β are recorded; they do not enter the lens).
`experiments/r6_tdmpc2/extract.py` stops on any other layout.

## D2. The LayerNorm ε

Pre-registered: ε "read from the checkpoint and the code, never assumed".
Neither is possible: a state_dict does not store ε, and no public code builds this
layout. We use ε = 1e-5, PyTorch's `nn.LayerNorm` default and the value every public
TD-MPC2 version uses (`nn.LayerNorm(self.out_features)`). This is an assumption. For a
non-degenerate lens, ε enters r_eff and ρ_eff only through κ = Hε/‖c⊥‖²; r*, κ and
‖c⊥‖ are reported, so every ε-dependent quantity can be recomputed for another ε.

## D3. Encoder position 0

Pre-registered: coordinates are "exactly what the layer receives".
Position 0 has no parameters or buffers, and no public code identifies it. It is
identified by a comparative latent-consistency test before any criterion quantity is
computed.

- Candidates for encoder position 0, each applied to the flattened observation o:
  (i) identity; (ii) symlog(o) = sign(o)·log(1 + |o|), elementwise; (iii) LayerNorm
  without affine parameters over the observation vector, ε = 1e-5.
- Every other parameter-free position follows the public code: encoder positions 3
  (Mish) and 6 (SimNorm, simnorm_dim = 8); dynamics `_dynamics.0.{2,5}` (Mish) and
  SimNorm after `_dynamics.1`; the dynamics input is the concatenation [z, a].
- For each task, seed 1, on all collected transitions (o_t, a_t, o_{t+1}) and for each
  candidate c: e_c = mean ‖d(enc_c(o_t), a_t) − enc_c(o_{t+1})‖², and the baseline
  e₀ = the same mean with o_{t+1} replaced by the observation at index j_t, where
  j = numpy.random.default_rng(0).integers(0, n, size=n) over the task's n
  transitions, computed under identity. e₀ under each other candidate, with the same
  j, is reported too.
- Identity is accepted for a task if it gives the lowest e_c AND e_identity ≤ 0.1·e₀.
- If another candidate's e_c is within 10% of identity's (e_c ≤ 1.1·e_identity),
  R6 is reported under both. Such a task counts toward gate G1 only if the
  criterion holds under both coordinate systems; for every other task, identity
  alone decides.
- If identity is rejected for any task, stop and consult the author. Any other
  hypothesis is a further, separately dated deviation.
- This test checks the whole assumed stack jointly (Mish, SimNorm, ε, the dynamics
  layout and position 0), not position 0 alone.

## D4. Acting policy for data collection

Pre-registered: "the released agent acting as in the repository's `evaluate.py`".
Neither repository's `evaluate.py` can load the released files (Background), and the
planner is not reimplemented. The agent acts with its policy prior alone: the
deterministic action a_t = tanh(μ(z_t)), where μ is the first half of `_pi`'s output
and z_t is the encoder output. This is the evaluation-mode action of the public code
(tdmpc2 `common/world_model.py:154`, `common/math.py:25`, `tdmpc2.py:118-119`;
tdmpc2-eval `tdmpc2.py:72, 82`). `_pi`'s keys are read from the checkpoint and its
layout is checked like the encoder's; its parameter-free positions (Mish) follow the
public code, as in D3.

Consequences, reported with the results:

- The states visited are those of the policy prior, not of the planning agent, so the
  data distribution (μ, C, D, the Populated and Inside criteria) may differ from what
  the planning agent would visit.
- The acting encoder uses candidate (i), identity, at position 0. The same collected
  data serve every candidate in D3.
- Policy-only returns (mean over each task's 50 episodes, per seed) are reported
  against the published return: the reward at the last step logged for the same seed
  in `results/tdmpc2/<task>.csv` of tdmpc2 e9f59321 (evaluated with the planner).
  They do not gate anything. Any task whose policy-only return is below 50% of the
  published value is flagged in the results.
- Everything else in the Data section is unchanged: 50 episodes per task, 10 with each
  environment seed 0–4, every observation of every step kept, with the public
  DMControl conventions (observation dict flattened in spec order as float32; action
  repeat 2; 500 agent steps; actions scaled to [−1, 1]).

## Unchanged

The question, tasks, seeds (seed 1 primary; seeds 2 and 3 analysed and reported but
not counted), definitions, criterion, gate G1 and the list of quantities reported
regardless of outcome are as tagged.
