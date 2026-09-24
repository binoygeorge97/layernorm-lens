# Deviations from the R6 pre-registration, part 2

Date: 24 September 2026. Author: Binoy George.
Applies to `prereg/r6.md` as tagged `prereg-r6` (tag object 086fbdd, commit 66b3a09)
and `prereg/r6-deviations.md` as tagged `prereg-r6-d1` (tag object 3a0bb10, commit
e5060cf), both unchanged. Decided after inspecting the checkpoints' keys and
metadata, before any observation was collected and before any criterion quantity
was computed.

## D5. Two checkpoint layouts

### Survey

All 15 configured checkpoints at Hugging Face revision 8fb2a82:

| Task | k | Seed 1 | Seed 2 | Seed 3 | Metadata date |
| --- | --- | --- | --- | --- | --- |
| cartpole-swingup | 5 | pre-release | public | public | 2023-06-03 |
| cheetah-run | 17 | public | public | public | 2023-06-03 |
| walker-run | 24 | public | public | public | 2023-06-03 |
| humanoid-run | 67 | public | public | pre-release | 2023-06-03 |
| dog-run | 223 | public | public | public | 2023-06-03 |

- Pre-release layout: `_encoder.state.0` has no parameters or buffers; Linear
  `_encoder.state.1`, LayerNorm-shaped `_encoder.state.2` (as in D1).
- Public layout: `NormedLinear` modules, keys `_encoder.state.0.weight`, `.bias`,
  `.ln.weight`, `.ln.bias`, as built by tdmpc2 e9f59321.
- k is the input dimension of each checkpoint's first Linear.

D1 to D4 were written when only the pre-release layout had been seen, and assumed
it for every checkpoint.

### Public-layout checkpoints

Read as the public code builds them, tdmpc2 e9f59321 `common/layers.py`:

- `NormedLinear.forward` (lines 107-111): Linear, then dropout if any, then
  LayerNorm, then the activation. `mlp()` (lines 128-133): every layer but the last
  is a `NormedLinear` with Mish; the last is a `NormedLinear` with the given
  activation, or a plain `nn.Linear` if none.
- ε = 1e-5: line 101, `self.ln = nn.LayerNorm(self.out_features)`, with PyTorch's
  default ε.
- Encoder (line 159): `mlp(obs_dim, [enc_dim], latent_dim, act=SimNorm)`. The
  first layer is `_encoder.state.0`: E and b are `_encoder.state.0.weight` and
  `.bias`, the LayerNorm is `_encoder.state.0.ln`. There is no position 0 in the
  sense of D3; the layer receives the flattened observation.
- Dynamics and policy from the same layout (`common/world_model.py` lines 26
  and 29): `_dynamics = mlp(latent + action, 2·[mlp_dim], latent, act=SimNorm)`,
  `_pi = mlp(latent, 2·[mlp_dim], 2·action)`.
- The encoder, dynamics and policy have no dropout (`mlp()` is called without it).

D1, D2 and D3 apply only to the two pre-release checkpoints.

### D3 for the pre-release checkpoints

- D3's decision rule applies to cartpole-swingup seed 1, which counts toward G1,
  and to humanoid-run seed 3, which is reported only (seeds 2 and 3 never count
  toward G1).
- If identity is rejected for cartpole-swingup seed 1, stop and consult the
  author, as D3 states.
- If identity is rejected for humanoid-run seed 3, the analysis does not stop: that
  checkpoint's R6 results are reported under each candidate, and the checkpoint is
  marked unidentified.

### Calibration

The same D3 computation (candidates identity, symlog and LayerNorm without affine
parameters, each applied to the flattened observation before the first Linear;
e_c and e₀ with the same random indices) is run on the public-layout seed-1
checkpoints of cheetah-run, walker-run and humanoid-run. For these the public code
fixes identity as the correct input. Their e/e₀ are reported as a reference for
what a correctly read network gives. They change neither D3's rule nor its
threshold.

### D4

D4 applies to all 15 checkpoints, whichever layout they use: acting with the policy
prior alone, a_t = tanh(μ(z_t)), with the encoder read according to the
checkpoint's layout (identity at position 0 for the pre-release layout).

### Layout detection

Each checkpoint's layout is detected from its keys: public if
`_encoder.state.0.weight` and `_encoder.state.0.ln.weight` are present, pre-release
if nothing is stored under `_encoder.state.0` and `_encoder.state.1.weight` and
`_encoder.state.2.weight` are. Any other layout, or a detected layout that differs
from the survey above, stops the analysis. The detected layout is recorded with the
results.

### G1

Unchanged.
