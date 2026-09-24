# layernorm-lens

Research code for a TMLR paper: LayerNorm after an affine map is a gnomonic
projection, which creates a *lens* (a Cauchy-shaped zone of amplified slopes) in
learned dynamics models.

- `docs/theory.md`: definitions and results the code implements (authoritative)
- `docs/plan.md`: what each experiment must produce
- `CLAUDE.md`: rules for working in this repo (float64, frozen core, pre-registration)

## Layout

| Path | Contents |
| --- | --- |
| `lens/` | `core.py` (the original kink_core.py, frozen), `geometry.py` (lens from weights) |
| `plants/` | toy systems and the quadrotor |
| `control/` | LQR and MPC |
| `experiments/<test>/` | one folder per test: `config.yaml` + `run.py` |
| `prereg/` | dated pre-registrations (never edited after tagging) |
| `tests/` | pytest suite; `tests/golden/` holds the core.py regression outputs |
| `figures/` | the only source of paper figures |
| `notebooks/` | exploration only |
| `slurm/` | TACC job scripts |

`results/`, `checkpoints/` and `data/` are git-ignored.

## Install

```bash
pip install -r requirements.txt           # CPU
pip install "jax[cuda12]==0.10.2"          # GPU nodes (CUDA 12), same JAX version
```

## Tests

```bash
python -m pytest -q
LENS_TOL=1 python -m pytest -q tests/test_core_regression.py   # other machines
```

`tests/test_core_regression.py` checks `lens/core.py` against the golden files
bit for bit. That is only meaningful in the environment recorded in
`tests/golden/manifest.json` (JAX 0.10.2, numpy 2.4.6, scipy 1.17.1, CPU). On any
other machine use `LENS_TOL=1`, which compares with rtol = 1e-12. To regenerate the
goldens after a reviewed change: `python -m tests.golden.make_golden --force`.
