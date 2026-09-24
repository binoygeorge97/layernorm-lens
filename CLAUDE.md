Research code for a TMLR paper: LayerNorm after an affine map is a gnomonic
projection, which creates a "lens" (a Cauchy-shaped zone of amplified slopes) in
learned dynamics models. Read `docs/theory.md` for definitions and `docs/plan.md`
for what each experiment must do.

## Rules

- Use float64 everywhere: call `jax.config.update("jax_enable_x64", True)` at the
  top of every entry point.
- Never modify `lens/core.py` unless `tests/test_core_regression.py` still passes
  bit for bit afterwards.
- Never edit a file in `prereg/` after it has been git-tagged (tags look like
  `prereg-r6`, `prereg-p1`). Scripts for a pre-registered test must refuse to run
  unless that tag exists.
- Cloud sessions cannot push tags. For a pre-registration, commit and push the
  file, then stop and ask the author to create and push the annotated tag from
  their own machine.
- Paper figures come only from scripts in `figures/`. Notebooks are exploration only.
- Every experiment run saves, next to its results: its config, the git commit hash,
  and the JAX (and PyTorch, if used) version.
- Take all definitions from `docs/theory.md`. If anything there is ambiguous or
  seems wrong, stop and ask. Never change a definition to make numbers agree.
- Do not compute an experiment's outcome metric before its pre-registration is
  tagged. Pipeline checks (training converges, tests pass) are fine.
- Large files (checkpoints, raw data, raw results) go in `checkpoints/`, `data/` and
  `results/`, which are git-ignored. Commit only small summary CSVs.
- Plan first for any task touching more than one file; wait for approval.

## Layout

- `lens/` library: `core.py` (original kink_core.py, frozen), `geometry.py`
  (lens from weights), `models.py`, `train.py`
- `plants/` toy systems and the quadrotor; `control/` LQR and MPC
- `experiments/<test>/` one folder per test: `config.yaml` + `run.py`
- `prereg/` dated pre-registrations; `tests/` pytest suite; `slurm/` TACC scripts

## Commands

- Run tests: `pytest -q`
- Experiments: `python experiments/<test>/run.py --config experiments/<test>/config.yaml`
