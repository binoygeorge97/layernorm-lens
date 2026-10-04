Research code for a TMLR paper: LayerNorm after an affine map is a gnomonic
projection, which creates a "lens" (a Cauchy-shaped zone of amplified slopes) in
learned dynamics models. Read `docs/theory.md` for definitions and `docs/plan.md`
for what each experiment must do.

Read the project state: @docs/STATE.md

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
- If any check fails, stop and ask the author. Never change a threshold,
  tolerance or check on your own, even before results exist.
- Large files (checkpoints, raw data, raw results) go in `checkpoints/`, `data/` and
  `results/`, which are git-ignored. Commit (with `git add -f`) only summary CSVs,
  metadata JSON and small derived arrays, each under 1 MB. Observation data is never
  committed: it stays on Drive, and a manifest of its file names and SHA-256 hashes
  is committed instead.
- Plan first for any task touching more than one file; wait for approval.

## Git safety

- Never create or push tags. The author creates and pushes every tag.
- Never push to `main`, and never force-push any branch.
- Ask the author before every `git push`.
- Commit a prereg file alone: nothing else in that commit.
- Never edit a tagged file.

## Working on the Windows laptop

- The shell is PowerShell. The venv is `.venv`: activate it with
  `.\.venv\Scripts\Activate.ps1`.
- The core regression test is bit-exact only in the golden environment. If it
  fails here, run it in tolerance mode: `$env:LENS_TOL='1'; pytest -q`.
- Data stays outside the repo, at `G:\My Drive\layernorm-lens-r6` (Google Drive for
  desktop). Always verify every file against the committed SHA-256 manifests before
  using it.
- Every run records the package versions in its meta.

## Layout

- `lens/` library: `core.py` (original kink_core.py, frozen), `geometry.py`
  (lens from weights), `models.py`, `train.py`
- `plants/` toy systems and the quadrotor; `control/` LQR and MPC
- `experiments/<test>/` one folder per test: `config.yaml` + `run.py`
- `prereg/` dated pre-registrations; `tests/` pytest suite; `slurm/` TACC scripts

## Commands

- Run tests: `pytest -q`
- Experiments: `python experiments/<test>/run.py --config experiments/<test>/config.yaml`
