# Plan: P-I revision queue (minibatch Adam, long-budget subset, TACC, draft v2)

Author-approved queue of 4 Oct 2026 (second queue); branch `quadrotor-sim`. As before,
nothing runs on quadrotor training data: every stage that touches it refuses to run
unless the annotated tag `prereg-p1` exists. Only the benchmark runs, on random targets.

## 1. Protocol amendments

`docs/protocol_amendments_v3.md`: the author's text, verbatim, committed alone.

## 2. Training and infrastructure

- `experiments/p1_quadrotor/config.yaml`: P-I's training default becomes minibatch Adam,
  batch 2,048. Held-out one-step MSE is measured on the full validation set every 500
  steps (the fixed evaluation interval), and early stopping is unchanged (patience 10% of
  steps, tolerance 1%). `lens/train.py`'s library default stays full batch, because
  synthetic tests use n < 2,048. `train` refuses a batch larger than the training set.
- Lens log along u_min: at every logging step, the current lens's narrowest principal
  direction u_min, D(u_min) from the training inputs, r*(u_min), r_eff(u_min),
  r_eff(u_min)/D(u_min) and S(u_min). This is prediction 5's primary quantity. d₁ stays
  as the secondary direction.
- The long-budget subset (stage `train_long`, gated): zero bias, 1 block, both
  architectures, seeds 0–4 (10 runs), a fixed budget with early stopping disabled. The lens
  is logged every 500 steps, and parameter snapshots are saved at a coarser interval, so
  that prediction 1's statistic can be traced over training after the tag. Both the final
  and the best parameters are kept. The budget is proposed from the minibatch benchmark.
- Prediction 5's rule as a pure function of a lens log (`p5_run`, `p5_cell`), tested on
  synthetic histories.
- One run per invocation: `--index i` selects the i-th member of a stage's grid.
  `--out-root DIR` puts data, checkpoints and results under DIR (for `$SCRATCH`). The
  data manifest is always the committed one. Each run writes its summary JSON and a
  SHA-256 manifest of its outputs. A `gather` stage (gated) merges the per-run summaries
  and manifests and verifies every file's SHA-256.
- `slurm/`: an array script (one array task per run: 40 for `train`, 10 for
  `train_long`), a gather script, and a README. Outputs go to `$SCRATCH`, the git commit
  is recorded, and the clean-tree and tag checks run before anything else.
- The benchmark runs again with batch 2,048 on random targets, into
  `results/p1/benchmark_bs2048/` (the full-batch benchmark stays in
  `results/p1/benchmark/`). It also times one lens record.
- Tests (synthetic only): minibatch default and refusal, u_min logging, `train_long` on
  the synthetic plant, `--index` and `--out-root`, per-run manifests and gather,
  prediction 5's rule, and the new gated stages refusing without the tag.

## 3. Draft v2 (`prereg/drafts/p1-draft.md`)

- Predictions 1–4 quoted exactly from the amendments, each with its operationalisation.
  The reconstructed hover prediction moves to the hover-check section as the ledger's
  new claim, and Corollary 1 becomes a pipeline check.
- Predictions 1 and 5 framed as a race between fitting and repair.
- The long-budget subset, prediction 5's u_min test, and initial values recomputed for
  u_min from initialisation alone.
- Drag off (a modelling choice), minibatch Adam, the evaluation interval and the
  budgets.
- Every open decision gets a recommendation, with reasoning and alternatives.

## 4. STATE.md, DECISIONS.md, push, report
