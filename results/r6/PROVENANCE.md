# Provenance of the R6 CPU-run results

`returns.csv` (collect stage, deviation D4) and `consistency.csv` (consistency
stage, deviations D3 and D5) come from a Colab CPU run of
`experiments/r6_tdmpc2/r6_colab.ipynb`. Their `meta_collect.json` and
`meta_consistency.json` were lost when the Colab runtime reset, so the exact commit
and package versions of that run were not recorded. The two CSVs, uploaded by the
author and committed byte-identical in bd6ba66, are the surviving record.

Code commit: the author recalled c73d7cf. That commit cannot have produced these
files: its `collect.py` writes no `layout` column in `returns.csv` and no `role`,
`layout`, `identified` or `report_r6_under` columns in `consistency.csv`, and it
does not require the `prereg-r6-d2` tag. Those columns and that requirement first
appear in c85262f. `collect.py` and `layouts.py` are identical from c85262f to
bd6ba66, and `config.yaml` changed in that range only by the `planner_check`
section, which `collect.py` does not read. The run therefore used the code of
c85262f.

The planner feasibility run (`planner_check.csv`, `meta_planner_check.json`) has
its own complete metadata.

## Addendum, 25 September 2026

The CPU run's metadata was not lost: it was never committed, because `.gitignore`
ignores `results/` and the files were not force-added. The author recovered it from
the Drive copy of 24 September 2026 and committed it in 00603a8
(`meta_extract.json`, `meta_lens.json`, `meta_collect.json`,
`meta_consistency.json`, with `consistency.json`, `lens_summary.csv`, `lens/` and
`weights/`). The Drive copies of `returns.csv` and `consistency.csv` are identical to
those committed in bd6ba66.

All four meta files record git_commit 4c3f129475adceb0c6e90e56df6696b0565d4986 with
git_dirty false (extract 21:58:22, lens 21:58:50, collect 22:13:18, consistency
22:14:44 UTC). The CPU run therefore used commit 4c3f129, not c85262f as stated above.

- 4c3f129 is on origin, in `claude/loving-johnson-wk9yb3` and `main`.
- c85262f is an ancestor of 4c3f129; between them are a0b6807 (tolerance-mode
  regression test), e3dcda7 (initial-lens config) and 4c3f129 (initial-lens results).
- `git diff --stat c85262f 4c3f129` touches only
  `experiments/initial_lens/config.yaml`, `results/initial_lens/meta.json`,
  `results/initial_lens/summary.csv` and `tests/test_core_regression.py`; nothing
  under `experiments/r6_tdmpc2/` or `lens/`. The R6 code that ran is therefore
  identical to c85262f's, which is what the column-based inference above found; the
  commit it named was the earliest one with that code, not the one recorded.
