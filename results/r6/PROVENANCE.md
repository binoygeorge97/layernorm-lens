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
