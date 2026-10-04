# TACC scripts (P-I fallback)

The P-I grid runs on the laptop by default. These scripts are the fallback: one Slurm
array task per training run on TACC, with outputs on `$SCRATCH`. They target Stampede3,
which supports job arrays (Stampede2 did not). Edit the `-p`, `-A` and `-t` lines
before the first submission.

| Script | What |
| --- | --- |
| `p1_array.slurm STAGE` | One run per array task: `train` (`--array=0-39`, the 40-model grid) or `train_long` (`--array=0-9`, the long-budget subset). Task i runs member i of `run.py`'s `grid()` |
| `p1_gather.slurm STAGE` | After the array: verifies every run's outputs against their SHA-256 manifests and checks that every run used one clean commit. Then writes `train_summary.csv` and `outputs_manifest.csv` |

Nothing here runs before the tag: `run.py` refuses every quadrotor stage unless the
annotated tag `prereg-p1` exists and `prereg/p1.md` matches it.

## Once per system

```bash
cd $WORK
git clone <the repository URL> layernorm-lens && cd layernorm-lens
git fetch --tags && git checkout <the commit to run>   # tag prereg-p1 present locally
module load python3                                     # a Python 3.12 if available
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt                         # the pinned CPU versions
```

Every run records the installed versions in its run JSON, so a TACC run can be told
apart from a laptop run.

## Data

Generate once on the laptop (`run.py generate`, after the tag), and commit
`results/p1/data_manifest.csv`. Then copy `data/p1/*.npz` to
`$SCRATCH/layernorm-lens/data/p1/`. Every task verifies each file against the committed
manifest before it trains, and stops on a mismatch. To check the copy by hand first:

```bash
python - <<'PY'
import os, sys; sys.path.insert(0, "experiments/r6_tdmpc2"); import provenance as pv
d = os.path.expandvars("$SCRATCH/layernorm-lens/data/p1")
for r in pv.read_manifest("results/p1/data_manifest.csv"):
    pv.verify_sha256(os.path.join(d, r["file"]), r["sha256"], r["file"]); print("ok", r["file"])
PY
```

## Submit

```bash
jid=$(sbatch --parsable --array=0-39 slurm/p1_array.slurm train)
sbatch --dependency=afterok:$jid slurm/p1_gather.slurm train
jid=$(sbatch --parsable --array=0-9 slurm/p1_array.slurm train_long)
sbatch --dependency=afterok:$jid slurm/p1_gather.slurm train_long
```

Wall time: `results/p1/benchmark_bs2048/meta_benchmark.json` gives laptop upper bounds,
with no early stop. Run one task first (`--array=0`) to measure the TACC node's speed
against the laptop, then set `-t` with a margin.

Environment overrides: `P1_REPO` (default `$WORK/layernorm-lens`), `P1_OUT` (default
`$SCRATCH/layernorm-lens`) and `P1_VENV` (default `$P1_REPO/.venv`).

## Afterwards

- `$SCRATCH` is purged periodically (see the system's user guide), so copy the outputs
  off promptly.
- Results (`results/p1/train*/`: histories, run JSONs, manifests, summaries; each under
  1 MB) come back to the laptop, where the analysis stages run and the small files are
  committed with `git add -f`.
- Checkpoints (`checkpoints/p1/`) go to Drive. `outputs_manifest.csv` verifies them
  anywhere.
- The analysis stages (`analyse`, `analyse_long`, `hover`, `p5`) can also run on TACC,
  with `--out-root`.
