#!/bin/bash
# P-I training on TACC, one Slurm array task per run (the fallback to the laptop;
# docs/DECISIONS.md). Set up first as in slurm/README.md. Submit from the repository:
#
#   jid=$(sbatch --parsable --array=0-39 slurm/p1_array.slurm train)       # the 40-model grid
#   sbatch --dependency=afterok:$jid slurm/p1_gather.slurm train
#   jid=$(sbatch --parsable --array=0-9 slurm/p1_array.slurm train_long)   # the long-budget subset
#   sbatch --dependency=afterok:$jid slurm/p1_gather.slurm train_long
#
# Array task i runs member i of the stage's grid (run.py grid() order). run.py refuses to
# run unless the annotated tag prereg-p1 exists and prereg/p1.md matches it, the working
# tree is clean, and the data match the committed SHA-256 manifest. Each run writes its
# outputs under $P1_OUT, with its git commit, versions, config and Slurm identity in
# results/p1/<stage>/runs/<name>.json and a SHA-256 manifest beside it.
#
#SBATCH -J p1
#SBATCH -p skx                  # EDIT: a Stampede3 CPU queue (job arrays are supported on Stampede3)
#SBATCH -A EDIT_ALLOCATION      # EDIT: the TACC allocation
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 24:00:00             # EDIT: one run's upper bound with margin (slurm/README.md)
#SBATCH -o p1-%x-%A_%a.out
#SBATCH -e p1-%x-%A_%a.err

set -euo pipefail
STAGE=${1:?usage: sbatch --array=0-N slurm/p1_array.slurm train|train_long}
case "$STAGE" in train|train_long) ;; *) echo "stage must be train or train_long" >&2; exit 2 ;; esac
TASK=${SLURM_ARRAY_TASK_ID:?not a Slurm array task: submit with --array}
REPO=${P1_REPO:-$WORK/layernorm-lens}
OUT=${P1_OUT:-$SCRATCH/layernorm-lens}
VENV=${P1_VENV:-$REPO/.venv}

cd "$REPO"
COMMIT=$(git rev-parse HEAD)
if [ -n "$(git status --porcelain --untracked-files=no)" ]; then
    echo "refusing to run: uncommitted changes to tracked files in $REPO" >&2
    exit 1
fi
echo "commit $COMMIT stage $STAGE job ${SLURM_ARRAY_JOB_ID:-?} task $TASK host $(hostname) start $(date -u +%FT%TZ)"
# shellcheck disable=SC1091
source "$VENV/bin/activate"
export PYTHONUTF8=1
python experiments/p1_quadrotor/run.py --config experiments/p1_quadrotor/config.yaml "$STAGE" \
    --index "$TASK" --out-root "$OUT"
echo "commit $COMMIT stage $STAGE task $TASK done $(date -u +%FT%TZ)"
