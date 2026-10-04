#!/bin/bash
# Verify and merge one P-I training stage's per-run outputs on $SCRATCH: every array task
# present, every file's SHA-256 against its run's manifest, one clean git commit for all
# runs. Writes results/p1/<stage>/train_summary.csv and outputs_manifest.csv under $P1_OUT.
# Submit after the array (see slurm/p1_array.slurm):
#
#   sbatch --dependency=afterok:<array job id> slurm/p1_gather.slurm train|train_long
#
#SBATCH -J p1-gather
#SBATCH -p skx                  # EDIT: as in p1_array.slurm
#SBATCH -A EDIT_ALLOCATION      # EDIT
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 00:30:00
#SBATCH -o p1-%x-%j.out
#SBATCH -e p1-%x-%j.err

set -euo pipefail
STAGE=${1:?usage: sbatch slurm/p1_gather.slurm train|train_long}
REPO=${P1_REPO:-$WORK/layernorm-lens}
OUT=${P1_OUT:-$SCRATCH/layernorm-lens}
VENV=${P1_VENV:-$REPO/.venv}

cd "$REPO"
echo "commit $(git rev-parse HEAD) gather $STAGE host $(hostname) start $(date -u +%FT%TZ)"
# shellcheck disable=SC1091
source "$VENV/bin/activate"
export PYTHONUTF8=1
python experiments/p1_quadrotor/run.py --config experiments/p1_quadrotor/config.yaml gather --of "$STAGE" \
    --out-root "$OUT"
