#!/usr/bin/env bash
#SBATCH --job-name=layernorm-lens
#SBATCH --array=0-6

python experiments/r1a/run.py
