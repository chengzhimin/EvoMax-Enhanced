#!/usr/bin/env bash
# H200: sbatch --gpus=1 -p gpu_h200 ./run.sh
source /etc/profile
set -euo pipefail
module load miniforge3/26.3.2-3
conda activate fair-esm

export PYTHONPATH="$PWD/src"
echo "JOB_ID=${SLURM_JOB_ID:-local}"
echo "HOST=$(hostname)"
python scripts/gpu_smoke.py
python scripts/core_selfcheck.py
