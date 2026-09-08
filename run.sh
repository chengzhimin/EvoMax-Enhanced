#!/usr/bin/env bash
# H200: sbatch --gpus=1 -p gpu_h200 ./run.sh
source /etc/profile
set -euo pipefail

export PYTHONPATH="$PWD/src"
PYTHON=/data/home/scwb286/.conda/envs/fair-esm/bin/python
echo "JOB_ID=${SLURM_JOB_ID:-local}"
echo "HOST=$(hostname)"
"$PYTHON" -V
"$PYTHON" -c 'import torch; print("TORCH", torch.__version__); print("CUDA", torch.cuda.is_available()); print("GPU", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NONE")'
"$PYTHON" scripts/gpu_smoke.py
"$PYTHON" scripts/core_selfcheck.py
