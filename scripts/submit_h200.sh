#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
sbatch --gpus=1 -p "${PARTITION:-hp_a800}" --time=00:10:00 ./run.sh
