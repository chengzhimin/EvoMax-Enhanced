set -euo pipefail
#!/usr/bin/env bash
source /etc/profile
set -euo pipefail
module load miniforge3/26.3.2-3
conda activate py311
python -m pytest -q
python scripts/gpu_smoke.py
