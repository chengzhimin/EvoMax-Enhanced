# EvoMax-Enhanced

## Verification status (2026-09-09)

This is an incomplete prototype, not yet a validated reproduction of EvoMax.
Core smoke checks use synthetic score providers. GPR training, a complete CLI,
ESM-C/EVOLVEpro integration, and paper-level ranking parity remain outstanding.
The current ESM-2 adapter uses 3B masked log-odds; it is not the paper's 650M baseline.

Deployment: `/data/home/scwb286/EvoMax-Enhanced`.
`run.sh` now tests actual ESM-2 checkpoint loading and inference as well as CUDA.
ESM-IF1 input tensors are moved to the model device. The uploaded checkpoint is
configured at `/data/run01/scwb286/esm2_deploy/weights/ESM-IF1/esm_if1_20220410.pt`.
Run `python scripts/if1_model_smoke.py` on a GPU node to verify loading.
The existing `fair-esm_test` scripts use ESMFold v1, not ESM-IF1.

This project keeps a literature-comparable EvoMax baseline separate from optional enhancements.

Baseline flow:

1. Enumerate all `L × 19` single mutants.
2. Batch-score all mutants with GPR and ESM-2 on GPU.
3. Robust median/IQR-normalize each channel.
4. Keep the top 1.5%.
5. Batch-rerank only that shortlist with ESM-IF and retain GPR/ESM-2 contributions.

ESM-C, ESM-1v, ProteinMPNN, and EVOLVEpro must be optional calibrated channels with separate ablations. ESMFold2 is a structure predictor and is not a drop-in replacement for ESM-IF.

The H200 model cache is referenced by configuration; weights are not copied into this repository.

## H200 smoke test

From the project directory on H200:

```bash
sbatch --gpus=1 -p gpu_h200 ./run.sh
squeue -u "$USER"
sacct -j <job_id> --format=JobID,State,Elapsed,ExitCode
```
