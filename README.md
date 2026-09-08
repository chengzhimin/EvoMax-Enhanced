# EvoMax-Enhanced

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
