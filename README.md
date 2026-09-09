# EvoMax-Enhanced

## Verification status (2026-09-09)

This repository provides a reproducible, no-label model-only EvoMax candidate
workflow. It is not a claim of experimental binding performance or paper-level
ranking parity: no project-specific experimental labels are included for GPR training.
The current ESM-2 adapter uses 3B masked log-odds; it is not the paper's 650M baseline.

Deployment: `/data/home/scwb286/EvoMax-Enhanced`.
`run.sh` now tests actual ESM-2 checkpoint loading and inference as well as CUDA.
ESM-IF1 input tensors are moved to the model device. The uploaded checkpoint is
configured at `/data/run01/scwb286/esm2_deploy/weights/ESM-IF1/esm_if1_20220410.pt`.
Run `python scripts/if1_model_smoke.py` on a GPU node to verify loading.
The project-level `torch_scatter` and Biotite compatibility shims are used only
for the installed fair-esm dependency set and are covered by the GPU smoke tests.
The existing `fair-esm_test` scripts use ESMFold v1, not ESM-IF1.

This project keeps a literature-comparable EvoMax baseline separate from optional enhancements.

Baseline flow:

1. Enumerate all `L × 19` single mutants.
2. Batch-score all mutants with GPR and ESM-2 on GPU.
3. Robust median/IQR-normalize each channel.
4. Keep the top 1.5%.
5. Batch-rerank only that shortlist with ESM-IF and retain GPR/ESM-2 contributions.

ESM-C, ESM-1v, ProteinMPNN, and EVOLVEpro must be optional calibrated channels with separate ablations. ESMFold2 is a structure predictor and is not a drop-in replacement for ESM-IF.

The optional ESM-C adapter is `evomax_enhanced.plm.esmc.ESMCMaskedScorer`.
It was loaded successfully on A800 in job `157975` with the cluster's
`esm2_v2` environment (`transformers` 4.57.6); the older `fair-esm`
environment remains the validated ESM-2/ESM-IF1 runtime.
Foundry ProteinMPNN is exposed through
`evomax_enhanced.inverse_folding.ProteinMPNNRunner`; it generates or filters
structure-compatible sequences and is not treated as an affinity predictor.
RF3/OpenDDE/tFold remain downstream structure/interface gates and must be
invoked using the installed Foundry/OpenDDE revision on the cluster.

The H200 model cache is referenced by configuration; weights and project data are not copied into this repository.

## Environment

The reproducible CPU/package specification is in `environment.yml`. On the
cluster, the validated GPU runtime is the existing
`/data/home/scwb286/.conda/envs/fair-esm` environment, which supplies
CUDA-enabled PyTorch and fair-esm. Model checkpoints remain outside GitHub and
are referenced by `configs/h200.yaml` and private project runners.

## Project-specific data

Project-specific antibody sequences, candidate libraries, structures, model
outputs and mutation rankings are intentionally excluded from this public
repository. Run the generic pipeline with private data mounted on the cluster.

## H200 smoke test

From the project directory on H200:

```bash
sbatch --gpus=1 -p gpu_h200 ./run.sh
squeue -u "$USER"
sacct -j <job_id> --format=JobID,State,Elapsed,ExitCode
```
