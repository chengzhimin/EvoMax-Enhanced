#!/usr/bin/env python3
"""Load the deployed ESM-IF1 checkpoint and verify its device placement."""

from pathlib import Path
from argparse import Namespace

CHECKPOINT = Path(
    "/data/run01/scwb286/esm2_deploy/weights/ESM-IF1/esm_if1_20220410.pt"
)


def main() -> None:
    import torch
    import esm

    if not CHECKPOINT.is_file():
        raise FileNotFoundError(CHECKPOINT)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the ESM-IF1 smoke test")

    with torch.serialization.safe_globals([Namespace]):
        model, _ = esm.pretrained.load_model_and_alphabet_local(str(CHECKPOINT))
    model = model.to("cuda").eval()
    parameter = next(model.parameters())
    print("ESMIF1_MODEL_SMOKE_OK")
    print(f"checkpoint={CHECKPOINT}")
    print(f"parameter_device={parameter.device}")
    print(f"parameter_dtype={parameter.dtype}")
    print(f"gpu={torch.cuda.get_device_name(0)}")


if __name__ == "__main__":
    main()
