#!/usr/bin/env python3
"""Load the configured local ESM-C checkpoint and score toy mutations."""

import json
import os

from evomax_enhanced.plm.esmc import ESMCMaskedScorer


def main() -> None:
    model_path = os.environ.get(
        "ESMC_MODEL", "/data/run01/scwb286/esm2_deploy/weights/ESMC-6B"
    )
    scorer = ESMCMaskedScorer(model_path, device="cuda", dtype="bf16", batch_size=2)
    sequence = "A" * 16
    mutations = [(3, "A", "E"), (8, "A", "L")]
    scores = scorer.score(mutations, sequence)
    print(json.dumps({"model": model_path, "scores": scores.tolist()}, ensure_ascii=False))


if __name__ == "__main__":
    main()
