"""Local-checkpoint ESM-IF1 scorer for EvoMax stage 2."""

from pathlib import Path
from typing import Sequence
import numpy as np

from ..core.pipeline import Mutation


class ESMIF1Scorer:
    def __init__(self, checkpoint: str, structure: str, chain: str = "A", device: str = "cuda"):
        import torch
        import esm
        from esm.inverse_folding import util

        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but no CUDA device is available")
        if not Path(checkpoint).is_file():
            raise FileNotFoundError(f"ESM-IF1 checkpoint not found: {checkpoint}")
        self.torch = torch
        self.device = torch.device(device)
        self.model, self.alphabet = esm.pretrained.load_model_and_alphabet_local(checkpoint)
        self.model = self.model.to(self.device).eval()
        self.coords, self.structure_sequence = util.load_coords(structure, chain)

    def score(self, mutations: Sequence[Mutation], sequence: str) -> np.ndarray:
        from esm.inverse_folding import util

        if sequence != self.structure_sequence:
            raise ValueError("structure sequence must match the WT sequence residue-for-residue")
        values = []
        with self.torch.inference_mode():
            for pos, wt, mutant in mutations:
                if sequence[pos] != wt:
                    raise ValueError(f"WT mismatch at zero-based position {pos}")
                mutated = sequence[:pos] + mutant + sequence[pos + 1 :]
                _, ll_with_coord = util.score_sequence(self.model, self.alphabet, self.coords, mutated)
                values.append(ll_with_coord)
        return np.asarray(values, dtype=float)
