"""GPU-batched ESM-2 masked-mutant scoring.

The score is log p(mutant | masked context) - log p(wild type | masked context).
"""

from collections.abc import Sequence
from argparse import Namespace
import numpy as np

from ..core.pipeline import Mutation


class ESM2MaskedScorer:
    def __init__(self, model_path: str, device: str = "cuda", dtype: str = "bf16", batch_size: int = 64):
        import torch
        import esm

        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but no CUDA device is available")
        self.torch = torch
        self.device = torch.device(device)
        self.dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[dtype]
        self.batch_size = batch_size
        with torch.serialization.safe_globals([Namespace]):
            self.model, self.alphabet = esm.pretrained.load_model_and_alphabet_local(model_path)
        self.model = self.model.to(self.device).eval()
        self.batch_converter = self.alphabet.get_batch_converter()

    def score(self, mutations: Sequence[Mutation], sequence: str) -> np.ndarray:
        out: list[float] = []
        for start in range(0, len(mutations), self.batch_size):
            batch = list(mutations[start : start + self.batch_size])
            _, _, tokens = self.batch_converter([("wt", sequence)] * len(batch))
            tokens = tokens.to(self.device)
            positions = self.torch.tensor([p + 1 for p, _, _ in batch], device=self.device)
            rows = self.torch.arange(len(batch), device=self.device)
            tokens[rows, positions] = self.alphabet.mask_idx
            with self.torch.inference_mode():
                autocast = self.torch.autocast("cuda", dtype=self.dtype) if self.device.type == "cuda" else self._nullcontext()
                with autocast:
                    logits = self.model(tokens, repr_layers=[])["logits"]
                selected = logits[rows, positions]
                wt_idx = self.torch.tensor([self.alphabet.get_idx(w) for _, w, _ in batch], device=self.device)
                mut_idx = self.torch.tensor([self.alphabet.get_idx(m) for _, _, m in batch], device=self.device)
                values = (selected.gather(1, mut_idx[:, None]) - selected.gather(1, wt_idx[:, None])).squeeze(1)
            out.extend(values.float().cpu().numpy().tolist())
        return np.asarray(out, dtype=float)

    class _nullcontext:
        def __enter__(self):
            return self
        def __exit__(self, *args):
            return False
