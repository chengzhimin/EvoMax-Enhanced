"""Local ESM-C masked-mutant scoring adapter.

ESM-C is used as an optional sequence-context channel. Its score is a masked
logit difference and must be calibrated/ablated separately from ESM-2.
"""

from argparse import Namespace
from collections.abc import Sequence

import numpy as np

from ..core.pipeline import Mutation


class ESMCMaskedScorer:
    def __init__(self, model_path: str, device: str = "cuda", dtype: str = "bf16", batch_size: int = 8):
        import torch
        from transformers import AutoModelForMaskedLM, AutoTokenizer

        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but no CUDA device is available")
        self.torch = torch
        self.device = torch.device(device)
        self.dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[dtype]
        self.batch_size = batch_size
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        if self.tokenizer.mask_token_id is None:
            raise ValueError("ESM-C tokenizer has no mask token")
        self.mask_id = int(self.tokenizer.mask_token_id)
        model_dtype = self.dtype if self.device.type == "cuda" else torch.float32
        self.model = AutoModelForMaskedLM.from_pretrained(
            model_path, local_files_only=True, torch_dtype=model_dtype
        ).to(self.device).eval()

    def _token_id(self, residue: str) -> int:
        token_ids = self.tokenizer.encode(residue, add_special_tokens=False)
        if len(token_ids) != 1:
            raise ValueError(f"residue {residue!r} is not a single ESM-C token")
        return int(token_ids[0])

    def score(self, mutations: Sequence[Mutation], sequence: str) -> np.ndarray:
        values: list[float] = []
        mask = self.tokenizer.mask_token
        for start in range(0, len(mutations), self.batch_size):
            batch = list(mutations[start : start + self.batch_size])
            masked_sequences = [sequence[:pos] + mask + sequence[pos + 1 :] for pos, _, _ in batch]
            inputs = self.tokenizer(masked_sequences, return_tensors="pt", padding=True)
            inputs = {key: value.to(self.device) for key, value in inputs.items()}
            mask_positions = []
            for row in inputs["input_ids"]:
                positions = (row == self.mask_id).nonzero(as_tuple=False).flatten()
                if len(positions) != 1:
                    raise ValueError("expected exactly one mask token per ESM-C input")
                mask_positions.append(int(positions[0]))
            rows = self.torch.arange(len(batch), device=self.device)
            positions = self.torch.tensor(mask_positions, device=self.device)
            with self.torch.inference_mode():
                autocast = self.torch.autocast("cuda", dtype=self.dtype) if self.device.type == "cuda" else self._nullcontext()
                with autocast:
                    logits = self.model(**inputs).logits
                selected = logits[rows, positions]
                wt_idx = self.torch.tensor([self._token_id(wt) for _, wt, _ in batch], device=self.device)
                mut_idx = self.torch.tensor([self._token_id(mut) for _, _, mut in batch], device=self.device)
                scores = (selected.gather(1, mut_idx[:, None]) - selected.gather(1, wt_idx[:, None])).squeeze(1)
            values.extend(scores.float().cpu().numpy().tolist())
        return np.asarray(values, dtype=float)

    class _nullcontext:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False
