"""Small PyTorch fallback for the reductions used by fair-esm ESM-IF1.

The cluster environment lacks a compatible compiled torch-scatter wheel. This
fallback intentionally implements only the public reductions required by the
installed ESM-IF1 code; it is not a replacement for the full package.
"""

from __future__ import annotations

import torch


def scatter_add(
    src: torch.Tensor,
    index: torch.Tensor,
    dim: int = 0,
    out: torch.Tensor | None = None,
    dim_size: int | None = None,
) -> torch.Tensor:
    if dim != 0:
        raise NotImplementedError("fallback supports dim=0 only")
    if index.dtype != torch.long:
        index = index.long()
    size = dim_size if dim_size is not None else (int(index.max().item()) + 1 if index.numel() else 0)
    shape = (size,) + tuple(src.shape[1:])
    result = out if out is not None else torch.zeros(shape, dtype=src.dtype, device=src.device)
    return result.index_add_(0, index.to(src.device), src)


def scatter(
    src: torch.Tensor,
    index: torch.Tensor,
    dim: int = 0,
    out: torch.Tensor | None = None,
    dim_size: int | None = None,
    reduce: str = "sum",
) -> torch.Tensor:
    if reduce == "sum" or reduce == "add":
        return scatter_add(src, index, dim=dim, out=out, dim_size=dim_size)
    if reduce == "mean":
        total = scatter_add(src, index, dim=dim, out=out, dim_size=dim_size)
        count = scatter_add(torch.ones_like(src), index, dim=dim, dim_size=total.shape[0])
        return total / count.clamp_min(1)
    raise NotImplementedError(f"fallback reduction unsupported: {reduce}")


__all__ = ["scatter", "scatter_add"]
