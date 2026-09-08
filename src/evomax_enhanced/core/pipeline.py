"""文献等价 EvoMax 两阶段流程；模型推理由可插拔 provider 提供。"""

from dataclasses import dataclass
from typing import Protocol, Sequence
import numpy as np

Mutation = tuple[int, str, str]


class ScoreProvider(Protocol):
    def score(self, mutations: Sequence[Mutation], sequence: str) -> np.ndarray: ...


@dataclass(frozen=True)
class EvoMaxConfig:
    stage1_fraction: float = 0.015
    final_count: int = 20
    device: str = "cuda"
    dtype: str = "bf16"

    def __post_init__(self) -> None:
        if not 0 < self.stage1_fraction <= 1:
            raise ValueError("stage1_fraction must be in (0, 1]")
        if self.final_count < 1:
            raise ValueError("final_count must be positive")
        if self.device == "cpu" and self.dtype == "bf16":
            raise ValueError("bf16 is reserved for CUDA in this pipeline")


def enumerate_single_mutants(sequence: str) -> list[Mutation]:
    amino_acids = "ACDEFGHIKLMNPQRSTVWY"
    return [
        (pos, wt, mutant)
        for pos, wt in enumerate(sequence)
        for mutant in amino_acids
        if mutant != wt
    ]


def robust_median_iqr(values: Sequence[float]) -> np.ndarray:
    x = np.asarray(values, dtype=np.float64)
    median = np.nanmedian(x)
    q1, q3 = np.nanpercentile(x, [25, 75])
    scale = q3 - q1
    return (x - median) / (scale if scale > 0 else 1.0)


class EvoMaxPipeline:
    def __init__(
        self,
        sequence: str,
        gpr: ScoreProvider,
        esm2: ScoreProvider,
        esm_if: ScoreProvider,
        config: EvoMaxConfig | None = None,
    ) -> None:
        self.sequence = sequence
        self.gpr = gpr
        self.esm2 = esm2
        self.esm_if = esm_if
        self.config = config or EvoMaxConfig()

    def run(self) -> dict[str, object]:
        mutations = enumerate_single_mutants(self.sequence)
        gpr = np.asarray(self.gpr.score(mutations, self.sequence), dtype=float)
        esm2 = np.asarray(self.esm2.score(mutations, self.sequence), dtype=float)
        stage1_score = 0.35 * robust_median_iqr(gpr) + 0.65 * robust_median_iqr(esm2)
        order = np.argsort(-stage1_score, kind="stable")
        shortlist_size = max(1, int(np.floor(len(mutations) * self.config.stage1_fraction)))
        shortlist_idx = order[:shortlist_size]
        shortlist = [mutations[i] for i in shortlist_idx]

        esm_if = np.asarray(self.esm_if.score(shortlist, self.sequence), dtype=float)
        final_score = (
            0.05 * robust_median_iqr(gpr[shortlist_idx])
            + 0.70 * robust_median_iqr(esm2[shortlist_idx])
            + 0.25 * robust_median_iqr(esm_if)
        )
        final_order = np.argsort(-final_score, kind="stable")[: self.config.final_count]
        ranked = [
            {"position": shortlist[i][0], "wt": shortlist[i][1], "mut": shortlist[i][2],
             "stage1_score": float(stage1_score[shortlist_idx[i]]),
             "final_score": float(final_score[i])}
            for i in final_order
        ]
        return {"all_mutations": len(mutations), "shortlist": shortlist_size, "ranked": ranked}
