"""Multi-channel EvoMax orchestration for antibodies and enzymes.

The class keeps candidate generation separate from scoring so enzyme projects
can provide an assay-justified mutable region or ProteinMPNN-generated
candidates without changing the scoring contract.
"""

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from .pipeline import Mutation, ScoreProvider, enumerate_single_mutants, robust_median_iqr


@dataclass(frozen=True)
class EnhancedEvoMaxConfig:
    stage1_fraction: float = 0.015
    final_count: int = 20
    # Original EvoMax-compatible weights when GPR is available.
    stage1_gpr_weight: float = 0.35
    stage1_esm2_weight: float = 0.65
    # Enhanced second-stage channels. Active weights are renormalized when
    # GPR is unavailable rather than silently imputing an experimental score.
    stage2_weights: dict[str, float] = field(
        default_factory=lambda: {"gpr": 0.05, "esm2": 0.45, "esmc": 0.25, "esm_if": 0.25}
    )

    def __post_init__(self) -> None:
        if not 0 < self.stage1_fraction <= 1:
            raise ValueError("stage1_fraction must be in (0, 1]")
        if self.final_count < 1:
            raise ValueError("final_count must be positive")
        if self.stage1_gpr_weight < 0 or self.stage1_esm2_weight < 0:
            raise ValueError("stage1 weights must be non-negative")
        if not self.stage2_weights or any(value < 0 for value in self.stage2_weights.values()):
            raise ValueError("stage2 weights must be non-negative and non-empty")


class EnhancedEvoMaxPipeline:
    """Run the ESM-2 → ESM-C/ESM-IF1 → optional GPR flow.

    ``candidates`` may be supplied by a private mutation library or a
    ProteinMPNN post-processor. If omitted, all single amino-acid substitutions
    are enumerated, which is suitable for a small assay-justified region.
    """

    def __init__(
        self,
        sequence: str,
        esm2: ScoreProvider,
        esm_if: ScoreProvider,
        esmc: ScoreProvider,
        gpr: ScoreProvider | None = None,
        candidates: Sequence[Mutation] | None = None,
        config: EnhancedEvoMaxConfig | None = None,
    ) -> None:
        self.sequence = sequence
        self.esm2 = esm2
        self.esm_if = esm_if
        self.esmc = esmc
        self.gpr = gpr
        self.candidates = list(candidates) if candidates is not None else None
        self.config = config or EnhancedEvoMaxConfig()

    @staticmethod
    def _active_weights(weights: dict[str, float], channels: dict[str, np.ndarray]) -> dict[str, float]:
        active = {name: weight for name, weight in weights.items() if name in channels and weight > 0}
        total = sum(active.values())
        if not active or total <= 0:
            raise ValueError("no active scoring channels")
        return {name: weight / total for name, weight in active.items()}

    def run(self) -> dict[str, object]:
        mutations = self.candidates if self.candidates is not None else enumerate_single_mutants(self.sequence)
        if not mutations:
            raise ValueError("candidate set is empty")
        esm2 = np.asarray(self.esm2.score(mutations, self.sequence), dtype=float)
        if len(esm2) != len(mutations):
            raise ValueError("ESM-2 returned an unexpected number of scores")

        stage1_channels = {"esm2": esm2}
        if self.gpr is not None:
            gpr = np.asarray(self.gpr.score(mutations, self.sequence), dtype=float)
            if len(gpr) != len(mutations):
                raise ValueError("GPR returned an unexpected number of scores")
            stage1_channels["gpr"] = gpr
            stage1 = self.config.stage1_gpr_weight * robust_median_iqr(gpr)
            stage1 += self.config.stage1_esm2_weight * robust_median_iqr(esm2)
        else:
            gpr = None
            stage1 = robust_median_iqr(esm2)

        order = np.argsort(-stage1, kind="stable")
        shortlist_size = max(1, int(np.floor(len(mutations) * self.config.stage1_fraction)))
        shortlist_idx = order[:shortlist_size]
        shortlist = [mutations[i] for i in shortlist_idx]

        stage2_channels = {
            "esm2": esm2[shortlist_idx],
            "esmc": np.asarray(self.esmc.score(shortlist, self.sequence), dtype=float),
            "esm_if": np.asarray(self.esm_if.score(shortlist, self.sequence), dtype=float),
        }
        if gpr is not None:
            stage2_channels["gpr"] = gpr[shortlist_idx]
        for name, values in stage2_channels.items():
            if len(values) != len(shortlist):
                raise ValueError(f"{name} returned an unexpected number of scores")
        weights = self._active_weights(self.config.stage2_weights, stage2_channels)
        final_score = sum(weight * robust_median_iqr(stage2_channels[name]) for name, weight in weights.items())
        final_order = np.argsort(-final_score, kind="stable")[: self.config.final_count]
        ranked = [
            {
                "mutation": shortlist[i],
                "stage1_score": float(stage1[shortlist_idx[i]]),
                "final_score": float(final_score[i]),
            }
            for i in final_order
        ]
        return {
            "all_mutations": len(mutations),
            "shortlist": shortlist_size,
            "active_stage2_weights": weights,
            "ranked": ranked,
        }
