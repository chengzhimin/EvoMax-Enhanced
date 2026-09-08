import numpy as np
import pytest

from evomax_enhanced.core.pipeline import EvoMaxConfig, EvoMaxPipeline, robust_median_iqr


class Provider:
    def __init__(self, values):
        self.values = np.asarray(values, dtype=float)
        self.calls = []

    def score(self, mutations, sequence):
        self.calls.append(list(mutations))
        return self.values[: len(mutations)]


def test_stage1_scores_all_then_sends_only_top_fraction_to_stage2():
    n = 20 * 19
    gpr = Provider(np.arange(n, dtype=float))
    esm2 = Provider(np.zeros(n))
    esm_if = Provider(np.arange(n, dtype=float))
    result = EvoMaxPipeline("A" * 20, gpr, esm2, esm_if).run()

    assert result["all_mutations"] == n
    assert result["shortlist"] == 5  # floor(380 * 0.015)
    assert len(gpr.calls[0]) == n
    assert len(esm2.calls[0]) == n
    assert len(esm_if.calls[0]) == 5


def test_robust_normalization_is_centered_and_scale_safe():
    result = robust_median_iqr([1, 2, 3, 4, 5])
    assert result[2] == pytest.approx(0.0)
    assert np.isfinite(result).all()


def test_cpu_cannot_silently_claim_bfloat16_gpu_mode():
    with pytest.raises(ValueError):
        EvoMaxConfig(device="cpu", dtype="bf16")
