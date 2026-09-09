import numpy as np

from evomax_enhanced.core.enhanced import EnhancedEvoMaxPipeline


class Provider:
    def __init__(self, values):
        self.values = np.asarray(values, dtype=float)
        self.calls = []

    def score(self, mutations, sequence):
        self.calls.append(list(mutations))
        return self.values[: len(mutations)]


def test_enhanced_pipeline_uses_esmc_and_if_only_on_shortlist_without_gpr():
    n = 20 * 19
    esm2 = Provider(np.arange(n, dtype=float))
    esmc = Provider(np.ones(n))
    esm_if = Provider(np.arange(n, dtype=float))
    result = EnhancedEvoMaxPipeline("A" * 20, esm2, esm_if, esmc).run()

    assert result["all_mutations"] == n
    assert result["shortlist"] == 5
    assert len(esm2.calls[0]) == n
    assert len(esmc.calls[0]) == 5
    assert len(esm_if.calls[0]) == 5
    assert "gpr" not in result["active_stage2_weights"]
