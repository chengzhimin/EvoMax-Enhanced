import numpy as np

from evomax_enhanced.core.pipeline import EvoMaxPipeline


class Provider:
    def __init__(self):
        self.calls = []

    def score(self, mutations, sequence):
        self.calls.append(len(mutations))
        return np.arange(len(mutations), dtype=float)


gpr, esm2, esm_if = Provider(), Provider(), Provider()
result = EvoMaxPipeline("A" * 20, gpr, esm2, esm_if).run()
assert result["all_mutations"] == 380
assert result["shortlist"] == 5
assert gpr.calls == [380] and esm2.calls == [380] and esm_if.calls == [5]
print("CORE_PIPELINE_OK", result["all_mutations"], result["shortlist"])
