"""Real ESM-2 inference check on a synthetic sequence."""
import json
import numpy as np
from evomax_enhanced.plm.esm2 import ESM2MaskedScorer

scorer = ESM2MaskedScorer(
    "/data/run01/scwb286/.cache/torch/hub/checkpoints/esm2_t36_3B_UR50D.pt",
    batch_size=2,
)
scores = scorer.score([(0, "A", "V"), (1, "C", "S")], "ACDEFGHIKLMNPQRSTVWY")
assert scores.shape == (2,) and np.isfinite(scores).all()
print("ESM2_MODEL_SMOKE_OK", json.dumps(scores.tolist()), flush=True)
