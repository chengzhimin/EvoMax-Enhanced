#!/usr/bin/env python3
"""Score a private candidate library with the optional ESM-C channel.

The script intentionally emits only candidate IDs and model scores. Input
sequences and structures remain in the private project directory.
"""

import argparse
import csv
from pathlib import Path

from evomax_enhanced.plm.esmc import ESMCMaskedScorer


def read_fasta(path: Path) -> str:
    return "".join(line.strip() for line in path.read_text().splitlines() if not line.startswith(">"))


def differences(wildtype: str, variant: str) -> list[tuple[int, str, str]]:
    if len(wildtype) != len(variant):
        raise ValueError("wildtype and variant lengths differ")
    return [(i, wt, mut) for i, (wt, mut) in enumerate(zip(wildtype, variant)) if wt != mut]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wt-fasta", type=Path, required=True)
    parser.add_argument("--library-csv", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--model", required=True,
                        help="Local ESM-C checkpoint directory")
    parser.add_argument("--sequence-column", default="full_scFv_aa")
    parser.add_argument("--id-column", default="variant_id")
    args = parser.parse_args()

    wt = read_fasta(args.wt_fasta)
    with args.library_csv.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    parsed = [(row, differences(wt, row[args.sequence_column])) for row in rows]
    unique = sorted({mutation for _, mutations in parsed for mutation in mutations})
    scorer = ESMCMaskedScorer(args.model, device="cuda", dtype="bf16", batch_size=8)
    scores = scorer.score(unique, wt)
    single_scores = dict(zip(unique, scores.tolist()))
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[args.id_column, "n_mutations", "esmc_additive_score"])
        writer.writeheader()
        for row, mutations in parsed:
            writer.writerow({
                args.id_column: row[args.id_column],
                "n_mutations": len(mutations),
                "esmc_additive_score": f"{sum(single_scores[m] for m in mutations):.8f}",
            })


if __name__ == "__main__":
    main()
