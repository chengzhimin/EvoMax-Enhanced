#!/usr/bin/env python3
"""Merge private ESM-2, ESM-C, ESM-IF1 and ProteinMPNN evidence.

ProteinMPNN contributes a structure-conditioned design-support channel. This
is intentionally reported separately from affinity/function and is not a
replacement for experimental labels or GPR calibration.
"""

import argparse
import csv
import math
from pathlib import Path

import numpy as np

from evomax_enhanced.core.pipeline import robust_median_iqr


def read_fasta(path: Path) -> str:
    return "".join(line.strip() for line in path.read_text().splitlines() if not line.startswith(">"))


def differences(wildtype: str, variant: str) -> list[tuple[int, str, str]]:
    if len(wildtype) != len(variant):
        raise ValueError("wildtype and variant lengths differ")
    return [(i, a, b) for i, (a, b) in enumerate(zip(wildtype, variant)) if a != b]


def records(path: Path) -> list[str]:
    output: list[str] = []
    current: list[str] = []
    for line in path.read_text().splitlines():
        if line.startswith(">"):
            if current:
                output.append("".join(current))
            current = []
        else:
            current.append(line.strip())
    if current:
        output.append("".join(current))
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--library-csv", type=Path, required=True)
    parser.add_argument("--base-csv", type=Path, required=True)
    parser.add_argument("--esmc-csv", type=Path, required=True)
    parser.add_argument("--wt-fasta", type=Path, required=True)
    parser.add_argument("--mpnn-fasta", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--light-chain-length", type=int, required=True)
    parser.add_argument("--linker-length", type=int, default=0)
    parser.add_argument("--heavy-chain-length", type=int, required=True)
    parser.add_argument("--target-length", type=int, required=True)
    parser.add_argument("--mpnn-weight", type=float, default=0.10)
    args = parser.parse_args()

    wt = read_fasta(args.wt_fasta)
    with args.library_csv.open(newline="") as handle:
        library = list(csv.DictReader(handle))
    with args.base_csv.open(newline="") as handle:
        base = {row["variant_id"]: row for row in csv.DictReader(handle)}
    with args.esmc_csv.open(newline="") as handle:
        esmc = {row["variant_id"]: row for row in csv.DictReader(handle)}
    mpnn = records(args.mpnn_fasta)
    expected_length = args.heavy_chain_length + args.light_chain_length + args.target_length
    if not mpnn or any(len(sequence) != expected_length for sequence in mpnn):
        raise ValueError("ProteinMPNN FASTA does not match the expected H/L/antigen chain layout")

    rows = []
    for row in library:
        identifier = row["variant_id"]
        if identifier not in base or identifier not in esmc:
            raise KeyError(identifier)
        mutations = differences(wt, row["full_scFv_aa"])
        support = 0.0
        for position, _, mutant in mutations:
            if position < args.light_chain_length:
                mpnn_position = args.heavy_chain_length + position
            elif position >= args.light_chain_length + args.linker_length:
                mpnn_position = position - args.light_chain_length - args.linker_length
            else:
                raise ValueError("candidate mutation falls inside the linker")
            count = sum(sequence[mpnn_position] == mutant for sequence in mpnn)
            frequency = (count + 1) / (len(mpnn) + 20)
            support += math.log(frequency / 0.05)
        rows.append({
            "variant_id": identifier,
            "n_mutations": len(mutations),
            "mutations": row.get("mutations", ""),
            "esm2": float(base[identifier]["esm2_additive_score"]),
            "esmc": float(esmc[identifier]["esmc_additive_score"]),
            "esm_if1": float(base[identifier]["esm_if1_additive_score"]),
            "mpnn_support": support,
        })

    channels = {name: np.asarray([row[name] for row in rows], dtype=float) for name in ("esm2", "esmc", "esm_if1", "mpnn_support")}
    weights = {"esm2": 0.40, "esmc": 0.25, "esm_if1": 0.25, "mpnn_support": args.mpnn_weight}
    total = sum(weights.values())
    weights = {name: value / total for name, value in weights.items()}
    combined = sum(weights[name] * robust_median_iqr(values) for name, values in channels.items())

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["variant_id", "mutations", "n_mutations", "esm2_additive_score", "esmc_additive_score", "esm_if1_additive_score", "mpnn_support_score", "enhanced_model_score"])
        writer.writeheader()
        for index in np.argsort(-combined, kind="stable"):
            row = rows[index]
            writer.writerow({
                "variant_id": row["variant_id"], "mutations": row["mutations"], "n_mutations": row["n_mutations"],
                "esm2_additive_score": f"{row['esm2']:.8f}", "esmc_additive_score": f"{row['esmc']:.8f}",
                "esm_if1_additive_score": f"{row['esm_if1']:.8f}", "mpnn_support_score": f"{row['mpnn_support']:.8f}",
                "enhanced_model_score": f"{combined[index]:.8f}",
            })


if __name__ == "__main__":
    main()
