#!/usr/bin/env python3
"""Run Foundry ProteinMPNN on a private structure with constrained sites."""

import argparse
import csv
from pathlib import Path

from evomax_enhanced.inverse_folding import ProteinMPNNConfig, ProteinMPNNRunner


def read_fasta(path: Path) -> str:
    return "".join(line.strip() for line in path.read_text().splitlines() if not line.startswith(">"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wt-fasta", type=Path, required=True)
    parser.add_argument("--library-csv", type=Path, required=True)
    parser.add_argument("--structure", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mpnn-python", required=True)
    parser.add_argument("--mpnn-script", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--sequence-column", default="full_scFv_aa")
    parser.add_argument("--light-chain-length", type=int)
    parser.add_argument("--linker-length", type=int, default=0)
    parser.add_argument("--light-chain-id", default="L")
    parser.add_argument("--heavy-chain-id", default="H")
    parser.add_argument("--designed-residues", nargs="*", default=None)
    args = parser.parse_args()

    designed = args.designed_residues
    if designed is None:
        if args.light_chain_length is None:
            raise ValueError("provide --designed-residues or --light-chain-length")
        wt = read_fasta(args.wt_fasta)
        with args.library_csv.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        positions = {
            pos for row in rows for pos, (wt_aa, mut_aa) in enumerate(zip(wt, row[args.sequence_column])) if wt_aa != mut_aa
        }
        designed = []
        for pos in sorted(positions):
            if pos < args.light_chain_length:
                designed.append(f"{args.light_chain_id}{pos + 1}")
            elif pos >= args.light_chain_length + args.linker_length:
                designed.append(f"{args.heavy_chain_id}{pos - args.light_chain_length - args.linker_length + 1}")
        if not designed:
            raise ValueError("no designed residues were found")

    config = ProteinMPNNConfig(
        python=args.mpnn_python,
        inference_script=str(args.mpnn_script),
        checkpoint=str(args.checkpoint),
        output_dir=str(args.output_dir),
        structure=str(args.structure),
        designed_residues=tuple(designed),
    )
    output = ProteinMPNNRunner(config).run()
    print(f"ProteinMPNN output directory: {output}")


if __name__ == "__main__":
    main()
