#!/usr/bin/env python3
"""Convert private Foundry ProteinMPNN complex FASTA to RF3 JSON inputs."""

import argparse
import json
from pathlib import Path


def fasta_records(path: Path) -> list[tuple[str, str]]:
    records: list[tuple[str, str]] = []
    header = None
    sequence: list[str] = []
    for line in path.read_text().splitlines():
        if line.startswith(">"):
            if header is not None:
                records.append((header, "".join(sequence)))
            header, sequence = line[1:], []
        else:
            sequence.append(line.strip())
    if header is not None:
        records.append((header, "".join(sequence)))
    return records


def pdb_chain_lengths(path: Path) -> list[tuple[str, int]]:
    chains: dict[str, set[tuple[str, str]]] = {}
    order: list[str] = []
    for line in path.read_text().splitlines():
        if not line.startswith("ATOM") or line[12:16].strip() != "CA":
            continue
        chain = line[21]
        if chain not in chains:
            chains[chain] = set()
            order.append(chain)
        chains[chain].add((line[22:26].strip(), line[26]))
    return [(chain, len(chains[chain])) for chain in order]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mpnn-fasta", type=Path, required=True)
    parser.add_argument("--structure", type=Path, required=True)
    parser.add_argument("--target-chain", required=True)
    parser.add_argument("--target-msa", type=Path)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()

    chains = pdb_chain_lengths(args.structure)
    if not chains or args.target_chain not in dict(chains):
        raise ValueError("target chain is absent from structure")
    total_length = sum(length for _, length in chains)
    entries = []
    for header, sequence in fasta_records(args.mpnn_fasta):
        if len(sequence) != total_length:
            raise ValueError("ProteinMPNN sequence length does not match structure chains")
        offset = 0
        components = []
        for chain, length in chains:
            chain_sequence = sequence[offset : offset + length]
            offset += length
            component = {"seq": chain_sequence, "chain_id": chain}
            if chain == args.target_chain and args.target_msa is not None:
                component["msa_path"] = str(args.target_msa)
            components.append(component)
        name = header.replace(",", "_").replace(" ", "_")[:120]
        entries.append({"name": name, "components": components})
    if not entries:
        raise ValueError("ProteinMPNN FASTA is empty")
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(entries, indent=2))
    print(f"RF3 inputs: {len(entries)}")


if __name__ == "__main__":
    main()
