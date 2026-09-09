"""Foundry ProteinMPNN bridge for structure-conditioned candidate filtering.

ProteinMPNN generates or checks sequences compatible with a supplied
structure. It is deliberately exposed as a candidate generator/filter rather
than an affinity predictor.
"""

from dataclasses import dataclass
from pathlib import Path
import subprocess


@dataclass(frozen=True)
class ProteinMPNNConfig:
    python: str
    inference_script: str
    checkpoint: str
    output_dir: str
    structure: str
    designed_chains: str = "A"
    batch_size: int = 8
    number_of_batches: int = 2
    temperature: float = 0.2
    write_structures: bool = True


class ProteinMPNNRunner:
    def __init__(self, config: ProteinMPNNConfig):
        self.config = config

    def command(self) -> list[str]:
        cfg = self.config
        command = [
            cfg.python,
            cfg.inference_script,
            "--model_type", "protein_mpnn",
            "--is_legacy_weights", "True",
            "--checkpoint_path", cfg.checkpoint,
            "--out_directory", cfg.output_dir,
            "--structure_path", cfg.structure,
            "--number_of_batches", str(cfg.number_of_batches),
            "--batch_size", str(cfg.batch_size),
            "--write_fasta", "True",
            "--temperature", str(cfg.temperature),
            "--designed_chains", cfg.designed_chains,
        ]
        if cfg.write_structures:
            command.extend(["--write_structures", "True"])
        return command

    def run(self) -> Path:
        cfg = self.config
        for path in (cfg.inference_script, cfg.checkpoint, cfg.structure):
            if not Path(path).is_file():
                raise FileNotFoundError(path)
        Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
        subprocess.run(self.command(), check=True)
        outputs = sorted(Path(cfg.output_dir).glob("*.fa"))
        if not outputs:
            raise RuntimeError("ProteinMPNN completed without a non-empty FASTA output")
        return Path(cfg.output_dir)
