#!/usr/bin/env python
"""Run representative evaluation commands for the three paper tasks."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(cmd):
    print(">>", " ".join(cmd))
    subprocess.check_call(cmd, cwd=str(ROOT))


def main():
    py = sys.executable
    pairs = [
        ("scripts/evaluate_denoising.py", "configs/denoising/qwen3.yaml"),
        ("scripts/evaluate_phase_picking.py", "configs/phase_picking/qwen3.yaml"),
        ("scripts/evaluate_source_localization.py", "configs/source_localization/qwen3.yaml"),
        ("scripts/evaluate_denoising.py", "configs/denoising/gpt2.yaml"),
        ("scripts/evaluate_phase_picking.py", "configs/phase_picking/gpt2.yaml"),
        ("scripts/evaluate_source_localization.py", "configs/source_localization/gpt2.yaml"),
    ]
    for script, cfg in pairs:
        run([py, script, "--config", cfg, "--pretrain", "false"])
    print("Reproduction sweep finished. See results/sample_outputs/*_metrics.json")
    print("Paper-table numbers require trained LoRA checkpoints (see checkpoints/README.md).")


if __name__ == "__main__":
    main()
