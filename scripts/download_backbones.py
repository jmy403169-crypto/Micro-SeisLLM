#!/usr/bin/env python
"""Download instructions / helper for pretrained backbones (does not ship weights)."""
from __future__ import annotations

import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backbone", choices=["qwen3-4b", "gpt2", "all"], default="all")
    parser.add_argument("--out", default="checkpoints/backbones")
    args = parser.parse_args()
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)

    print("Micro-SeisLLM does NOT redistribute full LLM weights.")
    print("Please download them yourself:\n")
    if args.backbone in ("qwen3-4b", "all"):
        print("Qwen3-4B (Hugging Face):")
        print("  huggingface-cli download Qwen/Qwen3-4B --local-dir", out / "Qwen3-4B")
        print("Or ModelScope:")
        print("  modelscope download --model Qwen/Qwen3-4B --local_dir", out / "Qwen3-4B")
    if args.backbone in ("gpt2", "all"):
        print("\nGPT-2:")
        print("  huggingface-cli download openai-community/gpt2 --local-dir", out / "GPT2")
    print("\nThen set model.llm_path in the YAML configs (or MICROSEISLLM_QWEN_PATH / MICROSEISLLM_GPT2_PATH).")
    print("Task-specific LoRA / fine-tuned adapters should be placed under checkpoints/ and documented in checkpoints/README.md.")


if __name__ == "__main__":
    main()
