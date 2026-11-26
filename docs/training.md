# Training

Hyperparameters live in YAML under `configs/`. Prefer editing those files over hard-coding values in scripts.

```bash
python scripts/train_denoising.py --config configs/denoising/qwen3.yaml
python scripts/train_phase_picking.py --config configs/phase_picking/qwen3.yaml
python scripts/train_source_localization.py --config configs/source_localization/qwen3.yaml
```

Useful flags:

- `--pretrain false` — random-init backbone (CI / wiring checks)
- `--pretrain true` — needs local Qwen3 or GPT-2 weights
- `--max-steps N` — short smoke train
- `--llm-path PATH` — override backbone directory

Early stopping, batch size, learning rate, LoRA rank/alpha, layer count, patch size, and SNR ranges are all in the YAML.
