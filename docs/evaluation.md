# Evaluation

```bash
python scripts/evaluate_denoising.py --config configs/denoising/qwen3.yaml --checkpoint checkpoints/lora_adapters/denoising/best.pt
python scripts/evaluate_phase_picking.py --config configs/phase_picking/qwen3.yaml --checkpoint checkpoints/lora_adapters/phase_picking/best.pt
python scripts/evaluate_source_localization.py --config configs/source_localization/qwen3.yaml --checkpoint checkpoints/lora_adapters/source_localization/best.pt
```

Without `--checkpoint`, scripts can still run a random-init pass on `data/sample/` to check the pipeline. That pass will **not** match `results/expected_metrics.json`.

Metrics:

- Denoising: ΔSNR (dB), correlation coefficient  
- Picking: MAE in samples and milliseconds, F1  
- Localization: loc4 MAE / RMSE in metres (synthetic local coordinates)

Default output directories for paper-style runs: `results/denoising/`, `results/phase_picking/`, `results/source_localization/`. Quick demo dumps: `results/sample_outputs/`.
