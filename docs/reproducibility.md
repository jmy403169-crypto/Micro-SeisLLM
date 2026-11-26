# Reproducibility

## Checklist

1. Create the env from `environment.yml` or `requirements.txt`, then `pip install -e .`.
2. Download backbones with `scripts/download_backbones.py` (paper runs only).
3. Download task checkpoints listed in `checkpoints/README.md` once URLs are published.
4. Point YAML `data_dir` at the full synthetic (or your) test set — not only `data/sample/`.
5. Run the evaluate scripts below and compare to `results/expected_metrics.json` within the stated tolerances.

## Paper results ↔ commands

| Paper content | Command | Config | Output |
|---------------|---------|--------|--------|
| Noise suppression | `python scripts/train_denoising.py --config configs/denoising/qwen3.yaml` / `python scripts/evaluate_denoising.py --config configs/denoising/qwen3.yaml --checkpoint checkpoints/lora_adapters/denoising/best.pt` | `configs/denoising/qwen3.yaml` (GPT-2: `gpt2.yaml`) | `results/denoising/` |
| First-arrival picking | `python scripts/train_phase_picking.py --config configs/phase_picking/qwen3.yaml` / `python scripts/evaluate_phase_picking.py --config configs/phase_picking/qwen3.yaml --checkpoint checkpoints/lora_adapters/phase_picking/best.pt` | `configs/phase_picking/qwen3.yaml` | `results/phase_picking/` |
| Source localization | `python scripts/train_source_localization.py --config configs/source_localization/qwen3.yaml` / `python scripts/evaluate_source_localization.py --config configs/source_localization/qwen3.yaml --checkpoint checkpoints/lora_adapters/source_localization/best.pt` | `configs/source_localization/qwen3.yaml` | `results/source_localization/` |

Unified smoke / sweep helper: `python scripts/reproduce_results.py`.

## Architecture defaults

| Task | Factory | Input |
|------|---------|-------|
| Noise suppression | `SeisMoLLM_denoise_event` | `(B,N,3,T)` + cross-station attention |
| Phase picking | `SeisMoLLM_dpk` | `(B,3,T)` |
| Source localization | `SeisMoLLM_loc_event` | `(B,N,3,T)` + station mask; physical units via `loc_norm_stats.json` |

Saved experimental checkpoints are truncated LLM + LoRA + heads (see `checkpoints/README.md`).

## Seeds and splits

YAML `random_seed` plus `data/splits/*.json`. Full synthetic corpus meta: `data/splits/full_synthetic_split_meta.json`.

## Expected numeric ranges

See `results/expected_metrics.json` (`tolerance` block). Field-data rows are documented for the paper but cannot be re-run from this public tree.
