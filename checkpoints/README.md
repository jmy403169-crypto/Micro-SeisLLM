# Checkpoints

Base LLMs and task weights are **not** committed. GitHub rejects multi-GB files, and redistributing Qwen/GPT-2 shards would also mix licenses.

## Base models

| Backbone | Suggested source | Default local path |
|----------|------------------|--------------------|
| Qwen3-4B | Hugging Face `Qwen/Qwen3-4B` or ModelScope | `checkpoints/backbones/Qwen3-4B` |
| GPT-2 | Hugging Face `openai-community/gpt2` | `checkpoints/backbones/GPT2` |

```bash
python scripts/download_backbones.py --backbone all
```

Environment overrides: `MICROSEISLLM_QWEN3_PATH`, `MICROSEISLLM_GPT2_PATH`.

## Task weights (paper runs)

Training used LoRA (`r=16`, `alpha=16`), but the saved `.pth` files are **truncated LLM + LoRA + task modules**, not PEFT-only adapters.

| Typical file | Approx. size | Approx. LoRA-only |
|--------------|--------------|-------------------|
| GPT-2 task (3 layers) | 110–420 MB | ~2–3 MB |
| Qwen3-4B task (3 layers) | 2.7–4.0 GB | ~20–25 MB |

After release, put files here:

```text
checkpoints/lora_adapters/
  denoising/best.pt
  phase_picking/best.pt
  source_localization/best.pt
  source_localization/loc_norm_stats.json
```

### Download table (fill when hosted)

Host on Zenodo / Hugging Face / institutional storage, then replace the TBD cells. Do not force-push multi-GB binaries into this git repo.

| File | Size | SHA256 | URL | Local path |
|------|------|--------|-----|------------|
| `denoising_qwen3.pt` | TBD — publish before journal camera-ready | TBD | TBD | `lora_adapters/denoising/best.pt` |
| `denoising_gpt2.pt` | TBD | TBD | TBD | `lora_adapters/denoising/best.pt` (gpt2 run) |
| `picking_qwen3.pt` | TBD | TBD | TBD | `lora_adapters/phase_picking/best.pt` |
| `picking_gpt2.pt` | TBD | TBD | TBD | `lora_adapters/phase_picking/best.pt` |
| `localization_qwen3.pt` | TBD | TBD | TBD | `lora_adapters/source_localization/best.pt` |
| `localization_gpt2.pt` | TBD | TBD | TBD | `lora_adapters/source_localization/best.pt` |
| `loc_norm_stats.json` | <1 MB | TBD | TBD | `lora_adapters/source_localization/loc_norm_stats.json` |

Verify after download:

```bash
# example once SHA256 is published
certutil -hashfile checkpoints/lora_adapters/denoising/best.pt SHA256   # Windows
# shasum -a 256 checkpoints/lora_adapters/denoising/best.pt            # Linux/macOS
```

## What the state dict looks like

- Embedder / head: `convs.*`, `out_head.*`, `station_attn.*`  
- LoRA: `...lora_A...`, `...lora_B...`  
- Frozen backbone slices: `...base_layer...`  
- Older runs may wrap weights under `model_dict`; loaders in this repo also accept `model`.

## Paper-run tags (local experiment logs; not shipped)

| Task | Backbone | Example tag |
|------|----------|-------------|
| Denoising | GPT-2 / Qwen3 | `*_denoise_event_*` |
| Picking | GPT-2 / Qwen3 | `*_dpk_*` |
| Localization | GPT-2 / Qwen3 | `*_loc_event_*` |
