# Third-party notices

This project depends on external models and libraries. We do **not** vendor their full source trees in this repository.

## Base models (weights downloaded by the user)

| Component | Typical source | Notes |
|-----------|----------------|-------|
| Qwen3-4B | Hugging Face `Qwen/Qwen3-4B` or ModelScope | License and usage terms follow the model card of the checkpoint you download. Not redistributed here. |
| GPT-2 | Hugging Face `openai-community/gpt2` | License and usage terms follow the model card. Not redistributed here. |

`scripts/download_backbones.py` only fetches these checkpoints to a local path; it does not relicense them.

## Major Python dependencies

| Package | Role |
|---------|------|
| PyTorch | Training / inference backend |
| Hugging Face `transformers` | LLM backbone loading |
| Hugging Face `peft` | LoRA adapters |
| NumPy, SciPy, scikit-learn, pandas | Numerics and I/O |
| einops, matplotlib, h5py, safetensors, PyYAML, tqdm | Utilities |

Each package remains under its own upstream license (typically BSD / Apache / similar). Install via `pip` / `conda` and consult their project pages for the authoritative text.

## Source modifications

This repository implements its own embedders, task heads, datasets, and training loops. It does **not** ship a forked copy of the Qwen or GPT-2 source trees; the models are loaded as pretrained weights through `transformers`.
