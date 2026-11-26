# License scope

Apache-2.0 in `LICENSE` covers **this repository’s source code** (Python package under `src/`, scripts, configs, docs text, and tests).

It does **not** automatically cover:

| Asset | Status in this repo | License / terms |
|-------|---------------------|-----------------|
| Bundled synthetic sample NPZ/CSV under `data/sample/` | Redistributed for workflow demos | Research / academic use with this code; not a substitute for field data |
| Field microseismic waveforms used in the paper | **Not** redistributed | Restricted by data-usage agreements |
| Qwen3-4B base weights | Download separately | Follow the Qwen model card / license at the download source |
| GPT-2 base weights | Download separately | Follow the OpenAI / Hugging Face model card |
| Task checkpoints (truncated LLM + LoRA + heads) | Hosted outside git when released | Same as this code for our trained adapters/heads; base LLM slices still under their upstream licenses |

Do not assume that cloning this repo grants rights to redistribute Qwen or GPT-2 weights.
