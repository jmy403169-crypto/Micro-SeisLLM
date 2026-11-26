# Denoising evaluation outputs

`scripts/evaluate_denoising.py` and `scripts/train_denoising.py` write metrics / predictions under this directory when `--output-dir` (or the YAML `output_dir`) points here.

Typical files after a full run: `metrics.json`, optional waveform dumps.
