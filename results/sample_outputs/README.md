# Sample outputs

After a successful quickstart / inference pass you should see files such as:

| File | Meaning |
|------|---------|
| `denoised.npz` | Denoised waveform |
| `picks.json` | P/S arrival picks |
| `location.json` | Estimated source location |

Evaluate scripts may also write `*_metrics.json` here during smoke runs. Paper-table metrics belong under `results/denoising/`, `results/phase_picking/`, `results/source_localization/` and are summarized in `../expected_metrics.json`.
