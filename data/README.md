# Data

## Shipped here

Synthetic mini-sets under `sample/` for install checks and CI:

- `sample/denoising/` — noisy/clean NPZ `(N_sta, T, 3)`, 500 Hz  
- `sample/phase_picking/` — waveforms + P/S sample indices  
- `sample/source_localization/` — `events/*.npz`, `stations.csv`, `catalogue.csv`, `velocity_model.json`

## Not public

Field microseismic waveforms from the paper stay off this repository because of data-usage limits. The README states that explicitly.

## Split metadata

`splits/full_synthetic_split_meta.json` records the seed and sizes for the full synthetic corpus. The small JSON files `*_split.json` only cover the demo sample.
