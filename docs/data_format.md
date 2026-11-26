# Data format

## Waveform convention

- Components: E, N, Z on the last axis (size 3)
- Default sampling rate: **500 Hz**
- Bundled synthetic window: **4000** samples (8 s)
- Preferred dtype: `float32`

## Denoising NPZ

Keys: `noisy`, `clean`, `event_id`, `sampling_rate_hz`  
Shape: `(N_sta, T, 3)`

## Phase-picking NPZ

Keys: `waveform`, `p_arrival_sample`, `s_arrival_sample`, `event_id`, `sampling_rate_hz`  
Arrivals are **sample indices**. Convert with `t_sec = sample / sampling_rate_hz`.

## Source localization

- `events/{event_id}.npz`: `waveform` `(N_sta, T, 3)`
- `stations.csv`: station code and coordinates
- `catalogue.csv`: event_id and true source fields (synthetic local metres)
- `velocity_model.json`: illustrative Vp/Vs

## Splits

JSON with `train_event_ids`, `validation_event_ids`, `test_event_ids`, and `random_seed`.
