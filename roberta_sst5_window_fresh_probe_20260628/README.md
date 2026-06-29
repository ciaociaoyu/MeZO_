# RoBERTa/SST-5 Fresh FP16/INT4 Probe

This is a fresh probe-only run. It recomputes `d_star`, `d_h`, `e_h`,
`scalar_nmse`, `rho_raw`, correlation, sign agreement, `dh_std`, and
`dh_zero_fraction`.

It does not reuse `/scratch/jy03364/MeZO_/roberta_sst5_theoretical_windows_20260627/raw_probe_metrics.csv`.

Requested directions: 128

FP16 h grid: 3e-06, 5e-06, 7e-06, 1e-05, 2e-05, 3e-05, 5e-05, 7e-05, 0.0001, 0.0003, 0.001, 0.0015, 0.002, 0.003, 0.005

INT4 h grid: 1e-05, 3e-05, 0.0001, 0.0003, 0.0005, 0.0007, 0.001, 0.0012, 0.0015, 0.002, 0.003, 0.005

## Probe Setup

- model/task: `roberta-large` / SST-5 full data
- seed/data_seed: 16/16
- batch size: 64; num_batches: 1
- direction seed base: 730000
- trainable subspace: all floating model parameters, matching the RTNClip dense runner
- low-bit forward oracle: G128 RTNClip shared-grid fake quantization on Linear.weight; non-Linear parameters remain unquantized in the forward state
- rho denominator: sampled vector random-direction floor `V_dir_sample`; scalar nMSE is reported separately and is not rho.

## Checkpoints

The underlying probe loader regenerates deterministic task-start states. Checkpoint files are local reproducibility artifacts and are ignored by `.gitignore`.

- fp16: `/scratch/jy03364/MeZO_/roberta_sst5_window_fresh_probe_20260628/checkpoints/task_start_seed16_deterministic_fp16_master.pt`
- int4: `/scratch/jy03364/MeZO_/roberta_sst5_window_fresh_probe_20260628/checkpoints/task_start_seed16_deterministic_int4_master.pt`

## Result Summary

| precision | first nonzero d_h h | practical primary | practical relaxed | smooth fit status | smooth W1 | interpretation |
|---|---:|---|---|---|---|---|
| fp16 | 5e-06 | [0.0003, 0.003] | [7e-05, 0.003] | window | [1.27691e-05, 0.0117266] | practical default-safe; smooth fit optional |
| int4 | 1e-05 | none | none | window | [0.00363621, 0.00478003] | empirical default-safe, no practical probe certificate |

## File Notes

- `raw_probe_metrics.csv` is freshly computed in this folder.
- `probe_summary_by_h.csv` adds `dh_zero_fraction`.
- `practical_probe_windows.csv` is threshold based and does not use accuracy.
- `rho_fit_windows.csv` is the dead-zone-filtered smooth rho fit.
- `rho_fit_windows_unfiltered_base_script.csv`, if present, is only the base script's unfiltered fit provenance and should not be used as the final fit table.
