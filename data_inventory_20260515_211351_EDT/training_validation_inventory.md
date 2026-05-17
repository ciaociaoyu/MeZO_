# Training Validation Inventory

### 300-Step Window Validation

- Source: `experiments/int8_update_sparse_plan/probe_window_h100_20260512/window_training_summary.csv`.
- Rows: 11.
- Update backends: direct_int8;fp16_master.
- Precision/quantization modes: bf16;fp32;int8.
- Direction/family types: dense;sparse.
- h/h_active values: 1e-05;0.0003;0.0003286;0.001;0.0012;0.003;0.006;0.01;0.012.
- Seeds: none detected.
- Steps range: 301 to 301.
- NaN rows: 0.
- Final accuracy ambiguous: True.

| run | backend | precision | direction | h_raw | h_active | lr | steps | seed | best_acc | last_acc | final_acc | nan |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| dense_bf16_h1e-2 | direct_int8 | bf16 | dense | 0.01 | 0.01 | 1e-05 | 301 |  | 0.279859 |  | 0.279859 | False |
| dense_bf16_h1e-3 | direct_int8 | bf16 | dense | 0.001 | 0.001 | 1e-05 | 301 |  | 0.327869 |  | 0.327869 | False |
| dense_bf16_h1e-5 | direct_int8 | bf16 | dense | 1e-05 | 1e-05 | 1e-05 | 301 |  | 0.290398 |  | 0.290398 | False |
| dense_fp32_h1e-2 | direct_int8 | fp32 | dense | 0.01 | 0.01 | 1e-05 | 301 |  | 0.279859 |  | 0.279859 | False |
| dense_fp32_h1e-3 | direct_int8 | fp32 | dense | 0.001 | 0.001 | 1e-05 | 301 |  | 0.327869 |  | 0.327869 | False |
| dense_fp32_h1e-5 | direct_int8 | fp32 | dense | 1e-05 | 1e-05 | 1e-05 | 301 |  | 0.336066 |  | 0.336066 | False |
| dense_int8_fp16master_h1e-2 | fp16_master | int8 | dense | 0.01 | 0.01 | 1e-05 | 301 |  | 0.282201 |  | 0.282201 | False |
| dense_int8_fp16master_h3e-3 | fp16_master | int8 | dense | 0.003 | 0.003 | 1e-05 | 301 |  | 0.338407 |  | 0.338407 | False |
| dense_int8_fp16master_h3e-4 | fp16_master | int8 | dense | 0.0003 | 0.0003 | 1e-05 | 301 |  | 0.350117 |  | 0.350117 | False |
| sparse_int8_fp16master_p0p003_h0p0003286 | fp16_master | int8 | sparse | 0.0003286 | 0.006 | 1e-05 | 301 |  | 0.324356 |  | 0.324356 | False |
| sparse_int8_fp16master_p0p01_h0p0012 | fp16_master | int8 | sparse | 0.0012 | 0.012 | 1e-05 | 301 |  | 0.326698 |  | 0.326698 | False |

### Future Dense INT8 FP16-Master 2k Package

- Source: `experiments/future_probe_window_training_20260512_235627_package/summary_all.csv`.
- Rows: 6.
- Update backends: fp16_master.
- Precision/quantization modes: none detected.
- Direction/family types: dense.
- h/h_active values: 0.0003;0.001;0.002;0.003;0.005;0.01.
- Seeds: 0.
- Steps range: 2001 to 2001.
- NaN rows: 0.
- Final accuracy ambiguous: True.

| run | backend | precision | direction | h_raw | h_active | lr | steps | seed | best_acc | last_acc | final_acc | nan |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| dense_int8_fp16master_h1em02_seed0 | fp16_master |  | dense | 0.01 | 0.01 | 1e-05 | 2001 | 0 | 0.271663 |  | 0.271663 | False |
| dense_int8_fp16master_h1em03_seed0 | fp16_master |  | dense | 0.001 | 0.001 | 1e-05 | 2001 | 0 | 0.459016 |  | 0.459016 | False |
| dense_int8_fp16master_h2em03_seed0 | fp16_master |  | dense | 0.002 | 0.002 | 1e-05 | 2001 | 0 | 0.447307 |  | 0.447307 | False |
| dense_int8_fp16master_h3em03_seed0 | fp16_master |  | dense | 0.003 | 0.003 | 1e-05 | 2001 | 0 | 0.434426 |  | 0.434426 | False |
| dense_int8_fp16master_h3em04_seed0 | fp16_master |  | dense | 0.0003 | 0.0003 | 1e-05 | 2001 | 0 | 0.285714 |  | 0.285714 | False |
| dense_int8_fp16master_h5em03_seed0 | fp16_master |  | dense | 0.005 | 0.005 | 1e-05 | 2001 | 0 | 0.343091 |  | 0.343091 | False |

### May 15 Dense A100 2k Runs

- Source: `experiments/int8_update_sparse_plan/next_window_sparse_residual_20260515/cluster_dense_a100/summary_dense.csv`.
- Rows: 8.
- Update backends: fp16_master.
- Precision/quantization modes: int8.
- Direction/family types: dense.
- h/h_active values: 0.001;0.002;0.003.
- Seeds: 0;1;2.
- Steps range: 2001 to 2001.
- NaN rows: 0.
- Final accuracy ambiguous: False.

| run | backend | precision | direction | h_raw | h_active | lr | steps | seed | best_acc | last_acc | final_acc | nan |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| dense_int8_fp16master_h1e-3_seed0 | fp16_master | int8 | dense | 0.001 | 0.001 | 1e-05 | 2001 | 0 | 0.447307 | 0.422717 | 0.422717 | False |
| dense_int8_fp16master_h1e-3_seed1 | fp16_master | int8 | dense | 0.001 | 0.001 | 1e-05 | 2001 | 1 | 0.322014 | 0.322014 | 0.322014 | False |
| dense_int8_fp16master_h1e-3_seed2 | fp16_master | int8 | dense | 0.001 | 0.001 | 1e-05 | 2001 | 2 | 0.347775 | 0.336066 | 0.336066 | False |
| dense_int8_fp16master_h2e-3_seed0 | fp16_master | int8 | dense | 0.002 | 0.002 | 1e-05 | 2001 | 0 | 0.464871 | 0.460187 | 0.460187 | False |
| dense_int8_fp16master_h2e-3_seed1 | fp16_master | int8 | dense | 0.002 | 0.002 | 1e-05 | 2001 | 1 | 0.344262 | 0.344262 | 0.344262 | False |
| dense_int8_fp16master_h2e-3_seed2 | fp16_master | int8 | dense | 0.002 | 0.002 | 1e-05 | 2001 | 2 | 0.373536 | 0.373536 | 0.373536 | False |
| dense_int8_fp16master_h3e-3_seed0 | fp16_master | int8 | dense | 0.003 | 0.003 | 1e-05 | 2001 | 0 | 0.453162 | 0.453162 | 0.453162 | False |
| dense_int8_fp16master_h3e-3_seed1 | fp16_master | int8 | dense | 0.003 | 0.003 | 1e-05 | 2001 | 1 | 0.312646 | 0.312646 | 0.312646 | False |

### Dense A100 By h

| h_active | n | mean_best_eval_acc | mean_last_eval_acc | mean_last_eval_loss |
| --- | --- | --- | --- | --- |
| 0.001 | 3 | 0.372365 | 0.360265 | 1.4945 |
| 0.002 | 3 | 0.394223 | 0.392662 | 1.46038 |
| 0.003 | 2 | 0.382904 | 0.382904 | 1.45118 |

### Incomplete Or Partial Runs

- 300-step validation is short validation and is not enough for final accuracy claims.
- Future sparse training is incomplete locally: `summary_sparse.csv` has no data rows, though a `sparse_partial/` update_stats file exists.
- May 15 dense A100 h=3e-3 has two seeds in `summary_dense_by_h.csv`, while h=1e-3 and h=2e-3 have three seeds.
- Older summaries with `final_acc` but no separate best/last columns are flagged as ambiguous in `summary_inventory.csv`.

### Suitable For Paper Main Text

- The May 12 dense/sparse probe tables are the strongest main-text evidence for perturbation visibility.
- The May 15 dense A100 2k table is suitable as preliminary training validation because it separates best and last accuracy.
- 300-step validation should be cited only as short validation or sanity checking.
