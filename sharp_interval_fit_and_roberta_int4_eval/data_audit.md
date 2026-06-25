# Data audit

- fit input rows: 32172
- fit-ready rows: 1532
- fitted groups: 79
- RoBERTa INT4 training rows indexed: 108
- candidate configs: 0
- missing selected training rows: 0

## Available A_true / nMSE sources
experiment_type
real_interval_geometry            30501
synthetic                          1482
training_summary_or_loss_probe      189

## Fit target policy
- Fit target `A_fit` is restricted to paper-compatible loss-level directional MSE/NMSE rows.
- Geometry-only fields such as `A_cross`, `A_interval`, `sigma_raw2`, `delta_visibility_nmse`, and current `lowbit_true_nmse=dequantized_effective_displacement_nmse_v1` are not used as the fit target.
- They can only enter sharp/interval-aware models as covariates.

## Target kind counts
target_kind
geometry_only_no_loss_directional_mse                                                   30501
geometry_only_not_target:lowbit_true_nmse:dequantized_effective_displacement_nmse_v1       84
missing_directional_mse_target                                                             28
paper_directional_mse:fd_true_mse                                                          77
paper_directional_nmse:synthetic_A_true                                                  1482

## RoBERTa INT4 training rows by task/mode/run_type
task     mode          run_type
mnli     dense         full         1
                       pilot        3
         prefix        full         3
         sparse_p0p1   full         4
                       pilot        2
rte      dense         full         1
                       pilot        3
         prefix        full         2
                       medium       1
         sparse_p0p1   full         4
                       pilot        2
sst-2    dense         full         3
                       medium       1
                       pilot        2
         prefix        full         3
         sparse_p0p1   full         8
                       pilot        2
sst-5    dense         full        11
                       pilot        4
         prefix        full         3
         sparse_p0p1   full         4
                       pilot        3
trec     dense         full         2
                       pilot        3
         prefix        full         3
         sparse_p0p1   full         4
                       pilot        2
unknown  dense         full         1
                       missing      6
                       pilot        1
         sparse_p0p01  missing      1
         sparse_p0p1   medium       2
                       missing     11
                       pilot        2
