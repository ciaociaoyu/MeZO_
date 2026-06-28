# V10 Medium Multi-Seed Recovery Summary

This recovery targets the 2026-06-27 V10 medium supplement queue.

## What Was Recovered

- Recovered 16 completed RoBERTa-large/SST-5 FP32 high-precision runs from trainer stderr logs.
- The emitted `eval_metrics.jsonl` files are empty, so `run_summary.json` is not reliable for these jobs.
- Each recovered run has 20 training-time dev evaluations; the duplicated final validation line was removed from curves.
- h values: 1e-05, 0.0001, 0.001, 0.003
- seeds: 32, 64, 128, 256

## High-Precision Aggregate

```
precision  task       h h_label  n_runs         seeds  best_eval_acc_mean  best_eval_acc_std  final_eval_acc_mean  final_eval_acc_std  best_eval_loss_mean  final_eval_loss_mean
     fp32 SST-5 0.00001    1e-5       4 32,64,128,256            0.455504           0.008710             0.455504            0.008710             1.366728              1.366728
     fp32 SST-5 0.00010    1e-4       4 32,64,128,256            0.454625           0.009861             0.454625            0.009861             1.367609              1.367609
     fp32 SST-5 0.00100    1e-3       4 32,64,128,256            0.456089           0.007971             0.456089            0.007971             1.375241              1.375241
     fp32 SST-5 0.00300    3e-3       4 32,64,128,256            0.429450           0.005845             0.429450            0.005845             1.444573              1.444573
```

## Variance Components

```
       metric                 analysis_set                       status  n_h  n_seeds  var_h_policy  var_seed_direction  var_residual_interaction  share_h_policy  share_seed_direction  share_residual_interaction                 h_values         seeds
best_eval_acc                        all_h descriptive_balanced_two_way    4        4  1.265955e-04            0.000027                  0.000024        0.713734              0.151347                    0.134920 1e-05,0.0001,0.001,0.003 32,64,128,256
best_eval_acc inner_plateau_1e-5_1e-4_1e-3 descriptive_balanced_two_way    3        4  3.618307e-07            0.000057                  0.000002        0.006079              0.964166                    0.029755       1e-05,0.0001,0.001 32,64,128,256
```

## Low-Bit Multi-Seed Status

The sparse/prefix INT4 seed32/seed64 jobs did not produce valid training results. They failed before training because the runner looked for missing seed-specific full-data directories such as `full-32` and `full-64`.

```
                status          failure_type                                                                     missing_path                                                           log_path
failed_before_training missing_dataset_split /scratch/jy03364/MeZO_/medium_models/data/k-shot-1k-test/sst-2/full-32/train.tsv v10_medium_supplement_results_20260627/jobs/v10-med_46453542_0.err
failed_before_training missing_dataset_split  /scratch/jy03364/MeZO_/medium_models/data/k-shot-1k-test/trec/full-32/train.csv v10_medium_supplement_results_20260627/jobs/v10-med_46453542_1.err
failed_before_training missing_dataset_split  /scratch/jy03364/MeZO_/medium_models/data/k-shot-1k-test/trec/full-32/train.csv v10_medium_supplement_results_20260627/jobs/v10-med_46453542_2.err
failed_before_training missing_dataset_split  /scratch/jy03364/MeZO_/medium_models/data/k-shot-1k-test/trec/full-32/train.csv v10_medium_supplement_results_20260627/jobs/v10-med_46453542_3.err
failed_before_training missing_dataset_split /scratch/jy03364/MeZO_/medium_models/data/k-shot-1k-test/sst-2/full-32/train.tsv v10_medium_supplement_results_20260627/jobs/v10-med_46453542_4.err
failed_before_training missing_dataset_split /scratch/jy03364/MeZO_/medium_models/data/k-shot-1k-test/sst-2/full-32/train.tsv v10_medium_supplement_results_20260627/jobs/v10-med_46453542_5.err
```

The earlier `v10_supplement_results_20260626` sparse/prefix aggregates are seed16-only despite their historical filenames; they should not be described as multi-seed confirmation.
