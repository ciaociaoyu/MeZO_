# V10 Medium Supplement Config Audit

Generated: 2026-06-27T17:02:33
Git commit: `5703318c54ce0aaf0c75e58e6d13ee7a53c7882b`

## Decisions

- High precision plateau uses FP32 RoBERTa-large/SST-5 from the existing `latest_main` family.
- Sparse p=0.1 INT4 uses the seed-fixed task-gradient mask family: `outputs/int4_sparse_prefix_seedfixed_int4fd_20k_20260523_171841`.
- Prefix INT4 uses the same seed-fixed prefix-quantized family.
- Required sparse reference policy is `hstar_lowbitL`; required prefix reference policy is `hstar_cleanGL`, as in V10 tables.
- Prefix RTE is not used for required runs because the V10 audit marks it incomplete/not comparable.
- Sparse TREC is not used for required runs because the prompt restricts required sparse tasks to SST-5/RTE and warns not to use sparse TREC unless fully audited.
- Existing seed16 full runs are reused; newly submitted runs fill missing seeds only.
- Direction-stream pairing is limited by existing runners. Base train/data seeds are paired by h; the low-bit runner's internal direction seed includes h, so exact identical direction streams across h are not guaranteed.

## Config Rows

```
                   family  task precision        mode                       reference_policy    h_ref                                                                                                              source_path                                                                               run_name_or_config  comparable_v10_family                                                               notes
   high_precision_plateau sst-5      fp32       dense existing latest_main FP32 seed16 sweep      NaN /scratch/jy03364/MeZO_/experiments/main_latest/mezo/roberta-large/sst5/fp32_fp16_h_sweep_11h_seed16_bs64_ckpt1k_20260517                                           fp32 high-precision plateau manifests from latest_main                   True FP32 high-precision plateau chosen from existing main latest sweep.
              prefix_int4 sst-2      int4      prefix                          hstar_cleanGL 0.088642                                        /scratch/jy03364/MeZO_/v10_supplement_results_20260626/v10_table_values_audit.csv     int4_prefix_quantized_seedfixed_sst2_hstar_cleanGL_h0p0886419832658_seed16_full_bs64_step20k                   True           V10 seed-fixed family; no legacy highest-abs sparse mask.
              prefix_int4  trec      int4      prefix                          hstar_cleanGL 0.094990                                        /scratch/jy03364/MeZO_/v10_supplement_results_20260626/v10_table_values_audit.csv     int4_prefix_quantized_seedfixed_trec_hstar_cleanGL_h0p0949896412435_seed16_full_bs64_step20k                   True           V10 seed-fixed family; no legacy highest-abs sparse mask.
     prefix_int4_optional sst-5      int4      prefix                          hstar_cleanGL 0.080472                                        /scratch/jy03364/MeZO_/v10_supplement_results_20260626/v10_table_values_audit.csv     int4_prefix_quantized_seedfixed_sst5_hstar_cleanGL_h0p0804720147886_seed16_full_bs64_step20k                   True           V10 seed-fixed family; no legacy highest-abs sparse mask.
         sparse_p0p1_int4 sst-5      int4 sparse_p0p1                          hstar_lowbitL 0.001367                                        /scratch/jy03364/MeZO_/v10_supplement_results_20260626/v10_table_values_audit.csv int4_sparsep0p1_taskgrad_seedfixed_sst5_hstar_lowbitL_h0p00136650442032_seed16_full_bs64_step20k                   True           V10 seed-fixed family; no legacy highest-abs sparse mask.
         sparse_p0p1_int4   rte      int4 sparse_p0p1                          hstar_lowbitL 0.000359                                        /scratch/jy03364/MeZO_/v10_supplement_results_20260626/v10_table_values_audit.csv int4_sparsep0p1_taskgrad_seedfixed_rte_hstar_lowbitL_h0p000359154678187_seed16_full_bs64_step20k                   True           V10 seed-fixed family; no legacy highest-abs sparse mask.
sparse_p0p1_int4_optional sst-2      int4 sparse_p0p1                          hstar_lowbitL 0.001044                                        /scratch/jy03364/MeZO_/v10_supplement_results_20260626/v10_table_values_audit.csv int4_sparsep0p1_taskgrad_seedfixed_sst2_hstar_lowbitL_h0p00104400574666_seed16_full_bs64_step20k                   True           V10 seed-fixed family; no legacy highest-abs sparse mask.
```

## Launch Summary

```
 high_precision_new_runs  lowbit_new_runs  existing_seed16_reused_lowbit_rows  max_concurrent_lanes target_gpu target_time                               git_commit
                      16               24                                  12                     6       H100    72:00:00 5703318c54ce0aaf0c75e58e6d13ee7a53c7882b
```

Submit with:

```bash
bash v10_medium_supplement_results_20260627/submit_v10_medium_supplement.sh
```

Monitor with:

```bash
OUT_DIR=v10_medium_supplement_results_20260627 bash v10_medium_supplement_results_20260627/monitor_v10_medium_supplement.sh
```

Summarize completed jobs with:

```bash
python v10_medium_supplement_results_20260627/scripts/summarize_v10_medium_results.py v10_medium_supplement_results_20260627
```
