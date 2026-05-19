# h-window estimator summary

Default thresholds: align >= 0.7, norm_ratio in [0.7, 1.5], code/active >= 0.01, Richardson <= 0.3.

## Selected h per method

| Precision | Direction | p | Quantizer | Policy | Selected h-scale | h_vis_min | h_loc_max | Valid window | Status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fp16 | dense |  | fp16_forward_oracle | geometry_lower_bound | 1e-05 | 1e-05 | 0.004 | [3e-05, 0.004] | selected |
| fp16 | dense |  | fp16_forward_oracle | log_midpoint_valid | 0.001 | 1e-05 | 0.004 | [3e-05, 0.004] | selected |
| fp16 | dense |  | fp16_forward_oracle | probe_best_corr_fd_true | 0.001 | 1e-05 | 0.004 | [3e-05, 0.004] | retrospective_only |
| fp16 | dense |  | fp16_forward_oracle | richardson_upper_bound | 0.004 | 1e-05 | 0.004 | [3e-05, 0.004] | selected |
| fp16 | dense |  | fp16_forward_oracle | score_min_valid | 0.001 | 1e-05 | 0.004 | [3e-05, 0.004] | selected |
| fp16 | dense |  | fp16_forward_oracle | smallest_valid | 3e-05 | 1e-05 | 0.004 | [3e-05, 0.004] | selected |
| fp32 | dense |  | identity | geometry_lower_bound | 1e-05 | 1e-05 | NA | none | selected |
| fp32 | dense |  | identity | log_midpoint_valid | NA | 1e-05 | NA | none | no_valid_window |
| fp32 | dense |  | identity | probe_best_corr_fd_true | 1e-05 | 1e-05 | NA | none | retrospective_only |
| fp32 | dense |  | identity | richardson_upper_bound | NA | 1e-05 | NA | none | no_fd_local |
| fp32 | dense |  | identity | score_min_valid | NA | 1e-05 | NA | none | no_valid_window |
| fp32 | dense |  | identity | smallest_valid | NA | 1e-05 | NA | none | no_valid_window |
| fp32 | dense |  | identity | geometry_lower_bound | 1e-05 | 1e-05 | 0.004 | [3e-05, 0.004] | selected |
| fp32 | dense |  | identity | log_midpoint_valid | 0.001 | 1e-05 | 0.004 | [3e-05, 0.004] | selected |
| fp32 | dense |  | identity | probe_best_corr_fd_true | 1e-05 | 1e-05 | 0.004 | [3e-05, 0.004] | retrospective_only |
| fp32 | dense |  | identity | richardson_upper_bound | 0.004 | 1e-05 | 0.004 | [3e-05, 0.004] | selected |
| fp32 | dense |  | identity | score_min_valid | 3e-05 | 1e-05 | 0.004 | [3e-05, 0.004] | selected |
| fp32 | dense |  | identity | smallest_valid | 3e-05 | 1e-05 | 0.004 | [3e-05, 0.004] | selected |
| int4 | dense |  | G128_groupwise_RTNClip_fake_quant | geometry_lower_bound | 0.003 | 0.003 | NA | none | selected |
| int4 | dense |  | G128_groupwise_RTNClip_fake_quant | log_midpoint_valid | NA | 0.003 | NA | none | no_valid_window |
| int4 | dense |  | G128_groupwise_RTNClip_fake_quant | richardson_upper_bound | NA | 0.003 | NA | none | no_fd_local |
| int4 | dense |  | G128_groupwise_RTNClip_fake_quant | score_min_valid | NA | 0.003 | NA | none | no_valid_window |
| int4 | dense |  | G128_groupwise_RTNClip_fake_quant | smallest_valid | NA | 0.003 | NA | none | no_valid_window |
| int8 | dense |  | groupwise_int8_block256_historical | geometry_lower_bound | 0.0003 | 0.0003 | 0.004 | [0.001, 0.004] | selected |
| int8 | dense |  | groupwise_int8_block256_historical | log_midpoint_valid | 0.002 | 0.0003 | 0.004 | [0.001, 0.004] | selected |
| int8 | dense |  | groupwise_int8_block256_historical | probe_best_corr_fd_true | 0.0015 | 0.0003 | 0.004 | [0.001, 0.004] | retrospective_only |
| int8 | dense |  | groupwise_int8_block256_historical | richardson_upper_bound | 0.004 | 0.0003 | 0.004 | [0.001, 0.004] | selected |
| int8 | dense |  | groupwise_int8_block256_historical | score_min_valid | 0.002 | 0.0003 | 0.004 | [0.001, 0.004] | selected |
| int8 | dense |  | groupwise_int8_block256_historical | smallest_valid | 0.001 | 0.0003 | 0.004 | [0.001, 0.004] | selected |
| int8 | dense |  | legacy_int8_fp16master_probe | geometry_lower_bound | 0.001 | 0.001 | NA | none | selected |
| int8 | dense |  | legacy_int8_fp16master_probe | log_midpoint_valid | NA | 0.001 | NA | none | no_valid_window |
| int8 | dense |  | legacy_int8_fp16master_probe | probe_best_corr_fd_true | 0.003 | 0.001 | NA | none | retrospective_only |
| int8 | dense |  | legacy_int8_fp16master_probe | richardson_upper_bound | NA | 0.001 | NA | none | no_fd_local |
| int8 | dense |  | legacy_int8_fp16master_probe | score_min_valid | NA | 0.001 | NA | none | no_valid_window |
| int8 | dense |  | legacy_int8_fp16master_probe | smallest_valid | NA | 0.001 | NA | none | no_valid_window |
| int8 | dense |  | G128_groupwise_RTNClip_fake_quant | geometry_lower_bound | 0.001 | 0.001 | NA | none | selected |
| int8 | dense |  | G128_groupwise_RTNClip_fake_quant | log_midpoint_valid | NA | 0.001 | NA | none | no_valid_window |
| int8 | dense |  | G128_groupwise_RTNClip_fake_quant | richardson_upper_bound | NA | 0.001 | NA | none | no_fd_local |
| int8 | dense |  | G128_groupwise_RTNClip_fake_quant | score_min_valid | NA | 0.001 | NA | none | no_valid_window |
| int8 | dense |  | G128_groupwise_RTNClip_fake_quant | smallest_valid | NA | 0.001 | NA | none | no_valid_window |
| int8 | dense |  | G128_groupwise_RTNClip_fake_quant | geometry_lower_bound | 0.0003 | 0.0003 | NA | none | selected |
| int8 | dense |  | G128_groupwise_RTNClip_fake_quant | log_midpoint_valid | NA | 0.0003 | NA | none | no_valid_window |
| int8 | dense |  | G128_groupwise_RTNClip_fake_quant | richardson_upper_bound | NA | 0.0003 | NA | none | no_fd_local |
| int8 | dense |  | G128_groupwise_RTNClip_fake_quant | score_min_valid | NA | 0.0003 | NA | none | no_valid_window |
| int8 | dense |  | G128_groupwise_RTNClip_fake_quant | smallest_valid | NA | 0.0003 | NA | none | no_valid_window |
| int8 | sparse | 0.003 | groupwise_int8_block256_historical | geometry_lower_bound | 0.012 | 0.012 | 0.024 | [0.012, 0.024] | selected |
| int8 | sparse | 0.003 | groupwise_int8_block256_historical | log_midpoint_valid | 0.024 | 0.012 | 0.024 | [0.012, 0.024] | selected |
| int8 | sparse | 0.003 | groupwise_int8_block256_historical | probe_best_corr_fd_true | 0.012 | 0.012 | 0.024 | [0.012, 0.024] | retrospective_only |
| int8 | sparse | 0.003 | groupwise_int8_block256_historical | richardson_upper_bound | 0.024 | 0.012 | 0.024 | [0.012, 0.024] | selected |
| int8 | sparse | 0.003 | groupwise_int8_block256_historical | score_min_valid | 0.012 | 0.012 | 0.024 | [0.012, 0.024] | selected |
| int8 | sparse | 0.003 | groupwise_int8_block256_historical | smallest_valid | 0.012 | 0.012 | 0.024 | [0.012, 0.024] | selected |
| int8 | sparse | 0.003 | legacy_int8_fp16master_probe | geometry_lower_bound | 0.006 | 0.006 | NA | none | selected |
| int8 | sparse | 0.003 | legacy_int8_fp16master_probe | log_midpoint_valid | NA | 0.006 | NA | none | no_valid_window |
| int8 | sparse | 0.003 | legacy_int8_fp16master_probe | probe_best_corr_fd_true | 0.006 | 0.006 | NA | none | retrospective_only |
| int8 | sparse | 0.003 | legacy_int8_fp16master_probe | richardson_upper_bound | NA | 0.006 | NA | none | no_fd_local |
| int8 | sparse | 0.003 | legacy_int8_fp16master_probe | score_min_valid | NA | 0.006 | NA | none | no_valid_window |
| int8 | sparse | 0.003 | legacy_int8_fp16master_probe | smallest_valid | NA | 0.006 | NA | none | no_valid_window |
| int8 | sparse | 0.01 | groupwise_int8_block256_historical | geometry_lower_bound | 0.00075 | 0.00075 | 0.024 | [0.003, 0.024] | selected |
| int8 | sparse | 0.01 | groupwise_int8_block256_historical | log_midpoint_valid | 0.006 | 0.00075 | 0.024 | [0.003, 0.024] | selected |
| int8 | sparse | 0.01 | groupwise_int8_block256_historical | probe_best_corr_fd_true | 0.012 | 0.00075 | 0.024 | [0.003, 0.024] | retrospective_only |
| int8 | sparse | 0.01 | groupwise_int8_block256_historical | richardson_upper_bound | 0.024 | 0.00075 | 0.024 | [0.003, 0.024] | selected |
| int8 | sparse | 0.01 | groupwise_int8_block256_historical | score_min_valid | 0.012 | 0.00075 | 0.024 | [0.003, 0.024] | selected |
| int8 | sparse | 0.01 | groupwise_int8_block256_historical | smallest_valid | 0.003 | 0.00075 | 0.024 | [0.003, 0.024] | selected |
| int8 | sparse | 0.01 | legacy_int8_fp16master_probe | geometry_lower_bound | 0.0015 | 0.0015 | NA | none | selected |
| int8 | sparse | 0.01 | legacy_int8_fp16master_probe | log_midpoint_valid | NA | 0.0015 | NA | none | no_valid_window |
| int8 | sparse | 0.01 | legacy_int8_fp16master_probe | probe_best_corr_fd_true | 0.012 | 0.0015 | NA | none | retrospective_only |
| int8 | sparse | 0.01 | legacy_int8_fp16master_probe | richardson_upper_bound | NA | 0.0015 | NA | none | no_fd_local |
| int8 | sparse | 0.01 | legacy_int8_fp16master_probe | score_min_valid | NA | 0.0015 | NA | none | no_valid_window |
| int8 | sparse | 0.01 | legacy_int8_fp16master_probe | smallest_valid | NA | 0.0015 | NA | none | no_valid_window |
| int8 | sparse | 0.03 | legacy_int8_fp16master_probe | geometry_lower_bound | 0.0015 | 0.0015 | NA | none | selected |
| int8 | sparse | 0.03 | legacy_int8_fp16master_probe | log_midpoint_valid | NA | 0.0015 | NA | none | no_valid_window |
| int8 | sparse | 0.03 | legacy_int8_fp16master_probe | probe_best_corr_fd_true | 0.006 | 0.0015 | NA | none | retrospective_only |
| int8 | sparse | 0.03 | legacy_int8_fp16master_probe | richardson_upper_bound | NA | 0.0015 | NA | none | no_fd_local |
| int8 | sparse | 0.03 | legacy_int8_fp16master_probe | score_min_valid | NA | 0.0015 | NA | none | no_valid_window |
| int8 | sparse | 0.03 | legacy_int8_fp16master_probe | smallest_valid | NA | 0.0015 | NA | none | no_valid_window |
| int8 | sparse | 0.1 | legacy_int8_fp16master_probe | geometry_lower_bound | 0.0015 | 0.0015 | NA | none | selected |
| int8 | sparse | 0.1 | legacy_int8_fp16master_probe | log_midpoint_valid | NA | 0.0015 | NA | none | no_valid_window |
| int8 | sparse | 0.1 | legacy_int8_fp16master_probe | probe_best_corr_fd_true | 0.006 | 0.0015 | NA | none | retrospective_only |
| int8 | sparse | 0.1 | legacy_int8_fp16master_probe | richardson_upper_bound | NA | 0.0015 | NA | none | no_fd_local |
| int8 | sparse | 0.1 | legacy_int8_fp16master_probe | score_min_valid | NA | 0.0015 | NA | none | no_valid_window |
| int8 | sparse | 0.1 | legacy_int8_fp16master_probe | smallest_valid | NA | 0.0015 | NA | none | no_valid_window |

## Windows by precision

| Precision | Direction | p | Quantizer | Geometry-visible | FD-local | Hybrid valid | h=1e-3 | Probe-best corr h |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| int8 | dense |  | groupwise_int8_block256_historical | [0.0003, 0.01] | [0.001, 0.004] | [0.001, 0.004] | valid | 0.0015 |
| int8 | sparse | 0.003 | groupwise_int8_block256_historical | [0.012, 0.024] | [0.0015, 0.024] | [0.012, 0.024] | missing | 0.012 |
| int8 | sparse | 0.01 | groupwise_int8_block256_historical | [0.00075, 0.024] | [0.003, 0.024] | [0.003, 0.024] | missing | 0.012 |
| fp32 | dense |  | identity | [1e-05, 0.01] | none | none | locality_unavailable | 1e-05 |
| int8 | dense |  | legacy_int8_fp16master_probe | [0.001, 0.01] | none | none | locality_unavailable | 0.003 |
| int8 | sparse | 0.003 | legacy_int8_fp16master_probe | [0.006, 0.024] | none | none | missing | 0.006 |
| int8 | sparse | 0.01 | legacy_int8_fp16master_probe | [0.0015, 0.024] | none | none | missing | 0.012 |
| int8 | sparse | 0.03 | legacy_int8_fp16master_probe | [0.0015, 0.024] | none | none | missing | 0.006 |
| int8 | sparse | 0.1 | legacy_int8_fp16master_probe | [0.0015, 0.024] | none | none | missing | 0.006 |
| fp16 | dense |  | fp16_forward_oracle | [1e-05, 0.01] | [3e-05, 0.004] | [3e-05, 0.004] | valid | 0.001 |
| fp32 | dense |  | identity | [1e-05, 0.01] | [3e-05, 0.004] | [3e-05, 0.004] | valid | 1e-05 |
| int4 | dense |  | G128_groupwise_RTNClip_fake_quant | [0.003, 0.01] | none | none | too_small_visibility | NA |
| int8 | dense |  | G128_groupwise_RTNClip_fake_quant | [0.001, 0.001] | none | none | locality_unavailable | NA |
| int8 | dense |  | G128_groupwise_RTNClip_fake_quant | [0.0003, 0.01] | none | none | missing | NA |

## Interpretation

1. Geometry-only diagnostics can estimate the visibility lower bound when effective displacement data exists. Current G128 INT8 geometry rows first pass at h=0.0003; locality is unavailable because the current INT8 G128 artifact lacks fixed-direction h/h2 finite-difference pairs.
2. Current G128 INT4 visibility starts around h=0.003 under default thresholds, while Richardson locality passes at h=none. The hybrid intersection is none, so INT4 shows the expected visibility/locality collision risk.
3. For FP32/FP16, geometry is essentially always visible on the loaded grid; the useful upper side is controlled by self-consistency/locality. Hybrid windows: fp16: [3e-05, 0.004], fp32: [3e-05, 0.004].
4. Historical dense INT8 calibration supports a quantization lower bound near h=0.001 and probe-best corr near h=0.003; this explains why h=1e-3 can be good but is not universal.
5. Sparse probes are reported in both raw h and active-coordinate h_active. The active-coordinate windows are more comparable across p: p=0.003: raw [0.0003286, 0.001315], active [0.006, 0.024]; p=0.01: raw [7.5e-05, 0.0024], active [0.00075, 0.024]; p=0.03: raw [0.0002598, 0.004157], active [0.0015, 0.024]; p=0.1: raw [0.0004743, 0.007589], active [0.0015, 0.024].
6. Between smallest_valid and log_midpoint_valid, log_midpoint_valid is the less brittle policy when the valid interval spans several grid points; smallest_valid is useful as a conservative lower-cost choice but can sit on the visibility boundary for INT8/INT4.
7. Minimal deployable diagnostics are effective displacement geometry plus matched-direction finite differences at h and a smaller reference h. True gradients and training accuracy are useful for validation, not for selecting h.
8. The loss-SNR baseline could not be evaluated from existing artifacts because repeated same-batch base-loss evaluations were not present.

## Threshold sensitivity

| Precision | Direction | Setting | Nonempty combos | Total combos | Most common windows |
| --- | --- | --- | --- | --- | --- |
| fp32 | dense | roberta_sst5_current_dense_probe_ckpt1k | 576 | 576 | [3e-05, 0.001]: 288; [3e-05, 0.004]: 288 |
| fp16 | dense | roberta_sst5_current_dense_probe_ckpt1k | 576 | 576 | [3e-05, 0.004]: 288; [0.001, 0.001]: 144; [0.0001, 0.004]: 144 |
| int8 | dense | historical_groupwise256_int8_probe | 432 | 576 | none: 144; [0.0015, 0.0025]: 144; [0.001, 0.004]: 144 |
| int8 | sparse p=0.01 | historical_groupwise256_int8_probe | 396 | 576 | none: 180; [0.006, 0.012]: 108; [0.003, 0.024]: 108 |
| int8 | sparse p=0.003 | historical_groupwise256_int8_probe | 324 | 576 | none: 252; [0.0015, 0.024]: 144; [0.012, 0.024]: 108 |
| int4 | dense | rtnclip_g128_int4_current_probe | 72 | 576 | none: 504; [0.005, 0.005]: 72 |

The final recommendation uses the default threshold set above; the grid shows where a conclusion depends on relaxing locality or visibility.

## Missing artifacts and probe-only commands

- Current G128 RTNClip INT8 does not have a complete fixed-batch/fixed-direction h-grid with Richardson pairs; only geometry diagnostics/smoke rows were found.
- Repeated same-batch base-loss evaluations for the loss-SNR floor baseline were not found.
- Current G128 RTNClip sparse p=0.01 and p=0.003 probe-only h-grids were not found; historical groupwise256 sparse INT8 probes are included as reference only.

Suggested probe-only extensions, if the missing G128 RTNClip grids are needed:

```bash
# INT8 G128 RTNClip fixed-batch/fixed-direction h-grid with Richardson pairs
CUDA_VISIBLE_DEVICES=0 DATALOADER_SHUFFLE=True python tools/rtnclip_roberta_sst5_batch.py --output_root outputs/rtnclip_lowbit_roberta_sst5_seed16_20260519_batch --bitwidth 8 --probe_dirs 16 probe-int4

# INT4 G128 RTNClip fixed-batch/fixed-direction h-grid, using the same generic probe-only path
CUDA_VISIBLE_DEVICES=0 DATALOADER_SHUFFLE=True python tools/rtnclip_roberta_sst5_batch.py --output_root outputs/rtnclip_lowbit_roberta_sst5_seed16_20260519_batch --bitwidth 4 --probe_dirs 16 probe-int4

# Sparse G128 RTNClip probe-only extension for p in {0.01, 0.003}; exact CLI may need adding to the smoke/probe harness
CUDA_VISIBLE_DEVICES=0 DATALOADER_SHUFFLE=True python tools/window_estimation/estimate_h_window.py --suggest-sparse-rtnclip-probe
```
