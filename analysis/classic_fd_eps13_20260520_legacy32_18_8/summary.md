# Classic FD eps^(1/3) check

This is a formula/diagnostic aggregation only; no training was launched.

## Naive eps^(1/3)
| setting | epsilon/delta | eps^(1/3) | note |
|---|---:|---:|---|
| unit_fp32_machine_eps | 1.1920929e-07 | 0.0049215666 | central-FD textbook unit-scale h |
| unit_fp16_machine_eps | 0.0009765625 | 0.099212566 | central-FD textbook unit-scale h |
| unit_bf16_machine_eps | 0.0078125 | 0.19842513 | included because INT8 code-step unit scale is often approximated this way |
| unit_int8_symmetric_code_eps_1_over_127 | 0.0078740157 | 0.19894457 | not the RTNClip dequantized groupwise scale |
| roberta_sst5_fp32_ulp_rms_delta | 6.5887116e-09 | 0.0018747074 | effective parameter snapping RMS from prior hstar package |
| roberta_sst5_fp16_ulp_rms_delta | 5.1141356e-05 | 0.037118528 | effective parameter snapping RMS from prior hstar package |
| roberta_sst5_fp16_strict_delta_ulp_rms | 4.4660731e-05 | 0.035479319 | strict FP16 package Delta; eps^(1/3) only, not h4/h6 formula |
| roberta_sst5_int8_rtnclip_g128_dequant_delta | 0.00093492336 | 0.097781945 | RTNClip G128 dequant reconstruction-derived Delta; eps^(1/3) only |

## Model-aware references from existing diagnostics
| setting | formula | h_cont | empirical/reference h | nMSE | corr |
|---|---|---:|---:|---:|---:|
| roberta_sst5_fp32_old_h4_absG_Lq90 | old_h4_L_smooth | 1.5617663e-06 | 1e-05 | 1.553072e-06 | 0.99999923 |
| roberta_sst5_fp16_strict | h4_fdG | 9.9190281e-05 | 0.0003 | 0.0050072857 | 0.99740446 |
| roberta_sst5_fp16_strict | h6_fdG_S3 | 0.0010425292 | 0.0003 | 0.0035171295 | 0.99830271 |
| roberta_sst5_fp16_strict | h6_fdG_rho3 | 0.00088046673 | 0.0003 | 0.0029414337 | 0.99854285 |
| roberta_sst5_fp16_strict | h6_fdG_rho3_gate | 0.00044023336 | 0.0003 | 0.0032699125 | 0.99837417 |
| roberta_sst5_int8_rtnclip_g128 | old_h4_Lsmooth_rtnclip_quantized_values | 0.00053925912 | 0.0015 |  |  |
| roberta_sst5_int8_rtnclip_g128 | new_h6_rho3_gate_rtnclip_quantized_values | 0.0013926695 | 0.0015 |  |  |
| roberta_sst5_int8_rtnclip_g128 | new_h6_S3_empirical_third_moment | 0.0030790361 | 0.0015 |  |  |

## h3=1e-5 high-order difference noise reference
For RoBERTa-large/SST-5 seed16 FP16 strict T3, h3=1e-5 has S3_sq=2.8355096e+17, rho3_q90=9.0431267e-05, low_h3_noise_suspected=True.

Interpretation: the textbook unit-scale eps^(1/3) values are not directly the MeZO h for low-bit/FP16 raw-Gaussian perturbations. The model-aware h4/h6 references are the relevant comparison.
