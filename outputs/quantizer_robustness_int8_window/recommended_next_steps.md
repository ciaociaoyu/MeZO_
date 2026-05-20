# Recommended Next Steps

1. RoBERTa-large AWQ-style INT8 window: yes based on probe valid_window.
2. OPT-1.3B AWQ-style INT8 window: yes based on probe valid_window.
3. AWQ-style selected_h shift vs RTNClip: RoBERTa=no, OPT=no.
4. AWQ-style small-h visibility failure reduced: not clearly; compare code_change_frac at 1e-5 and 3e-5.
5. AWQ-style large-h locality changed: yes by Richardson/self-consistency at 1e-2.
6. MeZO default h=1e-3 inside estimated window: RoBERTa RTNClip=True, RoBERTa AWQ=True, OPT RTNClip=True, OPT AWQ=True.
7. Qualitative window shape robust to quantizer change: yes for this sanity batch.
8. Add AWQ-style as appendix robustness ablation: yes; keep name awq_style_g128_fake_quant.
9. HQQ-style next: unnecessary for this batch because no easy existing HQQ path was found; run later only if a lightweight shared-grid fake-quant HQQ probe is added.

Plots:
- `outputs/quantizer_robustness_int8_window/plots/roberta_large_sst5_probe_alignment_norm_code.svg`
- `outputs/quantizer_robustness_int8_window/plots/opt_1p3b_sst5_probe_alignment_norm_code.svg`
- `outputs/quantizer_robustness_int8_window/plots/roberta_large_sst5_h_acc_eval_accuracy.svg`
- `outputs/quantizer_robustness_int8_window/plots/opt_1p3b_sst5_h_acc_eval_accuracy.svg`
- `outputs/quantizer_robustness_int8_window/plots/selected_h_comparison.svg`
- `outputs/quantizer_robustness_int8_window/plots/valid_window_overlay.svg`
