# Smoke Summary

All smoke runs passed: yes

| model | dataset | quantizer | status | steps | sampler | active_frac | alignment | norm_ratio | run_dir |
| --- | --- | --- | --- | ---: | --- | ---: | ---: | ---: | --- |
| roberta-large | SST-5 | G128_RTNClip_shared_grid_fake_quant | complete | 20 | RandomSampler | 0.9108 | 0.9948 | 1.001 | `outputs/quantizer_robustness_int8_window/smoke/roberta_large_sst5/rtnclip/h_1e-3` |
| roberta-large | SST-5 | awq_style_g128_fake_quant | complete | 20 | RandomSampler | 0.9108 | 0.9947 | 1.001 | `outputs/quantizer_robustness_int8_window/smoke/roberta_large_sst5/awq/h_1e-3` |
| opt-1.3b | SST-5 | G128_RTNClip_shared_grid_fake_quant | complete | 20 | RandomSampler | 0.8994 | 0.993 | 1.003 | `outputs/quantizer_robustness_int8_window/smoke/opt_1p3b_sst5/rtnclip/h_1e-3` |
| opt-1.3b | SST-5 | awq_style_g128_fake_quant | complete | 20 | RandomSampler | 0.8994 | 0.9928 | 1.002 | `outputs/quantizer_robustness_int8_window/smoke/opt_1p3b_sst5/awq/h_1e-3` |
