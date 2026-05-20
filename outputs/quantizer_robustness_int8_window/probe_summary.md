# Probe Summary

| model | quantizer | h_vis_min | h_loc_max | selected_h | valid_window | failure_mode | run_dir |
| --- | --- | ---: | ---: | ---: | --- | --- | --- |
| opt-1.3b | awq_style_g128_fake_quant | 1e-05 | 0.005 | 0.005 | True | window_exists | `outputs/quantizer_robustness_int8_window/probe/opt_1p3b_sst5/awq` |
| opt-1.3b | G128_RTNClip_shared_grid_fake_quant | 1e-05 | 0.01 | 0.005 | True | window_exists | `outputs/quantizer_robustness_int8_window/probe/opt_1p3b_sst5/rtnclip` |
| roberta-large | awq_style_g128_fake_quant | 1e-05 | 0.005 | 0.005 | True | window_exists | `outputs/quantizer_robustness_int8_window/probe/roberta_large_sst5/awq` |
| roberta-large | G128_RTNClip_shared_grid_fake_quant | 1e-05 | 0.005 | 0.005 | True | window_exists | `outputs/quantizer_robustness_int8_window/probe/roberta_large_sst5/rtnclip` |
