# h-acc Summary

| model | quantizer | policy | h | status | steps | best_acc | last_acc | run_dir |
| --- | --- | --- | ---: | --- | ---: | ---: | ---: | --- |
| opt-1.3b | awq_style_g128_fake_quant | bad-large | 0.01 | complete | 500 | 0.3438 | 0.3438 | `outputs/quantizer_robustness_int8_window/h_acc/opt_1p3b_sst5/awq/h_1e-2` |
| opt-1.3b | awq_style_g128_fake_quant | default | 0.001 | complete | 500 | 0.375 | 0.375 | `outputs/quantizer_robustness_int8_window/h_acc/opt_1p3b_sst5/awq/h_1e-3` |
| opt-1.3b | awq_style_g128_fake_quant | bad-small | 1e-05 | complete | 500 | 0.3438 | 0.3438 | `outputs/quantizer_robustness_int8_window/h_acc/opt_1p3b_sst5/awq/h_1e-5` |
| opt-1.3b | awq_style_g128_fake_quant | selected_h | 0.005 | complete | 500 | 0.3438 | 0.3438 | `outputs/quantizer_robustness_int8_window/h_acc/opt_1p3b_sst5/awq/h_5e-3` |
| opt-1.3b | G128_RTNClip_shared_grid_fake_quant | bad-large | 0.01 | complete | 500 | 0.25 | 0.25 | `outputs/quantizer_robustness_int8_window/h_acc/opt_1p3b_sst5/rtnclip/h_1e-2` |
| opt-1.3b | G128_RTNClip_shared_grid_fake_quant | default | 0.001 | complete | 500 | 0.25 | 0.25 | `outputs/quantizer_robustness_int8_window/h_acc/opt_1p3b_sst5/rtnclip/h_1e-3` |
| opt-1.3b | G128_RTNClip_shared_grid_fake_quant | bad-small | 1e-05 | complete | 500 | 0.1875 | 0.1875 | `outputs/quantizer_robustness_int8_window/h_acc/opt_1p3b_sst5/rtnclip/h_1e-5` |
| opt-1.3b | G128_RTNClip_shared_grid_fake_quant | selected_h | 0.005 | complete | 500 | 0.25 | 0.25 | `outputs/quantizer_robustness_int8_window/h_acc/opt_1p3b_sst5/rtnclip/h_5e-3` |
| roberta-large | awq_style_g128_fake_quant | bad-large | 0.01 | complete | 1000 | 0.2598 | 0.2598 | `outputs/quantizer_robustness_int8_window/h_acc/roberta_large_sst5/awq/h_1e-2` |
| roberta-large | awq_style_g128_fake_quant | default | 0.001 | complete | 1000 | 0.2832 | 0.2773 | `outputs/quantizer_robustness_int8_window/h_acc/roberta_large_sst5/awq/h_1e-3` |
| roberta-large | awq_style_g128_fake_quant | bad-small | 1e-05 | complete | 1000 | 0.3047 | 0.2832 | `outputs/quantizer_robustness_int8_window/h_acc/roberta_large_sst5/awq/h_1e-5` |
| roberta-large | awq_style_g128_fake_quant | selected_h | 0.005 | complete | 1000 | 0.2617 | 0.2578 | `outputs/quantizer_robustness_int8_window/h_acc/roberta_large_sst5/awq/h_5e-3` |
| roberta-large | G128_RTNClip_shared_grid_fake_quant | bad-large | 0.01 | complete | 1000 | 0.2754 | 0.2754 | `outputs/quantizer_robustness_int8_window/h_acc/roberta_large_sst5/rtnclip/h_1e-2` |
| roberta-large | G128_RTNClip_shared_grid_fake_quant | default | 0.001 | complete | 1000 | 0.2852 | 0.2852 | `outputs/quantizer_robustness_int8_window/h_acc/roberta_large_sst5/rtnclip/h_1e-3` |
| roberta-large | G128_RTNClip_shared_grid_fake_quant | bad-small | 1e-05 | complete | 1000 | 0.2754 | 0.2734 | `outputs/quantizer_robustness_int8_window/h_acc/roberta_large_sst5/rtnclip/h_1e-5` |
| roberta-large | G128_RTNClip_shared_grid_fake_quant | selected_h | 0.005 | complete | 1000 | 0.2559 | 0.2559 | `outputs/quantizer_robustness_int8_window/h_acc/roberta_large_sst5/rtnclip/h_5e-3` |
