# GPTQ-256 Dense Probe Summary

| precision_mode | quantization_algorithm | group_size | direction_type | h_raw | h_active | num_probe_rows | probe_active_frac_mean | probe_alignment_mean | probe_norm_ratio_mean | fd_zero_ratio | corr_fd_true | nMSE_fd_true | sign_agreement |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| int8 | groupwise_int8_block256 | 256 | dense | 0.0001 | 0.0001 | 50 | 0.991308 | 0.462863 | 2.16047 | 0 | 0.695081 | 1.49019 | 0.7 |
| int8 | groupwise_int8_block256 | 256 | dense | 0.0003 | 0.0003 | 50 | 0.994254 | 0.768879 | 1.3006 | 0 | 0.857001 | 0.3312 | 0.86 |
| int8 | groupwise_int8_block256 | 256 | dense | 0.0007 | 0.0007 | 50 | 0.99689 | 0.937547 | 1.06662 | 0 | 0.96603 | 0.0682528 | 0.98 |
| int8 | groupwise_int8_block256 | 256 | dense | 0.001 | 0.001 | 50 | 0.997729 | 0.967802 | 1.03327 | 0 | 0.98278 | 0.0335388 | 0.92 |
| int8 | groupwise_int8_block256 | 256 | dense | 0.0015 | 0.0015 | 50 | 0.998446 | 0.985274 | 1.01494 | 0 | 0.990373 | 0.0202737 | 0.96 |
| int8 | groupwise_int8_block256 | 256 | dense | 0.002 | 0.002 | 50 | 0.998823 | 0.991627 | 1.00844 | 0 | 0.989376 | 0.0221241 | 0.96 |
| int8 | groupwise_int8_block256 | 256 | dense | 0.0025 | 0.0025 | 50 | 0.999054 | 0.994611 | 1.00542 | 0 | 0.982922 | 0.0368804 | 0.96 |
| int8 | groupwise_int8_block256 | 256 | dense | 0.003 | 0.003 | 50 | 0.999209 | 0.996243 | 1.00377 | 0 | 0.976402 | 0.0524592 | 0.94 |
| int8 | groupwise_int8_block256 | 256 | dense | 0.004 | 0.004 | 50 | 0.999404 | 0.997875 | 1.00213 | 0 | 0.944013 | 0.115875 | 0.88 |
| int8 | groupwise_int8_block256 | 256 | dense | 0.005 | 0.005 | 50 | 0.999522 | 0.998632 | 1.00137 | 0 | 0.882561 | 0.224474 | 0.84 |
| int8 | groupwise_int8_block256 | 256 | dense | 0.01 | 0.01 | 50 | 0.999755 | 0.999646 | 1.00035 | 0 | 0.21594 | 1.06311 | 0.62 |
