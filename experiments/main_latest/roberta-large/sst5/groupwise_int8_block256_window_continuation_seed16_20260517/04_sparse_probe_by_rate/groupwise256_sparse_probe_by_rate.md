# groupwise_int8_block256 Sparse Probe By Rate

| precision_mode | quantization_algorithm | group_size | block_size | direction_type | sparse_rate | sparse_mode | sparse_rescale | h_raw | h_active | num_probe_rows | probe_active_frac_mean | probe_alignment_mean | probe_norm_ratio_mean | fd_zero_ratio | corr_fd_true | nMSE_fd_true | sign_agreement |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| int8 | groupwise_int8_block256 | 256 | 256 | sparse | 0.003 | bernoulli | inv_sqrt_p | 4.10792e-05 | 0.00075 | 50 | 0.00531798 | 0.907916 | 1.1014 | 0 | 0.926419 | 0.167371 | 0.82 |
| int8 | groupwise_int8_block256 | 256 | 256 | sparse | 0.003 | bernoulli | inv_sqrt_p | 8.21584e-05 | 0.0015 | 50 | 0.0058574 | 0.97095 | 1.02994 | 0 | 0.954974 | 0.0807953 | 0.9 |
| int8 | groupwise_int8_block256 | 256 | 256 | sparse | 0.003 | bernoulli | inv_sqrt_p | 0.000164317 | 0.003 | 50 | 0.00644721 | 0.991576 | 1.00849 | 0 | 0.988908 | 0.0211759 | 0.96 |
| int8 | groupwise_int8_block256 | 256 | 256 | sparse | 0.003 | bernoulli | inv_sqrt_p | 0.000328634 | 0.00600001 | 50 | 0.00758115 | 0.997519 | 1.00249 | 0 | 0.994398 | 0.0112997 | 0.98 |
| int8 | groupwise_int8_block256 | 256 | 256 | sparse | 0.003 | bernoulli | inv_sqrt_p | 0.000657267 | 0.012 | 50 | 0.0111776 | 0.999129 | 1.00087 | 0 | 0.997417 | 0.00487749 | 0.98 |
| int8 | groupwise_int8_block256 | 256 | 256 | sparse | 0.003 | bernoulli | inv_sqrt_p | 0.00131453 | 0.0239999 | 50 | 0.0280519 | 0.99956 | 1.00044 | 0 | 0.988177 | 0.0210843 | 1 |
| int8 | groupwise_int8_block256 | 256 | 256 | sparse | 0.01 | bernoulli | inv_sqrt_p | 7.5e-05 | 0.00075 | 50 | 0.0176536 | 0.908918 | 1.10017 | 0 | 0.910019 | 0.209038 | 0.9 |
| int8 | groupwise_int8_block256 | 256 | 256 | sparse | 0.01 | bernoulli | inv_sqrt_p | 0.00015 | 0.0015 | 50 | 0.0194296 | 0.971472 | 1.02936 | 0 | 0.956198 | 0.0900781 | 0.86 |
| int8 | groupwise_int8_block256 | 256 | 256 | sparse | 0.01 | bernoulli | inv_sqrt_p | 0.0003 | 0.003 | 50 | 0.0213727 | 0.991779 | 1.00828 | 0 | 0.985137 | 0.0323215 | 0.92 |
| int8 | groupwise_int8_block256 | 256 | 256 | sparse | 0.01 | bernoulli | inv_sqrt_p | 0.0006 | 0.006 | 50 | 0.0250801 | 0.997595 | 1.00241 | 0 | 0.992936 | 0.0150781 | 0.92 |
| int8 | groupwise_int8_block256 | 256 | 256 | sparse | 0.01 | bernoulli | inv_sqrt_p | 0.0012 | 0.012 | 50 | 0.0366432 | 0.999167 | 1.00083 | 0 | 0.993941 | 0.0122821 | 1 |
| int8 | groupwise_int8_block256 | 256 | 256 | sparse | 0.01 | bernoulli | inv_sqrt_p | 0.0024 | 0.024 | 50 | 0.0877089 | 0.999587 | 1.00041 | 0 | 0.95842 | 0.0825575 | 0.88 |
| int8 | groupwise_int8_block256 | 256 | 256 | sparse | 0.03 | bernoulli | inv_sqrt_p | 0.000129904 | 0.000750001 | 50 | 0.0523786 | 0.910977 | 1.09772 | 0 | 0.927637 | 0.179581 | 0.88 |
| int8 | groupwise_int8_block256 | 256 | 256 | sparse | 0.03 | bernoulli | inv_sqrt_p | 0.000259808 | 0.0015 | 50 | 0.0575702 | 0.972449 | 1.02832 | 0 | 0.983323 | 0.0393003 | 0.96 |
| int8 | groupwise_int8_block256 | 256 | 256 | sparse | 0.03 | bernoulli | inv_sqrt_p | 0.000519615 | 0.003 | 50 | 0.0631617 | 0.992146 | 1.00791 | 0 | 0.988385 | 0.0228043 | 0.92 |
| int8 | groupwise_int8_block256 | 256 | 256 | sparse | 0.03 | bernoulli | inv_sqrt_p | 0.00103923 | 0.006 | 50 | 0.0735748 | 0.997728 | 1.00228 | 0 | 0.994961 | 0.0100613 | 0.98 |
| int8 | groupwise_int8_block256 | 256 | 256 | sparse | 0.03 | bernoulli | inv_sqrt_p | 0.00207846 | 0.012 | 50 | 0.104768 | 0.999232 | 1.00077 | 0 | 0.989676 | 0.0218251 | 0.96 |
| int8 | groupwise_int8_block256 | 256 | 256 | sparse | 0.03 | bernoulli | inv_sqrt_p | 0.00415692 | 0.024 | 50 | 0.223813 | 0.999637 | 1.00036 | 0 | 0.903907 | 0.184727 | 0.9 |
| int8 | groupwise_int8_block256 | 256 | 256 | sparse | 0.1 | bernoulli | inv_sqrt_p | 0.000237171 | 0.000750001 | 50 | 0.168724 | 0.914625 | 1.09335 | 0 | 0.968434 | 0.0962161 | 0.92 |
| int8 | groupwise_int8_block256 | 256 | 256 | sparse | 0.1 | bernoulli | inv_sqrt_p | 0.000474342 | 0.0015 | 50 | 0.184403 | 0.974067 | 1.02663 | 0 | 0.98898 | 0.0249074 | 0.98 |
| int8 | groupwise_int8_block256 | 256 | 256 | sparse | 0.1 | bernoulli | inv_sqrt_p | 0.000948683 | 0.003 | 50 | 0.200546 | 0.992723 | 1.00733 | 0 | 0.995501 | 0.00969185 | 0.94 |
| int8 | groupwise_int8_block256 | 256 | 256 | sparse | 0.1 | bernoulli | inv_sqrt_p | 0.00189737 | 0.00600001 | 50 | 0.228506 | 0.997937 | 1.00207 | 0 | 0.993441 | 0.0160401 | 0.98 |
| int8 | groupwise_int8_block256 | 256 | 256 | sparse | 0.1 | bernoulli | inv_sqrt_p | 0.00379473 | 0.012 | 50 | 0.302171 | 0.999341 | 1.00066 | 0 | 0.948873 | 0.111761 | 0.9 |
| int8 | groupwise_int8_block256 | 256 | 256 | sparse | 0.1 | bernoulli | inv_sqrt_p | 0.00758947 | 0.024 | 50 | 0.501312 | 0.999729 | 1.00027 | 0 | 0.222111 | 1.06228 | 0.58 |
