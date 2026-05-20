# MSE-bound h-window summary

## Main Fits
| Precision | Quantizer | Direction | Coord | Status | y source | h* | W2 | W_tau=0.1 | Selected |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| int8 | groupwise_int8_block256_historical | dense | raw | ok | {"nMSE_fd_true": 11} | 0.001143 | [0.0007, 0.002] | [0.0007, 0.003] | 0.001 |
| int8 | groupwise_int8_block256_historical | sparse p=0.003 | active | ok | {"nMSE_fd_true": 6} | 0.008497 | [0.003, 0.024] | [0.0015, 0.024] | 0.006 |
| int8 | groupwise_int8_block256_historical | sparse p=0.01 | active | ok | {"nMSE_fd_true": 6} | 0.005708 | [0.003, 0.012] | [0.0015, 0.024] | 0.006 |
| bf16 | bf16_forward_oracle | dense | raw | ok | {"nMSE_fd_true": 11} | 0.0005851 | [0.0003, 0.001] | [0.0003, 0.003] | 0.0003 |
| fp32 | identity | dense | raw | alpha_zero | {"nMSE_fd_true": 11} | NA | [1e-05, 1e-05] | [1e-05, 0.003] | 1e-05 |
| int8 | legacy_int8_fp16master_probe | dense | raw | ok | {"nMSE_fd_true": 11} | 0.0009758 | [0.0003, 0.004] | none | 0.001 |
| int8 | legacy_int8_fp16master_probe | sparse p=0.003 | active | ok | {"nMSE_fd_true": 6} | 0.01667 | [0.003, 0.024] | [0.003, 0.024] | 0.006 |
| int8 | legacy_int8_fp16master_probe | sparse p=0.01 | active | ok | {"nMSE_fd_true": 6} | 0.009638 | [0.003, 0.024] | none | 0.006 |
| int8 | legacy_int8_fp16master_probe | sparse p=0.03 | active | ok | {"nMSE_fd_true": 6} | 0.006109 | [0.003, 0.012] | [0.003, 0.012] | 0.006 |
| int8 | legacy_int8_fp16master_probe | sparse p=0.1 | active | ok | {"nMSE_fd_true": 6} | 0.003659 | [0.003, 0.006] | [0.003, 0.006] | 0.003 |
| fp16 | fp16_forward_oracle | dense | raw | ok | {"nMSE_fd_true": 11} | 0.0002099 | [0.0001, 0.0003] | [3e-05, 0.003] | 0.0001 |
| fp32 | identity | dense | raw | ok | {"nMSE_fd_true": 11} | 1.76e-05 | [1e-05, 3e-05] | [1e-05, 0.003] | 3e-05 |
| int4 | G128_groupwise_RTNClip_fake_quant | dense | raw | proxy_only;beta_zero | {"geometry_fd_proxy": 11} | NA | [3e-05, 0.01] | none | 0.0003 |
| int8 | G128_groupwise_RTNClip_fake_quant | dense | raw | proxy_only;no_fit | {"geometry_fd_proxy": 1} | NA | none | none | NA |
| int8 | G128_groupwise_RTNClip_fake_quant | dense | raw | proxy_only;beta_zero | {"geometry_fd_proxy": 6} | NA | [0.0001, 0.01] | none | 0.0003 |

## Answers

- Empirical MSE envelope fit: 17 non-proxy groups fit cleanly; poor/proxy/no-fit groups are marked in `mse_bound_window_fits.csv`.
- h* moves with precision where true nMSE exists: bf16: median h*=0.0005851; fp16: median h*=0.0002099; fp32: median h*=1.76e-05; int8: median h*=0.00115
- Theory proxy agreement: median agreement factor is 1 where both exist; this is a consistency check because the proxy uses fitted coefficients.
- FP32/FP16 recovery: fp32 W2=[1e-05, 1e-05], h*=NA; fp16 W2=[0.0001, 0.0003], h*=0.0002099; fp32 W2=[1e-05, 3e-05], h*=1.76e-05
- INT8 shift: historical dense/sparse INT8 nMSE fits shift the W2 window upward relative to FP32/FP16 when quantization visibility is poor.
- INT4 collision: current G128 INT4 has no true nMSE curve. Proxy fits are prototype-only; W_tau rows should not be treated as paper-ready until a true-gradient probe is run.
- Sparse h_active: active-coordinate fits are written for 6 sparse groups and should be compared against raw fits in the CSV; historical p=0.01/p=0.003 active windows align better than raw h intervals.
- Recommended paper estimator: use the empirical nMSE-envelope fit as the main method where true-gradient probe nMSE exists; use the theory proxy as supporting explanation, not the main selector yet.
- Most robust selected-h policy: `log_midpoint_W2`, because it avoids sitting exactly on the small-h visibility boundary and does not bias toward `1e-3`.
- Additional cheap probe needed: current G128 RTNClip INT8/INT4 true-gradient h-grid probes, plus sparse p=0.01 and p=0.003 if sparse low-bit claims are needed.

## INT4 Window-Collision Check

| Quantizer | Coord | Status | h* | W2 | W_tau=.05 | W_tau=.1 | W_tau=.2 | W_tau=.1 empty? |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| G128_groupwise_RTNClip_fake_quant | raw | proxy_only;beta_zero | NA | [3e-05, 0.01] | none | none | none | yes |
