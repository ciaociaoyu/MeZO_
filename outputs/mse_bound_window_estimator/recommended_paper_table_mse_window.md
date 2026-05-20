# Recommended paper table: MSE-window estimator

| Precision | Quantizer | Direction | Empirical h* | Theory h* | Estimated window | Default valid? | Oracle/probe-best | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| int8 | groupwise_int8_block256_historical | dense | 0.001143 | 0.001143 | [0.0007, 0.002] | yes | 0.0015 | paper-ready |
| int8 | groupwise_int8_block256_historical | sparse p=0.003 (active) | 0.008497 | 0.008497 | [0.003, 0.024] | no | 0.012 | paper-ready |
| int8 | groupwise_int8_block256_historical | sparse p=0.01 (active) | 0.005708 | 0.005708 | [0.003, 0.012] | no | 0.012 | paper-ready |
| bf16 | bf16_forward_oracle | dense | 0.0005851 | 0.0005851 | [0.0003, 0.001] | yes | 0.001 | paper-ready |
| fp32 | identity | dense | NA | NA | [1e-05, 1e-05] | no | 1e-05 | prototype |
| int8 | legacy_int8_fp16master_probe | dense | 0.0009758 | 0.0009758 | [0.0003, 0.004] | yes | 0.003 | paper-ready |
| int8 | legacy_int8_fp16master_probe | sparse p=0.003 (active) | 0.01667 | 0.01667 | [0.003, 0.024] | no | 0.006 | paper-ready |
| int8 | legacy_int8_fp16master_probe | sparse p=0.01 (active) | 0.009638 | 0.009638 | [0.003, 0.024] | no | 0.012 | paper-ready |
| int8 | legacy_int8_fp16master_probe | sparse p=0.03 (active) | 0.006109 | 0.006109 | [0.003, 0.012] | no | 0.006 | paper-ready |
| int8 | legacy_int8_fp16master_probe | sparse p=0.1 (active) | 0.003659 | 0.003659 | [0.003, 0.006] | no | 0.003 | paper-ready |
| fp16 | fp16_forward_oracle | dense | 0.0002099 | 0.0002099 | [0.0001, 0.0003] | no | 0.0003 | paper-ready |
| fp32 | identity | dense | 1.76e-05 | 1.76e-05 | [1e-05, 3e-05] | no | 1e-05 | paper-ready |
| int4 | G128_groupwise_RTNClip_fake_quant | dense | NA | NA | proxy-only [3e-05, 0.01] | NA | NA | collapsed/missing true-nMSE |
| int8 | G128_groupwise_RTNClip_fake_quant | dense | NA | NA | proxy-only none | NA | NA | prototype |
| int8 | G128_groupwise_RTNClip_fake_quant | dense | NA | NA | proxy-only [0.0001, 0.01] | NA | NA | prototype |
