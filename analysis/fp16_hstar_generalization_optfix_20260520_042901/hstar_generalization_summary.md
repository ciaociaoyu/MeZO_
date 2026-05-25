# FP16 h-star generalization summary

Generated: 2026-05-20T04:29:09.048877

Primary selector: `calibrated_hstar_absG_Lclean32_q90`.

## Seed robustness: RoBERTa-large/SST-5 FP16

| model | dataset | seed | hstar h | empirical min-MSE h | nmse ratio | corr gap | pass | L_h2 | G_h |
|---|---|---:|---:|---:|---:|---:|---|---:|---:|
| _none completed_ | | | | | | | | | |

## Task robustness: RoBERTa-large/RTE FP16

| model | dataset | seed | hstar h | empirical min-MSE h | nmse ratio | corr gap | pass | L_h2 | G_h |
|---|---|---:|---:|---:|---:|---:|---|---:|---:|
| _none completed_ | | | | | | | | | |

## Model robustness: OPT/SST-2 FP16

| model | dataset | seed | hstar h | empirical min-MSE h | nmse ratio | corr gap | pass | L_h2 | G_h |
|---|---|---:|---:|---:|---:|---:|---|---:|---:|
| _none completed_ | | | | | | | | | |

## Skipped settings

- C_model OPT-1.3B sst-2 seed 16: ValueError('Due to a serious vulnerability issue in `torch.load`, even with `weights_only=True`, we now require users to upgrade torch to at least v2.6 in order to use the function. This version restriction does not apply when loading files with safetensors.\nSee the vulnerability report here https://nvd.nist.gov/vuln/detail/CVE-2025-32434')
- C_model OPT-1.3B sst-2 seed 17: ValueError('Due to a serious vulnerability issue in `torch.load`, even with `weights_only=True`, we now require users to upgrade torch to at least v2.6 in order to use the function. This version restriction does not apply when loading files with safetensors.\nSee the vulnerability report here https://nvd.nist.gov/vuln/detail/CVE-2025-32434')

