# Fresh Probe Window Result Summary

This folder was generated from a fresh probe run. It does not reuse the historical `roberta_sst5_theoretical_windows_20260627/raw_probe_metrics.csv`.

- model/task: RoBERTa-large / SST-5 full data
- precisions: FP16 and INT4
- probe target: `d_star=<grad,u>`, `d_h=[F(w+hu)-F(w-hu)]/(2h)`, `e_h=d_h-d_star`
- vector rho: `mean(e_h^2 ||u||^2) / V_dir_sample`; scalar nMSE is reported separately.

## fp16

- d_h=0 -> nonzero transition: `5e-06`.
- Empirical accuracy good set: `[3e-06, 0.0015]`.
- Practical primary window: `[0.0003, 0.003]`.
- Practical relaxed window: `[7e-05, 0.003]`.
- Smooth rho fit status: `window`; W1: `[1.27691e-05, 0.0117266]`.
- Interpretation: `practical default-safe; smooth fit optional`.

- Best scalar nMSE point in this fresh grid: h=`0.0003`, nMSE=`0.00448208`, corr=`0.9977570742090754`.

## int4

- d_h=0 -> nonzero transition: `1e-05`.
- Empirical accuracy good set: `[0.001, 0.001]`.
- Practical primary window: `none`.
- Practical relaxed window: `none`.
- Smooth rho fit status: `window`; W1: `[0.00363621, 0.00478003]`.
- Interpretation: `empirical default-safe, no practical probe certificate`.

- Best scalar nMSE point in this fresh grid: h=`0.005`, nMSE=`0.962111`, corr=`0.5546412213091806`.
