# FP32/FP16 Left-Tail H-Sweep Summary

Output root: `/scratch/jy03364/MeZO_/experiments/main_latest/mezo/roberta-large/sst5/fp32_fp16_h_sweep_left_tail_seed16_bs64_ckpt1k_20260519`
Completed runs: 11 / 12
Failed/incomplete runs: 1
NaN-marked runs: 3

## Run Table

| precision | h | status | steps | best_acc | last_acc | best_loss | last_loss | nan |
|---|---:|---|---:|---:|---:|---:|---:|---|
| fp16 | 1e-09 | completed | 19990 | 0.24824355971896955 | 0.24824355971896955 | 1.7724609375 | 1.7724609375 | False |
| fp16 | 1e-08 | completed | 19990 | None | None | None | None | True |
| fp16 | 1e-07 | completed | 19990 | None | None | None | None | True |
| fp16 | 3e-07 | completed | 19990 | 0.21428571428571427 | 0.20140515222482436 | 1.6201171875 | 1.822265625 | True |
| fp16 | 1e-06 | completed | 19990 | 0.30210772833723654 | 0.26229508196721313 | 1.552734375 | 1.5947265625 | False |
| fp16 | 3e-06 | completed | 19990 | 0.4566744730679157 | 0.4566744730679157 | 1.3701171875 | 1.3701171875 | False |
| fp32 | 1e-09 | completed | 19990 | 0.2962529274004684 | 0.27400468384074944 | 1.5518829822540283 | 1.5917367935180664 | False |
| fp32 | 1e-08 | incomplete | 19990 | 0.45081967213114754 | 0.45081967213114754 | 1.309429407119751 | 1.309429407119751 | False |
| fp32 | 1e-07 | completed | 19990 | 0.4812646370023419 | 0.477751756440281 | 1.2079410552978516 | 1.2079410552978516 | False |
| fp32 | 3e-07 | completed | 19990 | 0.48009367681498827 | 0.48009367681498827 | 1.2125359773635864 | 1.2125359773635864 | False |
| fp32 | 1e-06 | completed | 19990 | 0.48243559718969553 | 0.48243559718969553 | 1.21710205078125 | 1.21710205078125 | False |
| fp32 | 3e-06 | completed | 19990 | 0.48009367681498827 | 0.48009367681498827 | 1.2141931056976318 | 1.2141931056976318 | False |

## Interpretation Notes

- This report is updated whenever the summarizer runs; incomplete jobs remain marked as missing/incomplete.
- Left-tail conclusions should only be drawn after all 12 runs complete.
- Merged rows currently available: 34.

