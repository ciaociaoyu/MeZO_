# Final GPTQ-256 / Groupwise-256 INT8 Rerun Report

## Quantizer Status

Exact GPTQ was not available in this code path. The requested GPTQ-256 rerun therefore used the explicitly labeled fallback `groupwise_int8_block256`, implemented as symmetric groupwise weight fake-quant/dequant with group/block size 256 and no Hessian/second-order calibration.

The quantizer report is in `01_quantizer_checks/gptq256_quantizer_report.md`. It records:

- `quantization_exact_gptq = false`
- `quantization_algorithm = groupwise_int8_block256`
- `quantization_algorithm_impl = groupwise_symmetric`
- `bits = 8`
- `group_size = 256`
- `calibration_samples = 0`
- activation quantization was not added
- current QuZO path quantizes all floating model parameters, including LayerNorm, bias, and classifier parameters

## Launch Contract

The main dense run and probe metadata used the expected seed and data contract:

- `seed = 16`
- `data_seed = 16`
- `dataset_mode = full`
- `per_device_train_batch_size = 64`
- `gradient_accumulation_steps = 1`
- `DATALOADER_SHUFFLE=True` was exported, and run config recorded `dataloader_shuffle = true`
- no `SequentialSampler` string appears in the new GPTQ-256 experiment logs; the current logs do not explicitly print the sampler class name

## Probe Window

The dense probe rerun broadly reproduces the previous INT8 h-window finding. The best correlation was at `h=1.5e-3`, with `corr_fd_true=0.990373` and `nMSE_fd_true=0.020274`. `h=2e-3` remained essentially tied (`corr=0.989376`, `nMSE=0.022124`), and `h=3e-3` remained usable (`corr=0.976402`, `nMSE=0.052459`).

Small h is still distorted: `h=1e-4` had `corr=0.695081` and `nMSE=1.49019`. Large h still fails locality despite near-perfect snap geometry: `h=1e-2` had `probe_alignment=0.999646` but `corr=0.21594` and `nMSE=1.06311`.

Conclusion: groupwise/block256 shifts the best probe slightly lower, toward `1.5e-3` to `2e-3`, but it does not remove the need for an h-window.

## Training Results

| route | run | steps | best_eval_acc | last_eval_acc | best_eval_loss | last_eval_loss | status |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| dense FP16-master | `dense_gw256_fp16master_h2e-3_step5000` | 5000 | 0.474239 | 0.360656 | 1.206379 | 1.460353 | late collapse |
| direct INT8 | `direct_gw256_h3e-3_lr1e-5_step100` | 100 | 0.284543 | 0.284543 | 1.570971 | 1.570971 | diagnostic failed |
| residual-grid | `residual_grid_gw256_h3e-3_lr7e-5_clip3_step2000` | 2000 | 0.460187 | 0.447307 | 1.328674 | 1.328674 | mechanically clean |
| sparse FP16-master | `smoke_sparse_gw256_p0p01_hraw0p0006_step20` | 20 | 0.279859 | 0.279859 | 1.620218 | 1.621151 | smoke only |

Dense FP16-master reached the best peak accuracy, but the 5k trajectory was not stable. Eval accuracy peaked at 2500 steps (`0.474239`) and degraded to `0.360656` at 5000. The last-five eval losses were `[1.211541, 1.206379, 1.255737, 1.326675, 1.460353]`, so the late run is a real regression, not only metric noise.

Residual-grid was the most mechanically clean true INT8 lattice route. At 2k, it reached `best_eval_acc=0.460187` with `last_eval_acc=0.447307`; `grid_error_norm=0`, `scale_drift_max=0`, `residual_bound_violation_frac=0`, and `ef_error_norm=0`. Its final commit geometry was much less dense than direct update (`active_frac=0.129804`) but still not ideal (`global_acc_actual_cos=0.417574`, `global_actual_over_acc_norm_ratio=0.717596`).

Direct INT8 remains a poor update backend. The 100-step diagnostic had `active_frac=0.986147`, `cos_intended_actual=0.349846`, and `actual_over_intended_norm_ratio=2.85881`. This is less extreme than the earlier tensor-INT8 distortion reported for direct update, but accuracy stayed near random/early baseline and the update is still near-dense lattice snapping.

Sparse was only smoke-tested because it was slower and lower priority. No full sparse conclusion should be drawn from this rerun.

## Answers

1. Was GPTQ-256 actually used?
   - No exact GPTQ. The experiment used the labeled fallback `groupwise_int8_block256`.

2. Was exact GPTQ used or groupwise fallback?
   - Groupwise fallback. No Hessian/second-order GPTQ calibration was run.

3. Does the rerun reproduce the previous INT8 probe window?
   - Yes, qualitatively. The useful window remains around the low `1e-3` scale, with best results at `1.5e-3` to `2e-3`.

4. Does selected h remain around 2e-3 to 3e-3?
   - Mostly yes, but the best probe shifts slightly down: `1.5e-3` and `2e-3` were strongest; `3e-3` remained usable but weaker.

5. Does dense INT8 + FP16 master still train best inside the window?
   - It has the best peak result in this rerun (`0.474239`), but the h=2e-3 5k run collapsed late. It is still the strongest route by peak accuracy, not yet stable enough by last accuracy.

6. Does direct INT8 update still fail?
   - Yes. It remains near-dense and distorted, with low accuracy.

7. Does sparse behavior match prior sparse behavior?
   - Not enough data. Only a 20-step smoke run was completed.

8. Does residual-grid remain clean and useful?
   - Yes as a diagnostic and secondary backend candidate. It stayed mechanically clean through 2k and reached competitive accuracy, but did not beat the dense FP16-master peak.

9. Did block/group size 256 materially improve or shift the window?
   - It improved small-h probe behavior relative to a more distorted tiny-h regime and shifted the best probe slightly lower. It did not eliminate the locality failure at `h=1e-2`.

10. Should this be included in the paper?
   - Include as a robustness/appendix ablation, not as the main setting. It supports the h-window story under a different weight quantizer, but it is not exact GPTQ and the dense 5k route was not stable to the end.

## Recommendation

Do not promote this as the main quantizer setting. For a tighter robustness ablation, run a smaller follow-up grid around `h=1.5e-3` and `h=2e-3` with either shorter early-stopped reporting or a lower learning rate to check whether the late dense collapse is avoidable. Residual-grid can remain a clean secondary diagnostic/backend, while sparse should be deferred until the dense stability question is resolved.
