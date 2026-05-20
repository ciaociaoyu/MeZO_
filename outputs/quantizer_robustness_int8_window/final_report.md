# Quantizer Robustness INT8 Window Final Report

Run date: 2026-05-20

Execution:
- One local H100 lane, no scheduler fanout.
- Smoke jobs: 4/4 complete.
- Probe jobs: 4/4 complete.
- Short h-accuracy jobs: 16/16 complete.
- Failed experiment jobs: 0.
- Postprocessing note: the first aggregate report pass hit missing `matplotlib`; postprocess-only regeneration wrote SVG plots instead.

Scope:
- RoBERTa-large / SST-5 and OPT-1.3B / SST-5.
- OPT-1.3B/SST-5 was supported; no SST-2 fallback was used.
- INT8, group size 128, dense two-point MeZO, FP16 master update.
- Quantized modules: Linear weights only. Embeddings, LayerNorm, and bias stayed FP16 for both quantizers.
- Compared `G128_RTNClip_shared_grid_fake_quant` vs `awq_style_g128_fake_quant`.
- HQQ-style skipped because no easy existing shared-grid HQQ path was present.

Probe verdict:
- RoBERTa-large AWQ-style INT8 window exists: yes.
- OPT-1.3B AWQ-style INT8 window exists: yes.
- Selected h did not shift under AWQ-style: RTNClip and AWQ-style both selected `5e-3` for both models.
- MeZO default `1e-3` is inside the estimated window for every model/quantizer pair.
- Small-h visibility failure was not clearly reduced by AWQ-style; `1e-5` was still visible by the code-change threshold in this fake-quant probe, although probe quality was worse than the intermediate region.
- Large-h locality changed under AWQ-style: AWQ-style marked `1e-2` as locality-failing for both models, while RTNClip did so for RoBERTa and not OPT.
- Qualitative window shape is robust enough for an appendix ablation.

Short h-accuracy validation:
- RoBERTa-large used 1000 steps per h policy.
- OPT-1.3B used 500 steps per h policy.
- Eval was capped by the runner (`eval_batches=8`), so these accuracies are sanity-check signals rather than full validation numbers.
- RoBERTa last eval acc: RTNClip `1e-5=0.2734`, `1e-3=0.2852`, `5e-3=0.2559`, `1e-2=0.2754`; AWQ `1e-5=0.2832`, `1e-3=0.2773`, `5e-3=0.2578`, `1e-2=0.2598`.
- OPT last eval acc: RTNClip `1e-5=0.1875`, `1e-3=0.25`, `5e-3=0.25`, `1e-2=0.25`; AWQ `1e-5=0.3438`, `1e-3=0.375`, `5e-3=0.3438`, `1e-2=0.3438`.

Artifacts:
- Output root: `outputs/quantizer_robustness_int8_window/`
- Lightweight package: `outputs/packages/quantizer_robustness_int8_window_results_20260520.tar.gz`
- The package excludes multi-GB `state.pt` tensors. Full checkpoints remain locally in the output tree.
- Plots are in `outputs/quantizer_robustness_int8_window/plots/`.
