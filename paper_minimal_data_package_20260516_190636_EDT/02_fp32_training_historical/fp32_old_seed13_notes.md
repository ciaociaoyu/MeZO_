# FP32 Old Seed13 / 20k Notes

Source family: `medium_models/sh_file/sst5/bs32/h_precision_sweep/`.

Label this evidence as `historical_old_sweep`: old seed13, single seed, 20k steps where confirmed by `h_precision_summary.csv`, old code path. Treat it as a provisional replacement for clean modern FP32 training validation.

Allowed use in paper:

- FP32 has a broad usable h range in the old sweep.
- Probe MSE/correlation can rule out very bad h, especially large/nonlocal h such as `1e-2`.
- Within good h values, final/eval accuracy can be relatively flat or noisy.
- Minimum MSE is not necessarily the best final-accuracy point.

Caveats:

- Do not label this as modern FP32 training validation.
- It is single-seed historical evidence.
- Dataloader shuffle / old code path caveats should be mentioned if using it in main text.
- Prefer clean modern FP32 validation before final submission.
