# FP16 Old Seed13 / 20k Notes

Source family: `medium_models/sh_file/sst5/bs32/h_precision_sweep/`.

Label this evidence as `historical_old_sweep`: old FP16 seed13, single seed, 20k steps where confirmed, old code path. Do not describe it as BF16.

Observed pattern from the extracted summary:

- FP16 old seed13 shows a strong relation between low probe MSE/correlation quality and trainability.
- Bad tiny h can produce NaNs or near-random behavior in the old sweep.
- Bad large h collapses or performs poorly.
- Best training h is not necessarily the minimum-MSE h. In the extracted table, the minimum-MSE row is `h=3e-4` while the best last5-eval-acc row is `h=3e-5`.
- `h=3e-5` appears to balance finite-difference quality and update amplitude in the old FP16 sweep if using the update-scale proxy column. Treat this as a hypothesis supported by historical data, not a final universal rule.

Caveats:

- Historical, single-seed, old code path.
- Not a clean modern BF16 validation.
- Use for MSE/trainability intuition and appendix/context unless a clean modern FP16/BF16 validation is unavailable.
