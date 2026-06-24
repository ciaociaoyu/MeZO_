# Synthetic Fit Repair Summary

## Why the previous synthetic fit was not clean
The previous broad high-dimensional sweep mixed linear, nonlinear, clipping, and saturation regimes. Linear oracles do not contain a locality tail, and ideal symmetric central differences on smooth functions can show an h^4 squared-error tail rather than h^2. Some old fits were also dominated by extreme small-h rows or clipping regimes.

## Best models for main oracle families
config_id,family,fit_model,p,R2_log,RMSE_log,h_star,h_star_interior,selection_reason,status
combined_clean_d1e4_int4_D1e-3_a8,combined,MIA_loc,,0.999596356271533,0.0918657398566901,0.01,True,interval_aware_model_best,ok
combined_clean_d1e4_int8_D1e-4_a8,combined,MIA_loc,,0.9997912252586062,0.0690952329845983,0.003,True,interval_aware_model_best,ok
combined_clip_appendix_d1e4_int4,combined,MIAp,1.0,0.9997988266667536,0.0701171295922021,0.003,False,interval_aware_model_best,ok;boundary_solution
linear_visibility_d1e4_int4,linear,MIA2,2.0,0.9991285368384636,0.2427564003341148,1.0,False,visibility_only_oracle_not_for_full_u_shape,right_tail_missing;boundary_solution
locality_fp_d1e4_a4,locality,MIA_loc,,0.9999999643312232,0.0015534462932352,0.003,True,interval_aware_model_best,ok
locality_fp_norm_d1e4_a12,locality,MIA_loc,,0.999830032191286,0.0808173077750152,0.1,True,interval_aware_model_best,ok

## M2 strict-success diagnostic

No combined quantized nonlinear config satisfied the strict old-envelope M2 criteria simultaneously. The clean combined configs are explained much better by interval-aware models, especially `MIA_loc`. This is an important negative result: the coarse `alpha/h^2 + beta h^2 + gamma` form is a useful practical envelope, but the clean synthetic central-difference data prefers interval-aware visibility plus measured locality terms.

M2 diagnostic rows are saved in `synthetic_m2_diagnostic.csv`.

## Honest interpretation
- Linear visibility-only oracle is useful for the left/visibility side only; beta and h_star are not meaningful there.
- Full-precision nonlinear central difference generally prefers M4 or learned-p tails when squared error follows h^4.
- Combined quantized nonlinear results should use interval-aware models when they improve log-RMSE over the coarse M2 envelope.
- M2 is best described as a practical envelope/proxy, not a strict Taylor law for every ideal central-difference synthetic oracle.

## Recommended main-paper figures
- `fig_clean_u_shape_model_comparison.pdf`
- `fig_combined_quantized_nonlinear.pdf`
- `fig_interval_aware_vs_coarse.pdf`
- `fig_highdim_window_scaling.pdf`

## Appendix figures
- `fig_visibility_only_linear.pdf`
- `fig_locality_only_fullprecision.pdf`
- `fig_h_range_repair_example.pdf`
