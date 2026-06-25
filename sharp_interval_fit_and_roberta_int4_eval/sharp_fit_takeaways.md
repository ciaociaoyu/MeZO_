# Sharp fit takeaways

- M2, M4, Mp, M_sharp_norm, M_sharp_constrained, and MIA_loc were fit with nonnegative coefficients.
- Fit target is paper-compatible loss-level directional MSE/NMSE only: `(d_Q(h,u)-d_ref(u))^2`, optionally normalized by `E[d_ref^2]`.
- Interval geometry metrics (`A_cross`, `A_interval`, effective-displacement lowbit nMSE) are explanatory covariates, not the target.
- Main ranking uses log-space RMSE; rows with clipping >5% are excluded when enough clean points exist.
- h_sharp is probe-only: it is selected from fitted/probe metrics, not from training accuracy.
- h_safe keeps h=1e-3 whenever default lies inside the sharp window and passes visibility/locality checks.

## Best log-space fits

| group_key | fit_model | R2_log | RMSE_log | h_star_pred | status |
| --- | --- | --- | --- | --- | --- |
| locality_fp_d1e4_a4 | MIA_loc | 1 | 0.000670159 | 0.00297635 | ok |
| scaling_d1000_p0.1_int8_D0.0001 | MIA_loc | 0.999868 | 0.0149335 | 0.00313502 | ok |
| scaling_d1000_p0.01_int8_D0.001 | MIA_loc | 0.999533 | 0.0161905 | 0.00841031 | ok |
| scaling_d10000_p0.01_int8_D0.001 | MIA_loc | 0.99975 | 0.0170161 | 0.00758052 | ok |
| scaling_d100000_p1_int8_D1e-05 | MIA_loc | 0.999936 | 0.0174632 | 0.00982836 | ok |
| scaling_d10000_p0.01_int4_D0.0001 | MIA_loc | 0.99986 | 0.0175927 | 0.00297635 | ok |
| scaling_d10000_p0.01_int4_D0.001 | MIA_loc | 0.999701 | 0.0188946 | 0.00982836 | ok |
| scaling_d100000_p1_int4_D1e-05 | MIA_loc | 0.999916 | 0.02011 | 0.00982836 | ok |
| scaling_d10000_p0.1_int8_D0.001 | MIA_loc | 0.999894 | 0.0225064 | 0.00982836 | ok |
| scaling_d1000_p0.01_int8_D0.0001 | MIA_loc | 0.999704 | 0.0226837 | 0.00297635 | ok |
| scaling_d1000000_p1_int4_D1e-05 | MIA_loc | 0.999874 | 0.024473 | 0.00982836 | ok |
| scaling_d1000_p0.1_int8_D0.001 | MIA_loc | 0.999498 | 0.0246176 | 0.00982836 | ok |
| scaling_d10000_p1_int8_D0.0001 | MIA_loc | 0.999693 | 0.02679 | 0.00982836 | ok |
| scaling_d10000_p1_int4_D0.001 | MIA_loc | 0.999646 | 0.0277625 | 0.00982836 | ok |
| scaling_d1000_p1_int4_D0.0001 | MIA_loc | 0.999682 | 0.0282081 | 0.00982836 | ok |
