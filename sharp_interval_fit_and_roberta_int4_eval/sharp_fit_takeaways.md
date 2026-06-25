# Sharp fit takeaways

- M2, M4, Mp, M_sharp_norm, M_sharp_constrained, and MIA_loc were fit with nonnegative coefficients.
- Main ranking uses log-space RMSE; rows with clipping >5% are excluded when enough clean points exist.
- h_sharp is probe-only: it is selected from fitted/probe metrics, not from training accuracy.
- h_safe keeps h=1e-3 whenever default lies inside the sharp window and passes visibility/locality checks.

## Best log-space fits

| group_key | fit_model | R2_log | RMSE_log | h_star_pred | status |
| --- | --- | --- | --- | --- | --- |
| roberta-large|mnli|int4|prefix | MIA_loc | 1 | 0 | 3.33333e-06 | boundary_solution |
| roberta-large|mnli|int4|dense | M_sharp_constrained | 1 | 0 | 0.00133528 | boundary_solution |
| roberta-large|mnli|int4|dense | M_sharp_norm | 1 | 0 | 0.00133528 | boundary_solution |
| roberta-large|unknown|int4|dense | M_sharp_constrained | 1 | 0 | 0.0102227 | ok |
| roberta-large|mnli|int4|dense | MIA_loc | 1 | 0 | 0.00133528 | boundary_solution |
| roberta-large|sst-5|int4|dense | M_sharp_norm | 1 | 0 | 0.0100328 | boundary_solution |
| roberta-large|sst-5|int4|prefix | MIA_loc | 1 | 0 | 3.33333e-06 | boundary_solution |
| roberta-large|sst-2|int4|dense | M_sharp_constrained | 1 | 0 | 0.00152225 | boundary_solution |
| roberta-large|sst-2|int4|prefix | MIA_loc | 1 | 0 | 3.33333e-06 | boundary_solution |
| roberta-large|sst-5|int4|sparse_p0p1 | MIA_loc | 1 | 0 | 0.0100328 | boundary_solution |
| roberta-large|sst-5|int4|sparse_p0p1 | M_sharp_norm | 1 | 0 | 0.0100328 | boundary_solution |
| roberta-large|rte|int4|sparse_p0p1 | M_sharp_norm | 1 | 0 | 0.00701635 | boundary_solution |
| roberta-large|rte|int4|prefix | MIA_loc | 1 | 0 | 3.33333e-06 | boundary_solution |
| roberta-large|trec|int4|prefix | MIA_loc | 1 | 0 | 0.0966253 | boundary_solution |
| roberta-large|trec|int4|dense | M_sharp_constrained | 1 | 0 | 0.00302864 | boundary_solution |
