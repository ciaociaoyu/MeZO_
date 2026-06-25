# Sharp interval-aware fit target check

## Required paper target

The paper-defined directional MSE is

```text
A_true(h) = E[(d_Q(h, u) - d_ref(u))^2]

d_Q(h, u) =
  [L(Q(w + h u)) - L(Q(w - h u))] / (2h)

d_ref(u) =
  <grad L(w), u>
```

Some existing files store the normalized version

```text
E[(d_Q - d_ref)^2] / (E[d_ref^2] + eps)
```

This is still a loss-level directional MSE target up to an h-independent scale,
so it is acceptable for h-window fitting. It must not be confused with
interval/geometry-only displacement error.

## What the pipeline was fitting before this check

Before the fix in `tools/sharp_interval_workflow.py`, the synthetic rows were
fit against the synthetic `A_true` column, which is a directional-MSE-style
target.

However, several real RoBERTa INT4 rows were incorrectly loaded as follows:

```text
A_true    <- lowbit_true_nmse
nMSE_loss <- lowbit_true_nmse
A_cross   <- lowbit_true_nmse or fd_true_nmse
```

For the recent sharp RoBERTa INT4 run, the file
`outputs/sharp_interval_roberta_int4_eval/int4_hsearch_summary.csv` records

```text
nMSE_metric_version = dequantized_effective_displacement_nmse_v1
true_dir_lowbit_field_name = delta_q_dequantized_fp32_over_2h
```

That means `lowbit_true_nmse` is based on

```text
Delta_Q / (2h)
```

and not on

```text
[L(Q(w+h u)) - L(Q(w-h u))] / (2h) - <grad L(w), u>.
```

Required explicit answer:

```text
Before the fix, the real-model sharp fit partly fit effective-displacement
geometry (`lowbit_true_nmse=dequantized_effective_displacement_nmse_v1`), not
the paper-defined loss-level directional MSE.
```

This also created a leakage path: the same `lowbit_true_nmse` could appear both
as the target and as `A_cross`, making real-model sharp fits look artificially
perfect.

## Data flow after the fix

The corrected data flow is:

```text
raw quantizer / existing probe files
    |
    +--> geometry covariates:
    |      A_cross / A_interval / delta_visibility_nmse
    |      p_active, V_align, V_norm, p_clip, relative_disp
    |
    +--> canonical target only if available:
           fd_true_mse
           fd_true_nmse
           nMSE_fd_true
           fd_true_nmse_default
    |
    v
fit target A_fit =
    A_true or normalized A_true only when
    target_is_paper_directional_mse = True
```

Then the sharp models fit

```text
A_fit(h) ~ a * A_cross(h)
         + b * h^2 * sqrt(A_cross(h))
         + c * h^4
         + gamma
```

or the corresponding M2/M4/Mp/MIA_loc baselines.

The final fitted quantity is now:

```text
paper-compatible loss-level directional MSE/NMSE:
E[(d_Q(h,u)-d_ref(u))^2], optionally normalized by E[d_ref(u)^2].
```

## Code changes made

`tools/sharp_interval_workflow.py` was updated to add:

- `canonical_directional_target(row)`;
- `target_kind`;
- `target_is_paper_directional_mse`;
- a strict `prepare_fit_frame()` filter that excludes non-canonical target rows;
- audit text explaining that geometry-only fields are covariates, not targets.

Accepted target fields:

```text
fd_true_mse
fd_true_nmse
nMSE_fd_true
fd_true_nmse_default
```

Rejected as fit targets:

```text
lowbit_true_nmse
default_nmse
delta_visibility_nmse
A_cross
A_interval
sigma_raw2
p_active
```

`lowbit_true_nmse` may become acceptable in a future run only if its
`nMSE_metric_version` explicitly labels it as a loss-level directional metric.
The current known version is geometry-only and is rejected.

## Corrected fit-input counts

After regenerating `sharp_interval_fit_and_roberta_int4_eval/`:

```text
fit input rows: 32172
target true rows: 1559
fit-ready rows: 1532
fitted groups: 79
```

Target-kind counts:

```text
geometry_only_no_loss_directional_mse                                                   30501
paper_directional_nmse:synthetic_A_true                                                  1482
geometry_only_not_target:lowbit_true_nmse:dequantized_effective_displacement_nmse_v1       84
paper_directional_mse:fd_true_mse                                                          77
missing_directional_mse_target                                                             28
```

## Consequences

1. Synthetic sharp/MIA fits remain valid because they used true directional
   oracle targets.
2. Geometry-only interval probes remain useful, but only as explanatory
   covariates or diagnostics.
3. The previous real-model sharp fit quality should not be cited as evidence
   that the sharp model fit paper-defined directional MSE, because those rows
   partly used effective-displacement NMSE.
4. The corrected real-model fit has much less real RoBERTa target coverage. In
   particular, many RoBERTa task/mode candidates disappear unless a true
   loss-level directional MSE probe exists.

Final explicit answer after the fix:

```text
当前拟合已经限制为论文定义的 directional MSE / normalized directional MSE；A_cross、A_interval、sigma_raw2 和 effective-displacement lowbit_true_nmse 不再作为 fit target。
```
