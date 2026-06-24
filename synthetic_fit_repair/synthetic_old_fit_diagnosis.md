# Synthetic Old Fit Diagnosis

- Previous fit rows: 912
- Rows with alpha<=0, beta<=0, or invalid h_star: 912

Bad rows by metric:
metric,rows
A_interval_grad,456
A_true,456

Likely causes:
- Linear oracle rows naturally lack a locality tail, so beta and h_star are not meaningful.
- Ideal symmetric central differences on smooth oracles often produce a squared h^4 tail, not h^2.
- Some previous configs mixed clipping/saturation regimes into a single OLS fit.
- Linear-space OLS can be dominated by extreme small-h points; log/weighted fits are needed.

Global log-correlation between A_interval_grad and A_true in previous raw table: 0.8322
This suggests interval-aware terms are still useful even when alpha/beta envelope fits fail.
