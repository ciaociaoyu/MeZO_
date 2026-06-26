# RoBERTa Policy Mapping Audit

Rows are selected by fixed policy mapping, not by accuracy.

- `fixed-small`: training row with `h=1e-5` / `fixed_small`.
- `MeZO default`: training row with `h=1e-3` / `mezo_default`.
- `frozen reference`: sparse p=0.1 uses existing `hstar_lowbitL`; prefix INT4 uses existing `hstar_cleanGL`, because those are the pre-existing analytical-reference rows for those modes.

## mnli / prefix
- MeZO default: h=0.001, frozen/reference hstar_cont=0.105987, log-distance=2.02525, source=outputs/int4_sparse_prefix_seedfixed_int4fd_20k_20260523_171841/int4_hsearch_summary.csv; selected by policy name/h, not accuracy.
- fixed-small: h=1.000000e-05, frozen/reference hstar_cont=0.105987, log-distance=4.02525, source=outputs/int4_sparse_prefix_seedfixed_int4fd_20k_20260523_171841/int4_hsearch_summary.csv; selected by policy name/h, not accuracy.
- frozen reference: h=0.105987, frozen/reference hstar_cont=0.105987, log-distance=0, source=outputs/int4_sparse_prefix_seedfixed_int4fd_20k_20260523_171841/int4_hsearch_summary.csv; selected by policy name/h, not accuracy.

## mnli / sparse_p0p1
- MeZO default: h=0.001, frozen/reference hstar_cont=8.531356e-04, log-distance=0.0689819, source=outputs/int4_sparse_prefix_seedfixed_int4fd_20k_20260523_171841/int4_hsearch_summary.csv; selected by policy name/h, not accuracy.
- fixed-small: h=1.000000e-05, frozen/reference hstar_cont=8.531356e-04, log-distance=1.93102, source=outputs/int4_sparse_prefix_seedfixed_int4fd_20k_20260523_171841/int4_hsearch_summary.csv; selected by policy name/h, not accuracy.
- frozen reference: h=4.373247e-04, frozen/reference hstar_cont=4.373247e-04, log-distance=0, source=outputs/int4_sparse_prefix_seedfixed_int4fd_20k_20260523_171841/int4_hsearch_summary.csv; selected by policy name/h, not accuracy.

## rte / sparse_p0p1
- MeZO default: h=0.001, frozen/reference hstar_cont=4.352529e-04, log-distance=0.361258, source=outputs/int4_sparse_prefix_seedfixed_int4fd_20k_20260523_171841/int4_hsearch_summary.csv; selected by policy name/h, not accuracy.
- fixed-small: h=1.000000e-05, frozen/reference hstar_cont=4.352529e-04, log-distance=1.63874, source=outputs/int4_sparse_prefix_seedfixed_int4fd_20k_20260523_171841/int4_hsearch_summary.csv; selected by policy name/h, not accuracy.
- frozen reference: h=3.591547e-04, frozen/reference hstar_cont=3.591547e-04, log-distance=0, source=outputs/int4_sparse_prefix_seedfixed_int4fd_20k_20260523_171841/int4_hsearch_summary.csv; selected by policy name/h, not accuracy.

## sst-2 / prefix
- MeZO default: h=0.001, frozen/reference hstar_cont=0.088642, log-distance=1.94764, source=outputs/int4_sparse_prefix_seedfixed_int4fd_20k_20260523_171841/int4_hsearch_summary.csv; selected by policy name/h, not accuracy.
- fixed-small: h=1.000000e-05, frozen/reference hstar_cont=0.088642, log-distance=3.94764, source=outputs/int4_sparse_prefix_seedfixed_int4fd_20k_20260523_171841/int4_hsearch_summary.csv; selected by policy name/h, not accuracy.
- frozen reference: h=0.088642, frozen/reference hstar_cont=0.088642, log-distance=0, source=outputs/int4_sparse_prefix_seedfixed_int4fd_20k_20260523_171841/int4_hsearch_summary.csv; selected by policy name/h, not accuracy.

## sst-2 / sparse_p0p1
- MeZO default: h=0.001, frozen/reference hstar_cont=4.583148e-04, log-distance=0.338836, source=outputs/int4_sparse_prefix_seedfixed_int4fd_20k_20260523_171841/int4_hsearch_summary.csv; selected by policy name/h, not accuracy.
- fixed-small: h=1.000000e-05, frozen/reference hstar_cont=4.583148e-04, log-distance=1.66116, source=outputs/int4_sparse_prefix_seedfixed_int4fd_20k_20260523_171841/int4_hsearch_summary.csv; selected by policy name/h, not accuracy.
- frozen reference: h=0.00104401, frozen/reference hstar_cont=0.00104401, log-distance=0, source=outputs/int4_sparse_prefix_seedfixed_int4fd_20k_20260523_171841/int4_hsearch_summary.csv; selected by policy name/h, not accuracy.

## sst-5 / prefix
- MeZO default: h=0.001, frozen/reference hstar_cont=0.080472, log-distance=1.90564, source=outputs/int4_sparse_prefix_seedfixed_int4fd_20k_20260523_171841/int4_hsearch_summary.csv; selected by policy name/h, not accuracy.
- fixed-small: h=1.000000e-05, frozen/reference hstar_cont=0.080472, log-distance=3.90564, source=outputs/int4_sparse_prefix_seedfixed_int4fd_20k_20260523_171841/int4_hsearch_summary.csv; selected by policy name/h, not accuracy.
- frozen reference: h=0.080472, frozen/reference hstar_cont=0.080472, log-distance=0, source=outputs/int4_sparse_prefix_seedfixed_int4fd_20k_20260523_171841/int4_hsearch_summary.csv; selected by policy name/h, not accuracy.

## sst-5 / sparse_p0p1
- MeZO default: h=0.001, frozen/reference hstar_cont=0.00133886, log-distance=0.126734, source=outputs/int4_sparse_prefix_seedfixed_int4fd_20k_20260523_171841/int4_hsearch_summary.csv; selected by policy name/h, not accuracy.
- fixed-small: h=1.000000e-05, frozen/reference hstar_cont=0.00133886, log-distance=2.12673, source=outputs/int4_sparse_prefix_seedfixed_int4fd_20k_20260523_171841/int4_hsearch_summary.csv; selected by policy name/h, not accuracy.
- frozen reference: h=0.0013665, frozen/reference hstar_cont=0.0013665, log-distance=0, source=outputs/int4_sparse_prefix_seedfixed_int4fd_20k_20260523_171841/int4_hsearch_summary.csv; selected by policy name/h, not accuracy.

## trec / prefix
- MeZO default: h=0.001, frozen/reference hstar_cont=0.0949896, log-distance=1.97768, source=outputs/int4_sparse_prefix_seedfixed_int4fd_20k_20260523_171841/int4_hsearch_summary.csv; selected by policy name/h, not accuracy.
- fixed-small: h=1.000000e-05, frozen/reference hstar_cont=0.0949896, log-distance=3.97768, source=outputs/int4_sparse_prefix_seedfixed_int4fd_20k_20260523_171841/int4_hsearch_summary.csv; selected by policy name/h, not accuracy.
- frozen reference: h=0.0949896, frozen/reference hstar_cont=0.0949896, log-distance=0, source=outputs/int4_sparse_prefix_seedfixed_int4fd_20k_20260523_171841/int4_hsearch_summary.csv; selected by policy name/h, not accuracy.

## Incomplete groups
- rte/prefix: missing fixed-small; excluded from main table.
- trec/sparse_p0p1: missing fixed-small; excluded from main table.
