# Lowbit L h-star check

Source: existing 20k runs. Matching is by dataset column.

| task | dir | default h | default best | orig hstar h | orig hstar best | cleanL h | cleanL best | lowbitL h | lowbitL best | Llow/Lclean | lowbit h2 | reliability |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| mnli | dense |  |  |  |  | 0.00131271 | 0.472727 | 5.09807e-05 | 0.344792 | 284.45 | 3e-05 |  |
| rte | dense |  |  |  |  | 0.00170563 | 0.60241 | 9.13125e-05 | 0.566265 | 213.168 | 3e-05 |  |
| sst-2 | dense | 0.001 | 0.899035 | 0.0015 | 0.654937 | 0.00148314 | 0.80876 | 0.000621479 | 0.88196 | 4.84891 | 0.001 | nan |
| sst-5 | dense |  |  |  |  | 0.00155094 | 0.42623 | 0.000614543 | 0.476581 | 5.94926 | 0.001 |  |
| trec | dense |  |  |  |  | 0.00069759 | 0.827839 | 0.00162425 | 0.714286 | 1.21288 | 0.002 |  |
| mnli | sparse |  |  |  |  |  |  | 0.000437325 | 0.576802 | 5.50108 | 0.001 |  |
| rte | sparse |  |  |  |  |  |  | 0.000359155 | 0.618474 | 10.7582 | 0.0003 |  |
| sst-2 | sparse | 0.001 | 0.557832 | 0.002 | 0.442168 | 0.000458315 | 0.89562 | 0.00104401 | 0.891017 | 1.80978 | 0.001 | deprecated_untrusted_legacy_abs_importance_sparse |
| sst-5 | sparse |  |  |  |  |  |  | 0.0013665 | 0.295082 | 4.23683 | 0.002 |  |
| trec | sparse |  |  |  |  |  |  | 0.000682558 | 0.838828 | 6.45437 | 0.001 |  |

## Aggregate
- dense: lowbitL beats cleanL on 2/5; beats h=1e-3 default on 0/1; mean best default/cleanL/lowbitL = 0.8990/0.6276/0.5968.
- sparse: lowbitL beats cleanL on 0/1; beats h=1e-3 default on 1/1; mean best default/cleanL/lowbitL = 0.5578/0.8956/0.6440.
