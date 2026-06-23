# OPT-1.3B INT4 lowbit-L h-star check

| task | cleanL h* old | lowbitL h* | empirical min nMSE h | empirical corr+ min h | G lowbit | L clean32 | L lowbit | Llow/Lclean | lowbit h2 | best acc h=1e-3 | best acc matched old-hstar |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| sst-2 | 0.000505617 (0.0003) | 0.00713215 (0.01) | 0.003 | 0.003 | 88.5683 | 0.000334083 | 1.1261e-06 | 0.00337073 | 0.03 | 0.896789 | 0.876147 |
| sst-5 | 0.000654348 (0.001) | 0.000620109 (0.001) | 0.1 | 0.1 | 130.617 | 0.000561731 | 0.000219688 | 0.391091 | 0.001 | 0.462307 | 0.441417 |

## Notes
- `G` is low-bit FD abs median over h=1e-4/3e-4/1e-3 in both old and new rows.
- New run changes only L selector from clean32 to lowbit shared-grid second difference.
- Existing training comparison uses previous h values; lowbitL h values have not been trained yet.
