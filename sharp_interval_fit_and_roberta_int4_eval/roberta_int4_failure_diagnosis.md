# RoBERTa INT4 failure diagnosis

Missing or underperforming h_sharp rows are not filled in with synthetic values.
Use `roberta_int4_missing_training_launch_manifest.csv` for exact missing training candidates.

## Clear sharp losses vs default
| task | mode | h_sharp | h_sharp_acc | default_acc | sharp_vs_default_delta |
| --- | --- | --- | --- | --- | --- |
| mnli | prefix | 1e-05 | 0.335905 | 0.39427 | -0.0583652 |
| sst-2 | prefix | 1e-05 | 0.552932 | 0.612027 | -0.0590943 |
| sst-5 | dense | 0.01 | 0.257611 | 0.478923 | -0.221311 |
| sst-5 | prefix | 1e-05 | 0.275176 | 0.319672 | -0.0444965 |
| sst-5 | sparse_p0p1 | 0.01 | 0.240047 | 0.481265 | -0.241218 |
| trec | sparse_p0p1 | 0.003 | 0.272894 | 0.787546 | -0.514652 |
