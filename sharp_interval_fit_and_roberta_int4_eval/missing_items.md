# Missing items

- Real-model sharp fitting is limited by available loss-level nMSE/probe grids; geometry-only rows cannot provide A_true.
- New full RoBERTa INT4 training is not fabricated. Missing selected h runs are listed in `roberta_int4_missing_training_launch_manifest.csv`.
- Dense multi-task default full rows are incomplete in the discovered outputs for some tasks; existing 2k rows are marked pilot.

## Missing selected training rows
| task | mode | policy | h_value | reason |
| --- | --- | --- | --- | --- |
| mnli | sparse_p0p1 | sharp | 0.01 | no_existing_training_at_selected_h |
| mnli | sparse_p0p1 | sharp_cons | 0.01 | no_existing_training_at_selected_h |
| mnli | sparse_p0p1 | safe | 0.01 | no_existing_training_at_selected_h |
| rte | sparse_p0p1 | sharp | 0.01 | no_existing_training_at_selected_h |
| rte | sparse_p0p1 | sharp_cons | 0.01 | no_existing_training_at_selected_h |
| rte | sparse_p0p1 | safe | 0.01 | no_existing_training_at_selected_h |
| sst-2 | sparse_p0p1 | sharp | 0.003 | no_existing_training_at_selected_h |
| sst-2 | sparse_p0p1 | sharp_cons | 0.003 | no_existing_training_at_selected_h |
| sst-2 | sparse_p0p1 | safe | 0.003 | no_existing_training_at_selected_h |
