# Missing items

- Real-model sharp fitting is limited by available loss-level nMSE/probe grids; geometry-only rows cannot provide A_true.
- New full RoBERTa INT4 training is not fabricated. Missing selected h runs are listed in `roberta_int4_missing_training_launch_manifest.csv`.
- Dense multi-task default full rows are incomplete in the discovered outputs for some tasks; existing 2k rows are marked pilot.

- Figure generation failed: "None of [Index(['task', 'mode'], dtype='str')] are in the [columns]"
