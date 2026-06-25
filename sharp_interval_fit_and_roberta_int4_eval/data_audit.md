# Data audit

- fit input rows: 32172
- fit-ready rows: 1643
- fitted groups: 94
- RoBERTa INT4 training rows indexed: 108
- candidate configs: 15
- missing selected training rows: 9

## Available A_true / nMSE sources
experiment_type
real_interval_geometry            30501
synthetic                          1482
training_summary_or_loss_probe      189

## RoBERTa INT4 training rows by task/mode/run_type
task     mode          run_type
mnli     dense         full         1
                       pilot        3
         prefix        full         3
         sparse_p0p1   full         4
                       pilot        2
rte      dense         full         1
                       pilot        3
         prefix        full         2
                       medium       1
         sparse_p0p1   full         4
                       pilot        2
sst-2    dense         full         3
                       medium       1
                       pilot        2
         prefix        full         3
         sparse_p0p1   full         8
                       pilot        2
sst-5    dense         full        11
                       pilot        4
         prefix        full         3
         sparse_p0p1   full         4
                       pilot        3
trec     dense         full         2
                       pilot        3
         prefix        full         3
         sparse_p0p1   full         4
                       pilot        2
unknown  dense         full         1
                       missing      6
                       pilot        1
         sparse_p0p01  missing      1
         sparse_p0p1   medium       2
                       missing     11
                       pilot        2
