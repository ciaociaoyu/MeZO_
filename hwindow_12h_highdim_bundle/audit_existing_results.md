# Existing Result Audit

- Created: 2026-06-23T23:06:05
- Git commit: `4107bd6049c419c185b191d41f7884f33a6e3d2d`
- CSV files scanned: 2660
- Training-like rows indexed: 1629
- Probe-like rows indexed: 11064
- Full-like training rows (steps >= 20000): 408
- Medium/pilot training rows: 1323
- Interval metric source files detected: 7
- Loss/FD nMSE source files detected: 60

## Precision Sweeps / Training Coverage
model,task,precision,perturbation_mode,run_type,rows
roberta-large,sst-5,fp16,dense,pilot,308
roberta-large,sst-5,fp16,dense,medium,190
roberta-large,sst-5,int8,dense,pilot,130
nan,nan,int8,sparse,pilot,129
nan,nan,int8,sparse,medium,80
nan,nan,int4,dense,pilot,69
roberta-large,sst-5,int8,dense,full,44
nan,nan,int8,dense,pilot,34
nan,sst-5,int4,dense,full,30
roberta-large,sst-5,int4,dense,full,30
facebook/opt-1.3b,sst-5,int8,dense,pilot,28
nan,sst-5,int4,dense,pilot,25
facebook/opt-1.3b,sst-5,int4,dense,full,24
facebook/opt-1.3b,sst-2,int4,dense,full,24
facebook/opt-1.3b,trec,int4,dense,pilot,22
facebook/opt-1.3b,mnli,int4,dense,full,22
roberta-large,sst-5,fp32,dense,pilot,22
facebook/opt-1.3b,sst-5,int4,dense,pilot,22
nan,sst-5,nan,dense,full,21
facebook/opt-1.3b,trec,int4,dense,full,20
facebook/opt-1.3b,rte,int4,dense,full,20
nan,sst-2,int4,sparse_p0p1,pilot,20
facebook/opt-1.3b,rte,int4,dense,pilot,17
facebook/opt-1.3b,mnli,int4,dense,pilot,16
facebook/opt-1.3b,sst-2,int4,dense,pilot,14
roberta-large,sst-5,int4,dense,pilot,11
nan,sst-5,int8,dense,full,11
nan,nan,int8,dense,full,11
roberta-large,sst-5,bf16,dense,pilot,11
roberta-large,sst-5,int8,dense,medium,10

## Interval / Probe Coverage
model,task,precision,perturbation_mode,metric_type,rows
roberta-large,sst-5,int8,dense,interval,1941
roberta-large,sst-5,int8,sparse_p0p1,interval,1853
facebook/opt-1.3b,sst-5,int8,dense,interval,1729
facebook/opt-1.3b,sst-5,int8,sparse_p0p1,interval,1690
roberta-large,sst-5,int4,dense,interval,1537
roberta-large,sst-5,int4,sparse_p0p1,interval,1418
nan,sst-5,int4,sparse_p0p1,loss_or_fd,105
nan,sst-2,int4,sparse_p0p1,loss_or_fd,93
facebook/opt-1.3b,sst-5,int4,dense,loss_or_fd,90
facebook/opt-1.3b,sst-2,int4,dense,loss_or_fd,48
facebook/opt-1.3b,trec,int4,dense,loss_or_fd,45
facebook/opt-1.3b,rte,int4,dense,loss_or_fd,45
facebook/opt-1.3b,mnli,int4,dense,loss_or_fd,45
nan,mnli,int4,sparse_p0p1,loss_or_fd,42
nan,rte,int4,sparse_p0p1,loss_or_fd,42
nan,trec,int4,sparse_p0p1,loss_or_fd,41
roberta-large,sst-5,int8,dense,loss_or_fd,37
roberta-large,sst-5,int4,dense,loss_or_fd,37
facebook/opt-1.3b,mnli,int4,sparse_p0p01,loss_or_fd,27
facebook/opt-1.3b,trec,int4,sparse_p0p01,loss_or_fd,27
facebook/opt-1.3b,sst-5,int4,sparse_p0p01,loss_or_fd,27
facebook/opt-1.3b,rte,int4,sparse_p0p01,loss_or_fd,27
facebook/opt-1.3b,sst-2,int4,sparse_p0p01,loss_or_fd,27
roberta-large,sst-5,int8,sparse_p0p1,loss_or_fd,15
roberta-large,sst-5,int4,sparse_p0p1,loss_or_fd,15
facebook/opt-1.3b,sst-5,int8,dense,loss_or_fd,12
facebook/opt-1.3b,sst-5,int8,sparse_p0p1,loss_or_fd,12
facebook/opt-1.3b,nan,int4,dense,loss_or_fd,10
facebook/opt-1.3b,nan,int4,sparse_p0p01,loss_or_fd,9
facebook/opt-1.3b,nan,int4,sparse_p0p1,loss_or_fd,9

## Priority For This 12h Run
- New synthetic high-dimensional quantized-oracle benchmark.
- Real-model interval/probe aggregation from existing RoBERTa and OPT results.
- Targeted training table from existing full/medium/pilot logs; missing long training is listed in job list rather than fabricated.
