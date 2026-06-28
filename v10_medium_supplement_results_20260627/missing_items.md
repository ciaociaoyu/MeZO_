# Missing / Limited Items

- Phase 1 prefix visibility geometry is incomplete in existing summaries; true nMSE/corr rows are retained where available.
- Prefix RTE is not in required training because V10 audit marked it incomplete/not comparable.
- Sparse TREC is not in required training; V10 required sparse tasks are SST-5 and RTE.
- Existing low-bit seed16 runs are reused; new jobs fill seeds 32/64.
- Existing high-precision seed16 FP32 runs are reused; new jobs fill seeds 32/64/128/256.
- Exact direction stream matching across h is limited by existing runner internals; base train/data seeds are paired.
