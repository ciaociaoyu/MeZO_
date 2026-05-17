# GPTQ-256 INT8 Rerun, Seed 16

This is a GPTQ-256 INT8 robustness rerun for the MeZO RoBERTa-large / SST-5 low-precision ZO project.

The intended comparison target is the previous INT8 quantizer result set. This is not the primary main experiment setting unless explicitly promoted later.

Current implementation note: exact GPTQ/Hessian calibration is not implemented in `medium_models`. Unless a later quantizer report states otherwise, runs in this folder use the honest fallback label `groupwise_int8_block256`, meaning symmetric group-wise INT8 fake quantization with block/group size 256.

Pilot contract for this rerun:

- seed = 16
- data_seed = 16
- dataset_mode = full
- dataloader_shuffle = True
- per_device_train_batch_size = 64
- gradient_accumulation_steps = 1
- model = roberta-large
- task = SST-5

Subdirectories follow the requested layout: quantizer checks, probe window, dense FP16-master, sparse FP16-master, direct INT8 diagnostic, residual-grid, summaries, plots, manifests, jobs, and logs.
