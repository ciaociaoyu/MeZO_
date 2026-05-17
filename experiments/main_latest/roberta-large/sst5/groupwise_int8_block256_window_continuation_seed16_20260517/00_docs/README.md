# Groupwise INT8 Block-256 Window Continuation

This is NOT exact GPTQ.

This is group-wise INT8 block-256 quantization.

It is a quantizer ablation for the INT8 perturbation window.

It tests whether the wider window under block/group scales is reproducible.

It tests sparse-rate effects on the window under groupwise_int8_block256.

This continuation uses `seed=16`, `data_seed=16`, `dataset_mode=full`, `dataloader_shuffle=True`, `per_device_train_batch_size=64`, and `gradient_accumulation_steps=1` for the RoBERTa-large / SST-5 low-precision ZO setting unless a run log explicitly says otherwise.

