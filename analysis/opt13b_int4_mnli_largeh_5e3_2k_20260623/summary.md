# OPT-1.3B INT4 MNLI Dense Large-h Local Check

Local H100 run:

```bash
CUDA_VISIBLE_DEVICES=0 DATALOADER_SHUFFLE=True PYTHONUNBUFFERED=1 \
conda run -n mezo-env --no-capture-output \
python tools/train_opt13b_int4_dense_smoke.py \
  --output_root outputs/opt13b_int4_mnli_largeh_local_20260623 \
  --task mnli --task_path mezo_option --dataset_mode full \
  --num_train -1 --num_k 16 --model_id facebook/opt-1.3b \
  --h_values 5e-3 --h_labels large_5e-3 \
  --steps 2000 --lr 3e-7 --batch_size 16 --eval_batch_size 16 \
  --eval_samples 0 --eval_every 1000 --eval_max_batches 0 \
  --log_every 100 --diag_every 1000 --checkpoint_every 1000 \
  --max_seq_len 256 --bitwidth 4 --group_size 128 \
  --seed 16 --data_seed 16 --local_files_only --eval_at_start \
  --save_best_checkpoints --resume --skip_complete
```

Result:

| setting | h | steps | best eval acc | best step | last eval acc | last loss |
|---|---:|---:|---:|---:|---:|---:|
| local large-h check | 5e-3 | 2000 | 0.357820 | 0 | 0.351401 | 1.160497 |
| default reference | 1e-3 | 20000 | 0.390423 | 20000 | 0.390423 | 1.086005 |
| hstar_cont reference | 1.731e-4 | 20000 | 0.378604 | 2000 | 0.346001 | 1.103095 |

Eval trace for the large-h run:

| step | eval acc | eval loss |
|---:|---:|---:|
| 0 | 0.357820 | 1.247924 |
| 1000 | 0.340092 | 1.131058 |
| 2000 | 0.351401 | 1.160497 |

Conclusion:

`h=5e-3` is numerically stable for 2k steps, but it does not improve MNLI accuracy. The best evaluation remains the untrained step-0 model, and final 2k accuracy is below the initial value. This supports keeping `h=1e-3` as the conservative default for OPT-1.3B INT4 MNLI dense, rather than overriding upward.
