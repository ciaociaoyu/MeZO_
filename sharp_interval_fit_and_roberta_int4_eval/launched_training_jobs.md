# Launched training jobs

## TREC dense INT4 sharp h=3e-3 full run

- job_id: 46426110
- completed_at: 2026-06-24T18:02:32.243470
- resource_request: `gpu:A100:1`, `mem=240G`, `time=12:00:00`, partition `gpu_p`
- status: complete
- steps_completed: 20000
- best_eval_acc: 0.2564102564102564 at step 17000
- last_eval_acc: 0.22344322344322345 at step 20000
- run_dir: `outputs/sharp_interval_roberta_int4_eval/int4_hsearch/dense/int4_dense_trec_sharp_h3e-3_seed16_full_bs64_step20k`
- copied_summary_dir: `sharp_interval_fit_and_roberta_int4_eval/new_training_trec_dense_sharp_h3e-3`
- reason for Slurm path: interactive allocation had `ulimit -m=60GB`; full RoBERTa-large initialization was killed before step 1, while 20-step smoke passed.

Final squeue snapshot:

```
completed_not_in_squeue
```
