# Main Mechanism Initial-h Baselines Smoke Summary

- Result root: `/scratch/jy03364/MeZO_/outputs/smoke_main_mechanism_initial_h`
- Expected max steps: `2`
- Completed: fixed_small_1e-5-fp32, fixed_small_1e-5-fp16, fixed_small_1e-5-int8, fd_eps13_raw-fp32, fd_eps13_raw-fp16, fd_eps13_raw-int8
- Failed: none
- Default h=1e-3 paths found in preflight: `38`
- Default h=1e-3 was not rerun by this smoke.
- Fixed small final h: `1e-5` for FP32/FP16/INT8.
- Raw FD FP32 final h: `0.004921565763652325`.
- Raw FD FP16 final h: `0.0992431640625`, intentionally out of window and uncapped.
- Raw FD INT8 final h: `0.0992431640625`, FP16 proxy, not principled, intentionally out of window and uncapped.

Raw outputs are under the ignored `outputs/` tree and are not intended for commit.
