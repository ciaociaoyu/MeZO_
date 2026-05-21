# Main Mechanism Classical h Baselines Smoke Summary

- Result root: `/scratch/jy03364/MeZO_/outputs/smoke_main_mechanism_classical_h`
- Expected max steps: `2`
- Completed: fd_eps13-fp32, fd_eps13-fp16, fd_eps13-int8, spall_ck-fp32, spall_ck-fp16, spall_ck-int8
- Failed: none
- FD fp32 final h: `np.finfo(np.float32).eps ** (1/3)`
- FD fp16 raw h: `np.finfo(np.float16).eps ** (1/3)`; final h capped to `1e-2`
- FD int8 final h: capped stress `1e-2`, not principled
- Spall c_k: starts at `1e-3` and decays continuously with gamma `0.101`
- Early guard flags passed when available: `random_prediction_guard_enabled`, `zo_probe_health_guard_enabled`

Raw outputs are under the ignored `outputs/` tree and are not intended for commit.
