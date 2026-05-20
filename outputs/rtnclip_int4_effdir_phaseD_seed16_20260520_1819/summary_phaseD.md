# Phase D INT4 Effective-Displacement Update Summary

Output root: `outputs/rtnclip_int4_effdir_phaseD_seed16_20260520_1819`
Scope: dense directions and updates over INT4-quantized `Linear.weight` tensors only.
Runs completed: 20 / 20
Best run: `int4_effdir_standard_h1e-3_step300` best_eval_acc=0.3302 last_eval_acc=0.3302

| variant | h | status | steps | best_acc | last_acc | last_loss | active | align | norm_ratio | active_u/u | upd/std | skip | clip |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| effdir_active | 1e-3 | complete | 300 | 0.2939 | 0.2834 | 1.571 | 0.09929 | 0.4028 | 1.922 | 0.4744 | 0.4744 | 0 | 0 |
| effdir_active | 2e-3 | complete | 300 | 0.2728 | 0.2728 | 1.592 | 0.2333 | 0.5896 | 1.538 | 0.6901 | 0.6901 | 0 | 0 |
| effdir_active | 3e-3 | complete | 300 | 0.2108 | 0.2108 | 1.584 | 0.3498 | 0.7152 | 1.323 | 0.8119 | 0.8119 | 0 | 0 |
| effdir_active | 5e-4 | complete | 300 | 0.3021 | 0.3021 | 1.56 | 0.03183 | 0.2511 | 2.044 | 0.2911 | 0.2911 | 0 | 0 |
| effdir_global | 1e-3 | complete | 1000 | 0.2939 | 0.2892 | 1.563 | 0.07412 | 0.3751 | 1.582 | 0.4356 | 1 | 0 | 0 |
| effdir_global | 1e-3 | complete | 300 | 0.3056 | 0.2939 | 1.569 | 0.08117 | 0.3847 | 1.678 | 0.4481 | 1 | 0 | 0 |
| effdir_global | 2e-3 | complete | 300 | 0.3126 | 0.3126 | 1.6 | 0.2298 | 0.5893 | 1.521 | 0.6879 | 1 | 0 | 0 |
| effdir_global | 3e-3 | complete | 300 | 0.2775 | 0.2717 | 1.585 | 0.3485 | 0.7154 | 1.32 | 0.8114 | 1 | 0 | 0 |
| effdir_global | 5e-4 | complete | 1000 | 0.2951 | 0.2787 | 1.557 | 0.01103 | 0.1727 | 1.092 | 0.1983 | 1 | 0 | 0 |
| effdir_global | 5e-4 | complete | 300 | 0.3091 | 0.3091 | 1.563 | 0.01291 | 0.1826 | 1.191 | 0.2099 | 1 | 0 | 0 |
| effdir_secant | 1e-3 | complete | 300 | 0.3056 | 0.3056 | 1.565 | 0.09977 | 0.4031 | 1.929 | 0.4749 | 0.5185 | 0 | 0.02333 |
| effdir_secant | 2e-3 | complete | 300 | 0.2775 | 0.2717 | 1.655 | 0.2339 | 0.5896 | 1.54 | 0.6904 | 0.6492 | 0 | 0.03333 |
| effdir_secant | 3e-3 | complete | 300 | 0.2857 | 0.2857 | 1.575 | 0.3504 | 0.7151 | 1.325 | 0.8121 | 0.7549 | 0 | 0.02667 |
| effdir_secant | 5e-4 | complete | 300 | 0.2916 | 0.2916 | 1.563 | 0.02608 | 0.2356 | 1.805 | 0.2721 | 0.5539 | 0 | 0.02 |
| standard | 1e-3 | complete | 1000 | 0.2635 | 0.2108 | 1.594 | 0.118 | 0.408 | 2.184 | 0.4923 | 1 | 0 | 0 |
| standard | 1e-3 | complete | 300 | 0.3302 | 0.3302 | 1.563 | 0.118 | 0.408 | 2.183 | 0.4923 | 1 | 0 | 0 |
| standard | 2e-3 | complete | 300 | 0.274 | 0.274 | 1.582 | 0.2421 | 0.5882 | 1.583 | 0.6944 | 1 | 0 | 0 |
| standard | 3e-3 | complete | 300 | 0.1511 | 0.1499 | 1.63 | 0.3539 | 0.7141 | 1.336 | 0.8133 | 1 | 0 | 0 |
| standard | 5e-4 | complete | 1000 | 0.3138 | 0.3138 | 1.561 | 0.05709 | 0.2824 | 3.042 | 0.3412 | 1 | 0 | 0 |
| standard | 5e-4 | complete | 300 | 0.2799 | 0.274 | 1.569 | 0.05711 | 0.2824 | 3.043 | 0.3412 | 1 | 0 | 0 |

## Questions

- Best standard dense INT4 h in this Phase D scope: `1e-3` with best_eval_acc=0.3302.
- Effdir improves small-h INT4 in this short run: no; best effdir `int4_effdir_effdir_global_h5e-4_step300` vs best small standard `int4_effdir_standard_h1e-3_step300`.
- Most stable/best scaling by short-run accuracy: `standard` at h `1e-3`.
- Recommended extension candidate: none from this short run.
