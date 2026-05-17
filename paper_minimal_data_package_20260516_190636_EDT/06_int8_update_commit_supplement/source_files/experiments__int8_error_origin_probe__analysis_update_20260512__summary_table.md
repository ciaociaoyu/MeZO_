| run | backend | h | lr | sparse_rate | sparse_rescale | commit_mode | max_code_step | best_acc | final_acc | final_loss | active_frac | cos_intended_actual | norm_ratio | saturation_frac |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline_direct_int8 | direct_int8 | 0.003 | 1e-05 | 1 | none | round | 0 | 0.279859 | 0.279859 | 1.61217 | 0.96331 | 0.179645 | 5.56546 | 2.48028e-05 |
| residual_grid | residual_grid | 0.003 | 0.001 | 1 | none | round | 0 | 0.153396 | 0.153396 | 2.83751 | 0.971287 | 0.996209 | 0.985605 | 0.0183284 |
| sparse_residual_grid_p001 | residual_grid | 0.0003 | 0.0001 | 0.01 | inv_sqrt_p | round | 1 | 0.282201 | 0.282201 | 1.61856 | 0.00933652 | 0.755237 | 0.212237 | 2.67565e-05 |
