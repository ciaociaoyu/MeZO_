# Data Dictionary

- `h`: finite-difference perturbation radius.
- `true_directional_nmse`: normalized `E[(d_Q-grad^T u)^2]/E[(grad^T u)^2]`.
- `directional_correlation`: correlation between `d_Q` and `grad^T u`.
- `interval_geometry_error`: displacement geometry proxy, not true MSE.
- `crossing_active_fraction`: fraction of coordinates whose quantized code changes.
- `displacement_alignment`: cosine between `Delta_Q` and `2hu`.
- `displacement_norm_ratio`: `||Delta_Q|| / ||2hu||`.
- `best_dev_acc`: best development accuracy in a training run.
- `h_ref_current`: radius recomputed from the frozen theory.
- `legacy_hstar`: historical radius from earlier scripts; not automatically equal to current frozen h_ref.
- `training_h`: the actual h used by a training run.
