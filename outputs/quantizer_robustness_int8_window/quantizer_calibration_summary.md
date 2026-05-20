# Quantizer Calibration Summary

AWQ-style calibration uses activation RMS from a small training subset and activation-weighted weight MSE. RTNClip uses the existing unweighted clipping objective.

| model | quantizer | calibration_examples | activation_rms_modules | quantized_modules | objective | alpha_grid |
| --- | --- | ---: | ---: | ---: | --- | --- |
| roberta-large | rtnclip | 0 | 0 | 148 | unweighted_weight_mse_clip_search | `[1.0, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7]` |
| roberta-large | awq | 128 | 147 | 148 | activation_weighted_weight_mse | `[1.0, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5]` |
| opt-1.3b | rtnclip | 0 | 0 | 144 | unweighted_weight_mse_clip_search | `[1.0, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7]` |
| opt-1.3b | awq | 128 | 144 | 144 | activation_weighted_weight_mse | `[1.0, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5]` |
