# Sparse Auxiliary Notes

Sparse is auxiliary only. Do not present sparse ZO as the main contribution.

What this folder supports:

- Sparse INT8 probe evidence exists, including p=0.01 at h_active 0.006 and 0.012.
- h_active = h_raw / sqrt(p) is useful for interpreting sparse directions.
- Sparse training evidence is currently a 500-step screen, not a long validation.

Use sparse as appendix/context showing the window view can extend beyond dense directions. Keep the central story on dense FP32 -> BF16/FP16 -> INT8 perturbation windows.
