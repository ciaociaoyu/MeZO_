# Claim To Evidence Map

| claim | evidence_files | evidence_strength | main_text_or_appendix | caveats |
| --- | --- | --- | --- | --- |
| Dense FP32/BF16/INT8 have different h-windows. | `experiments/int8_update_sparse_plan/probe_window_h100_20260512/dense_probe_summary.csv`; probe-window archive listing | strong | main_text | Diagnostic h-window evidence, not final training accuracy. |
| Dense INT8 small h suffers from perturbation distortion. | `experiments/int8_update_sparse_plan/probe_window_h100_20260512/dense_probe_summary.csv` rows at h=1e-5,3e-5,1e-4,3e-4 | strong | main_text | Use alignment/norm_ratio/corr/nMSE; avoid overstating beyond probe. |
| Dense INT8 best probe correlation is around h=3e-3. | `experiments/int8_update_sparse_plan/probe_window_h100_20260512/dense_probe_summary.csv` | strong | main_text | Best corr row in this table is h=3e-3 for INT8. |
| BF16 best probe correlation is around h=1e-3. | `experiments/int8_update_sparse_plan/probe_window_h100_20260512/dense_probe_summary.csv` | strong | main_text | Best corr row in this table is h=1e-3 for BF16. |
| FP32 is stable at very small h. | `experiments/int8_update_sparse_plan/probe_window_h100_20260512/dense_probe_summary.csv` FP32 rows h=1e-5 and 3e-5 | strong | main_text | Probe stability only; check if manuscript needs training stability. |
| Sparse INT8 probe aligns better by h_active = h_raw / sqrt(p). | `experiments/int8_update_sparse_plan/probe_window_h100_20260512/sparse_probe_summary.csv`; sparse raw/active plots in archive | medium | main_text | Supported by structured sparse summary and plot availability; still diagnostic. |
| Sparse good h_active is around 0.006 to 0.012. | `experiments/int8_update_sparse_plan/probe_window_h100_20260512/sparse_probe_summary.csv` | medium | main_text | Best rows depend on p; cite as probe-window range, not training optimum. |
| 300-step training shows h matters but is not enough for final accuracy claims. | `experiments/int8_update_sparse_plan/probe_window_h100_20260512/window_training_summary.csv` | medium | appendix | Short validation; final_acc semantics are ambiguous because last/best are not separated. |
| Residual-grid supports update-commit distortion as a separate bottleneck. | `experiments/int8_update_sparse_plan/README.md`; residual consistency archive; residual local H100 package | medium | appendix | Post-fix evidence is stronger; pre-fix anomalies are diagnostic only. |
| Residual-grid should be secondary/appendix unless long-run evidence exists. | `results_packages/residual_local_h100_20260515_172944_essential.tar.gz`; residual summaries | medium | appendix | A 2k promoted residual run exists, but no 5k-20k multi-seed residual evidence was found. |
