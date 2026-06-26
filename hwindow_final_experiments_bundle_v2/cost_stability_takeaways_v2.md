# Cost and Stability Takeaways V2

- Existing practical calibration records use forward probes for `G` and `L_loc`; no backward pass is recorded for the practical method rows summarized here.
- Existing configs use roughly 8-16 forward probes for `G` and 9 probes for `L_loc` when those counts are logged.
- Runtime is included only in `probe_cost_stability_v2.csv` when source logs provide it; peak memory is not claimed without measured values.
- There are not enough repeat-probe groups to claim strong stability of `h_ref`; this remains a limitation.
