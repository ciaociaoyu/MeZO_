# Cost and Stability Takeaways

- The practical calibration rows found here use forward evaluations for `G` and `L_loc`; no backward pass is required in the recorded practical method rows.
- Runtime and peak-memory fields are only reported when present in source logs.
- Variation of `log10(h_ref)` can be computed from `probe_cost_stability.csv`; additional repeat probes were not launched.
