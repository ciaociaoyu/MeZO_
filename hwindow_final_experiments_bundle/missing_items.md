# Final Missing Items

- `main.tex` is missing from the local checkout; Stage 6 paper-source replacement and PDF visual inspection are blocked.
- FP32/FP16/BF16 RoBERTa/SST-5 `Delta_eff`, `G`, and `L_loc` are not all available in one frozen-formula provenance table; these were not back-solved from sweeps.
- Some prefix INT4 rows remain single-seed; no new training was launched by this packaging script.
- OPT rows are treated as cross-architecture sanity checks, not direct benchmark reproduction.
