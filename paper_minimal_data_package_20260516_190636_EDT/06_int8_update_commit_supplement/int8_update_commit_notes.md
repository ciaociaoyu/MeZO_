# INT8 Update-Commit Supplement Notes

This folder is a supplement / update-commit diagnostic. It is not the main perturbation-window contribution.

Use cases:

- Direct INT8 update demonstrates update snapping distortion if citing the h=3e-3 dryrun: active fraction is high, but cosine with intended update is low and actual/intended norm ratio is inflated.
- Residual-grid can mitigate some update-commit dead-zone behavior in short/promotion diagnostics.
- Residual-grid evidence should be appendix/supplement unless stronger long-run/multiseed evidence is added.

Caveats:

- The direct INT8 baseline currently available is a very short dryrun, not a paper-clean 100-300 step baseline.
- Some older residual-grid runs are pre-fix or diagnostic only; do not use them as final evidence without checking the source row.
- Keep the paper's main story as precision-aware perturbation/probe windows.
