# Phase 1 Probe Diagnostics

This phase reuses existing fixed-checkpoint probe outputs. It does not run new full training.

- True directional nMSE is copied only from sources that explicitly recorded finite-difference-vs-reference probe metrics.
- Geometry-only fields are kept separate as active fraction, norm ratio, and visible-direction cosine.
- Prefix active/norm geometry is incomplete in existing summaries and is left blank rather than fabricated.
- High-precision dense probe summaries provide alignment/norm-ratio only; true nMSE/corr are unavailable there.

Rows: 193
