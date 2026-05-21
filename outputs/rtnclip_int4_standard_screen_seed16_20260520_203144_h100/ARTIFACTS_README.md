# INT4 Standard Screen Artifacts

This directory contains the RoBERTa-large / SST-5 INT4 G128 RTNClip standard dense MeZO preliminary h screen run on the local H100.

The Git-tracked artifact set intentionally excludes checkpoint tensor files named `state.pt`. Those files remain on the local filesystem under the same run directories and total about 49 GB. All summaries, configs, metrics, logs, diagnostics, checkpoint manifests, and resume commands are included in the no-state artifact archive:

`outputs/rtnclip_int4_standard_screen_seed16_20260520_203144_h100_no_statept.tar.gz`

Key result: `h=5e-4` reached 5k with `best_eval_acc = last_eval_acc = 0.3864168618266979`, while `h=5e-3` stayed numerically finite but degraded to `last_eval_acc = 0.25995316159250587`.
