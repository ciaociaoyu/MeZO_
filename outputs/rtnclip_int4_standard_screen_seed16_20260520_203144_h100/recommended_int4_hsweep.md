# Recommended INT4 Standard H Sweep

1. Does standard INT4 continue training beyond 1k? yes. `h=5e-4` reached 5k with best_eval_acc=last_eval_acc=0.3864168618266979.
2. Which h is most stable? `5e-4`.
3. Is h=5e-4 still best? yes. It is the only 5k run that improved through the extension.
4. Does h=1e-3 collapse after 1k again? no hard collapse was observed, but it drifted down from best_eval_acc=0.3126463700234192 to last_eval_acc=0.28337236533957844 by 2k, so it is not the preferred center.
5. Should we launch a 5k/10k narrow sweep? yes, use a narrow grid centered on `5e-4`.
6. Exact h grid to use: primary grid `3e-4, 5e-4, 7e-4, 1e-3`; include `2e-4` as a left-failure anchor if budget allows.

Official preliminary grid was: `2e-4, 3e-4, 5e-4, 7e-4, 1e-3, 1p5e-3, 2e-3`; optional anchors `3e-3, 5e-3` were also run because the H100 was free.
2k candidates: `5e-4, 5e-3, 1e-3, 3e-4`.
5k candidates: `5e-4, 5e-3`. `5e-3` remained numerically finite but degraded to last_eval_acc=0.25995316159250587 and should be treated as a large-h negative/diagnostic anchor, not a training-center candidate.
