# FP16 old seed13/20k sweep: MSE, stability, train acc, and update-scale notes

Date: 2026-05-16  
Project: MeZO / RoBERTa-large / SST-5 / low-precision zeroth-order fine-tuning  
Scope: Read-only analysis of existing experiment records. No new experiments were run.

## 1. Context from today's discussion

We started from the question:

- Do lower-MSE probe points generally have better acc/loss?
- More specifically, for FP32 and FP16 h sweeps, after convergence, do lower-MSE points have better average accuracy and stability?
- Then we focused on the FP16 old `seed13/20k` sweep, because it has full local `metrics_logs` and probe CSVs.
- Finally, we looked at whether `h=3e-5` has another special property beyond low MSE: whether the actual ZO update `lr * d * u` is comparable in scale to the perturbation `h * u`.

Main answer:

> In FP16 old `seed13/20k`, low-MSE points clearly learn better and are more stable than bad-MSE points. But the best point is not the minimum-MSE point. `h=3e-5` looks like a sweet spot where the finite-difference estimate is already reliable, while the update scale is still large enough.

## 2. Main evidence files

- FP32/FP16 old sweep root:
  `/Users/jichaoyu/Documents/GitHub/MeZO/medium_models/sh_file/sst5/bs32/h_precision_sweep/result/sst5-bs32-h-precision-sweep`

- FP16 old seed13 metrics:
  `/Users/jichaoyu/Documents/GitHub/MeZO/medium_models/sh_file/sst5/bs32/h_precision_sweep/result/sst5-bs32-h-precision-sweep/fp16/eps_*/seed13/metrics_logs/metrics_adaptiveH-0_cscale-0.csv`

- FP16 old seed13 probe:
  `/Users/jichaoyu/Documents/GitHub/MeZO/medium_models/sh_file/sst5/bs32/h_precision_sweep/result/sst5-bs32-h-precision-sweep/fp16/eps_*/seed13/zo_directional_probe.csv`

- Existing old sweep summary:
  `/Users/jichaoyu/Documents/GitHub/MeZO/medium_models/sh_file/sst5/bs32/h_precision_sweep/result/figures/h_precision_summary.csv`

- FP16 newer full 50k seed16 summary:
  `/Users/jichaoyu/Documents/GitHub/MeZO/medium_models/sh_file/sst5/bs32/h_precision_sweep_16/workspace/result/sst-5-bs32-full-fp16-h-sweep-seed16/summary.jsonl`

## 3. Method notes

- The old `metrics_adaptiveH-0_cscale-0.csv` files contain duplicate rows per `global_step`; each step appears 10 times.
- For train statistics, rows were deduplicated by `global_step`, keeping the last row for each step.
- "last100 train acc" means the mean train accuracy over the final 100 unique training steps.
- "last500 train acc" means the mean train accuracy over the final 500 unique training steps.
- "last5 eval acc" means the mean of the last 5 eval points.
- This sweep is single-seed (`seed13`) and uses `DATALOADER_SHUFFLE=False`, so minibatch train accuracy has a strong batch-order pattern. Use train-acc stability as a relative diagnostic, not as a final scientific claim.

## 4. High-level FP32 vs FP16 pattern

### FP32 old seed13/20k

FP32 has a broad usable h range. Once FD is accurate enough, MSE no longer predicts accuracy strongly.

| h | probe MSE | corr | last5 eval acc | last5 acc std | last10 eval acc | last5 eval loss |
|---|---:|---:|---:|---:|---:|---:|
| `1e-8` | 163.537 | 0.954980 | 0.4450 | 0.01696 | 0.42875 | 1.31639 |
| `3e-8` | 25.8216 | 0.992249 | **0.4725** | 0.03102 | 0.44875 | 1.31783 |
| `1e-7` | 2.33598 | 0.999327 | 0.4525 | 0.01225 | 0.43375 | 1.35715 |
| `3e-7` | 0.277733 | 0.999917 | 0.4600 | 0.01658 | **0.45750** | **1.29424** |
| `1e-6` | 0.027526 | 0.999992 | 0.4700 | **0.00612** | 0.45625 | 1.32356 |
| `3e-6` | 0.016780 | 0.999965 | 0.4425 | 0.02449 | 0.44250 | 1.32087 |
| `1e-5` | 0.090578 | 0.999220 | 0.4450 | 0.01275 | 0.43625 | 1.31352 |
| `3e-5` | **0.001987** | 0.999991 | 0.4425 | 0.01000 | 0.43625 | 1.31560 |
| `1e-4` | 0.040046 | 0.999928 | 0.4500 | 0.01118 | 0.43750 | 1.31380 |
| `3e-4` | 0.438942 | 0.999290 | 0.4500 | 0.01369 | 0.43875 | 1.31253 |
| `1e-3` | 5.18923 | 0.997866 | 0.4475 | 0.01458 | 0.44250 | 1.30587 |
| `3e-3` | 84.6882 | 0.969730 | 0.4000 | 0.02739 | 0.37875 | 1.42461 |
| `1e-2` | 1402.0 | 0.463044 | 0.2000 | 0.00000 | 0.20000 | 1.77282 |

Correlations:

| Relationship | Spearman rho |
|---|---:|
| MSE vs last5 eval acc | -0.1628 |
| MSE vs last5 eval loss | +0.3022 |

Interpretation:

- FP32 MSE helps rule out very bad h, especially `1e-2`.
- Among good h values, accuracy/loss are mostly flat and noisy.
- Minimum MSE at `3e-5` is not the best eval-accuracy point.

### FP16 old seed13/20k

FP16 shows a much sharper relation between FD quality and training behavior.

| h | probe MSE | corr | NaN loss frac | last500 train acc | last100 train acc | last100 train loss | last5 eval acc | last5 eval loss |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `1e-7` | 2.987e9 | -0.0447 | 0.982 | 0.332 | 0.330 | NaN | 0.200 | NaN |
| `3e-7` | 7.462e7 | 0.0112 | 0.843 | 0.332 | 0.330 | NaN | 0.200 | NaN |
| `1e-6` | 1.529e7 | 0.1189 | 0.000 | 0.249 | 0.195 | 15.2108 | 0.200 | 16.9124 |
| `3e-6` | 6973.87 | 0.6679 | 0.000 | 0.314 | 0.305 | 1.6510 | 0.1975 | 1.7599 |
| `1e-5` | 216.108 | 0.9448 | 0.000 | **0.689** | 0.675 | 0.9683 | 0.4225 | 1.3220 |
| `3e-5` | 30.170 | 0.9911 | 0.000 | 0.684 | **0.730** | **0.9040** | **0.4750** | 1.3110 |
| `1e-4` | 3.392 | 0.9973 | 0.000 | 0.681 | 0.719 | 0.9344 | 0.4600 | 1.3168 |
| `3e-4` | **0.917** | **0.9991** | 0.000 | 0.678 | 0.720 | 0.9386 | 0.4500 | 1.3156 |
| `1e-3` | 5.286 | 0.9978 | 0.000 | 0.648 | 0.663 | 0.9982 | 0.4475 | **1.3054** |
| `3e-3` | 80.731 | 0.9723 | 0.000 | 0.505 | 0.530 | 1.3210 | 0.4000 | 1.4245 |
| `1e-2` | 1401.51 | 0.4633 | 0.000 | 0.332 | 0.330 | 1.6847 | 0.2000 | 1.7729 |

Correlations:

| Relationship | Spearman rho |
|---|---:|
| MSE vs last100 train acc | -0.807 |
| MSE vs last500 train acc | -0.697 |
| MSE vs last100 train loss | +0.800 |
| MSE vs last500 train loss | +0.800 |
| MSE vs last5 eval acc | -0.833 |
| MSE vs last5 eval loss | +0.833 |

Interpretation:

- In FP16, low MSE strongly tracks better training dynamics.
- Bad tiny h produces NaN or near-random behavior.
- Bad large h also collapses to near-random behavior.
- The best region is not exactly the minimum-MSE point; it is a window around `3e-5` to `3e-4`.

## 5. Why `h=3e-5` looks special

`h=3e-5` is special in the old FP16 sweep because it is the best or near-best by several training-side criteria:

| Metric | Best / notable h |
|---|---|
| lowest probe MSE | `3e-4` |
| highest probe corr | `3e-4` |
| highest last100 train acc | `3e-5` |
| lowest last100 train loss | `3e-5` |
| highest last5 eval acc | `3e-5` |
| lowest last5 eval loss | `1e-3`, but `3e-5` is close |

So `3e-5` is not simply "the most accurate finite-difference point". It is better described as a point where:

- finite-difference quality is already high enough;
- sign agreement and correlation are good;
- the update is still large enough to move parameters effectively;
- training has not entered the too-large-h curvature/bias regime.

## 6. Update-scale hypothesis

For old MeZO, ignoring weight decay:

```text
perturbation:  theta + h * u
projected grad: d = (loss1 - loss2) / (2h)
update:        theta <- theta - lr * d * u
```

Therefore the scale ratio between update and perturbation is:

```text
||lr * d * u|| / ||h * u|| = lr * |d| / h
```

Relevant code:

- perturbation applies `z * zero_order_eps`:
  `/Users/jichaoyu/Documents/GitHub/MeZO/medium_models/src/trainer_副本.py:247`

- projected gradient is `(loss1 - loss2) / (2 * zero_order_eps)`:
  `/Users/jichaoyu/Documents/GitHub/MeZO/medium_models/src/trainer_副本.py:621`

- direct update is `param.data = param.data - learning_rate * (projected_grad * z + weight_decay * param.data)`:
  `/Users/jichaoyu/Documents/GitHub/MeZO/medium_models/src/trainer_副本.py:725`

The old FP16 probe CSV only stores per-probe-step aggregate `fd_mean`, not every direction's individual `d_i`. So the following table uses `lr * |fd_mean| / h` as a proxy, with `lr=1e-6`.

| h | MSE | corr | sign acc | mean abs fd | median `lr*abs(d)/h` | mean `lr*abs(d)/h` | last100 train acc | last100 train loss |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `3e-6` | 6973.87 | 0.6679 | 0.7388 | 23.217 | 4.768 | 7.739 | 0.305 | 1.651 |
| `1e-5` | 216.108 | 0.9448 | 0.8806 | 9.760 | **0.811** | **0.976** | 0.675 | 0.968 |
| `3e-5` | 30.170 | 0.9911 | 0.9613 | 9.111 | 0.215 | 0.304 | **0.730** | **0.904** |
| `1e-4` | 3.392 | 0.9973 | 0.9831 | 9.211 | 0.068 | 0.092 | 0.719 | 0.934 |
| `3e-4` | **0.917** | **0.9991** | **0.9956** | 9.147 | 0.0218 | 0.0305 | 0.720 | 0.939 |
| `1e-3` | 5.286 | 0.9978 | 0.9838 | 9.029 | 0.00675 | 0.00903 | 0.663 | 0.998 |
| `3e-3` | 80.731 | 0.9723 | 0.9363 | 7.847 | 0.00176 | 0.00262 | 0.530 | 1.321 |

### Interpretation of this table

The exact "update matches perturbation" point is closer to `1e-5`, because its mean ratio is near 1. But `1e-5` still has much worse FD quality than `3e-5`:

- `1e-5`: MSE 216, corr 0.945, sign 0.881
- `3e-5`: MSE 30, corr 0.991, sign 0.961

`3e-5` appears to be a better compromise:

```text
not too noisy:      FD corr is already high
not too small:      update/perturbation ratio is still ~0.2-0.3
not too biased:     h has not entered large-h degradation
```

This is the most plausible feature we found:

> `3e-5` is not the minimum-MSE point. It is the point where finite-difference visibility is high enough while update amplitude remains meaningful.

This is stronger than saying "lowest MSE is best", and it fits the observed train acc/loss better.

## 7. Candidate paper/internal phrasing

Careful version:

> In the FP16 old SST-5 sweep, training performance is not maximized at the minimum probe MSE. Instead, the best point (`h=3e-5`) lies at an intermediate point where finite-difference estimates are already accurate while the induced update scale remains appreciable relative to the perturbation scale. This suggests that probe MSE identifies the viable visibility window, but update-scale considerations may explain where inside the window training is strongest.

Short version:

> The best h is not the argmin-MSE h; it is the point where FD visibility and update amplitude are jointly favorable.

## 8. Caveats

- This is single seed (`seed13`).
- Train accuracy is minibatch accuracy and the dataloader is not shuffled.
- The old probe only stores aggregate `fd_mean`; it does not store per-direction `d_i`.
- Therefore `lr*|fd_mean|/h` is only a proxy for `mean(lr*|d_i|/h)`.
- We cannot fully reconstruct the exact training update norms from these old files.
- A stronger future diagnostic would log per step:
  - `projected_grad`
  - `mean_abs_projected_grad`
  - `||h*u||`
  - `||lr*d*u||`
  - `lr*|d|/h`
  - actual parameter delta norm
  - finite-difference corr/MSE at the same step

## 9. Bottom-line takeaways

1. FP32 old sweep: MSE mainly rules out bad h; within the good range, acc/loss are weakly tied to MSE.
2. FP16 old sweep: MSE strongly tracks whether training works at all.
3. FP16 `h=3e-5` is special because it gives the best observed train/eval behavior, despite not having the lowest MSE.
4. The likely explanation is a two-factor tradeoff:
   - FD quality improves as h enters the visible window.
   - update/perturbation scale shrinks as h grows.
5. `3e-5` sits near the balance point for this old FP16 setup.

