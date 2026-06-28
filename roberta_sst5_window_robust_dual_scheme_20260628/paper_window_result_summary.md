# Paper Window Result Summary

## fp32

1. Empirical accuracy window: `[1e-07, 0.003]`.
2. Default h=1e-3 inside empirical window: `True`.
3. Tiny h=1e-5 inside empirical window: `True`.
4. Pure-theory plug-in status: `window`; W1: `[1e-08, 0.000758503]`.
5. Practical probe status: `primary_available`; primary: `[1e-07, 0.003]`; relaxed: `[1e-07, 0.004]`.
6. Smooth rho fit status: `window`; W1: `[8.74438e-09, 0.0100308]`.
7. Paper wording: `broad default-safe`.

## fp16

1. Empirical accuracy window: `[3e-06, 0.0015]`.
2. Default h=1e-3 inside empirical window: `True`.
3. Tiny h=1e-5 inside empirical window: `True`.
4. Pure-theory plug-in status: `window`; W1: `[3.00274e-05, 0.000702637]`.
5. Practical probe status: `primary_available`; primary: `[0.0001, 0.003]`; relaxed: `[0.0001, 0.005]`.
6. Smooth rho fit status: `no_stable_smooth_fit`; W1: `none`.
7. Paper wording: `broad default-safe`.

FP16 note: small h has a d_h=0 dead zone; this is not the same as no practical window.

## int8

1. Empirical accuracy window: `[3e-05, 0.001]`.
2. Default h=1e-3 inside empirical window: `True`.
3. Tiny h=1e-5 inside empirical window: `False`.
4. Pure-theory plug-in status: `no_window`; W1: `none`.
5. Practical probe status: `primary_available`; primary: `[0.001, 0.003]`; relaxed: `[0.001, 0.005]`.
6. Smooth rho fit status: `window`; W1: `[0.000121665, 0.0151067]`.
7. Paper wording: `broad default-safe`.

## int4

1. Empirical accuracy window: `[0.001, 0.001]`.
2. Default h=1e-3 inside empirical window: `True`.
3. Tiny h=1e-5 inside empirical window: `False`.
4. Pure-theory plug-in status: `no_window`; W1: `none`.
5. Practical probe status: `no_practical_probe_visible_window`; primary: `none`; relaxed: `none`.
6. Smooth rho fit status: `no_stable_smooth_fit`; W1: `none`.
7. Paper wording: `empirical plateau, no theory certificate`.

INT4 note: current true directional probe remains a boundary case; accuracy can be empirically default-safe without a stable smooth rho certificate.
