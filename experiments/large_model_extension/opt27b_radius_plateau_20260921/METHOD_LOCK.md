# OPT-2.7B radius extension method lock

Two full-data tasks: SST-2 and RTE, official MeZO templates, candidate verbalizers,
mean-token likelihood classification CE. Training is a full-data extension, not
an exact original few-shot benchmark reproduction. No chat template or classifier head.

Dense MeZO q=1. All unique floating model parameters are perturbed. FP16 master,
FP32 Gaussian directions sampled identically across h and precision; each side
is generated from the restored center. FP16 forward or INT8/INT4 RTNClip G128
linear-weight forward. RTNClip grid is refreshed from center once per step and
shared across both signs. No adapters, residual state, or gradient training.
The established master update casts the direction to FP16 before add_.

48 predeclared full runs: two tasks x three precisions x six h x seed16;
SST-2 FP16 all six h repeated at seeds32/64. Full train split and full validation;
20k steps, effective BS16, microbatch2, eval500, lr3e-7. Dataset order and
Gaussian seeds are independent of h and exactly resumable from step index.
Training h={1e-5,1e-4,1e-3,3e-3,1e-2,3e-2}. Final dev accuracy is primary;
best dev accuracy and initial accuracy are separate. A chance-level flat curve
does not establish a learned plateau. Differences across paired seeds are retained.

Probe: common FP16 pretrained center promoted to FP32 for true clean backward,
4 disjoint fixed train batches x16 examples, 64 common FP32 Gaussian directions.
Calibration uses separate 16 directions on batch0, forward second difference
[F(w+2rho*u)-2F(w+rho*u)+F(w)]/rho^2 divided by ||u||^2.
choose_l_plateau selects clean L_q90 before audit. Delta_eff is parameter-weighted
RTNClip scale RMS/sqrt(6). G is direct clean gradient norm. No tail fitting.
h_ref=.5*sqrt(Delta_eff*G/(L*sqrt(d*(d+2)))); c_d=(d+4)/(d+1);
rho_th=c_d*((Delta_eff/(2*h))^2+(2*h*L*sqrt(d*(d+2))/G)^2).
tau={1,5,10}; no certificate if rho_min>tau. FP32/FP16 lack a uniform RTN
Delta, so plug-in theory is explicitly unavailable for these modes.

Audit saves true scalar nMSE, V_total, V_dir, V_h_dep, direct cross term and rho.
No geometry proxy is called true MSE. Quantized curves are at one common center
and paired batches/seeds. Numerical failures and non-U curves are retained.
Primary theory/audit figures use calibration batch0. Batches1-3 are separately
reported robustness checks; pooled four-batch metrics have a distinct CSV.
