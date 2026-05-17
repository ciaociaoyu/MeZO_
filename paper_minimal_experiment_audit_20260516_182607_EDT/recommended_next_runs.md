# Recommended Next Runs: Minimal Paper Audit

Generated: 2026-05-16 18:26 EDT  
Scope: read-only audit of existing MeZO / RoBERTa-large / SST-5 low-precision ZO evidence. No training was run.

## Short Answer

Minimal paper currently possible: **conditional**.

The dense probe-window story is already strong enough for the main precision-aware perturbation-window claim. The training evidence is mixed: INT8 + FP16-master has good selected-candidate evidence at h=2e-3 and h=3e-3 with 3 seeds and 5k steps, but FP32/BF16 training validation is either only 300 steps or historical FP16/FP32 with older summaries. For a clean minimal paper, the main missing item is a small, modern FP32/BF16-or-FP16 training validation matrix.

Files scanned by extension filter: 4763  
Relevant result records inventoried: 162  
Archives found but not extracted: 10

## Answers To Required Questions

1. **Usable dense probe-window data for FP32/BF16/INT8?** Yes. `experiments/int8_update_sparse_plan/probe_window_h100_20260512/dense_probe_summary.csv` has all three precision modes and the requested h grid.
2. **Enough evidence that FP32 works at small h, BF16 needs larger h, INT8 needs around 1e-3 to 3e-3?** Yes for probe diagnostics. FP32 best row: h=1e-05, corr=0.9999997106604281, nMSE=6.666072230573788e-07, align=0.9999999972731268, norm_ratio=1.0000000791861667. BF16 best row: h=0.001, corr=0.9979581804430964, nMSE=0.004620757523848902, align=1.0000000099939217, norm_ratio=1.000000018702057. INT8 best row: h=0.003, corr=0.9375666196909768, nMSE=0.12639306022704247, align=0.9467247091642914, norm_ratio=1.0562734989199134.
3. **Self-consistency d_h vs d_h/2?** Not found. Mark P1: useful to strengthen locality, but current corr/nMSE against true derivative already gives a clean derivative-quality axis.
4. **FP32 training validation h=1e-5,1e-3,1e-2?** Partially. Modern 300-step rows exist; historical seed13/20k FP32 sweep exists. Clean modern 2k/5k best-vs-last matrix is missing.
5. **BF16/FP16 training validation h=1e-5,1e-3,1e-2?** Partially. Modern BF16 300-step rows exist. Historical FP16 seed13/20k and seed16/50k sweeps exist. Clean BF16 2k/5k matrix is missing if BF16 is the paper's low-precision training line.
6. **INT8-forward + FP16-master h=1e-3,2e-3,3e-3,1e-2?** Yes, partially strong. Seed0 2k exists for all anchors; h=2e-3 and h=3e-3 also have 3 seeds at 5k.
7. **Are h=2e-3 and h=3e-3 available with 3 seeds and 5k?** Yes: `experiments/int8_update_sparse_plan/next_round_window_sparse_20260516/dense_5k_runs.csv`.
8. **Direct INT8 baseline h=3e-3 showing update distortion?** Diagnostic only. A 2-step dryrun shows snapping distortion, but the requested 50-300 step baseline is missing.
9. **Residual-grid best config h=3e-3 seed0 2k/5k?** Yes for 2k appendix evidence: `results_packages/residual_improvements_local_h100_20260515_210216_summary.csv` contains `residual_topacc_active0p04_step500_promote_step2000`.
10. **Sparse auxiliary probe/training?** Probe yes. Sparse p=0.01 h_active=0.006/0.012 exists. Training is only a 500-step screen, not 2k/5k.
11. **Main paper results?** Dense FP32/BF16/INT8 probe table; INT8 FP16-master h=2e-3/3e-3 5k 3-seed table; cautiously, 2k INT8 anchors for h=1e-3 and h=1e-2.
12. **Appendix only?** Sparse probe/training, residual-grid, direct INT8 dryrun, historical FP16 MSE/accuracy analysis.
13. **Truly required before writing?** A clean training-validation decision: either run the modern FP32 plus BF16/FP16 three-h matrices, or explicitly write the paper as probe-first with training validation caveated as preliminary/historical.

## Minimal Prioritized Run List

| Priority | Experiment name | Precision | Model / Dataset | Update backend | Direction | h values | steps | seeds | why needed | expected outcome | P-level |
|---|---|---|---|---|---|---|---:|---|---|---|---|
| 1 | Clean FP32 training validation | FP32 | RoBERTa-large / SST-5 | standard MeZO | dense | 1e-5, 1e-3, 1e-2 | 2k or 5k | seed0 | Replaces 300-step/historical evidence with paper-clean best/last metrics. | FP32 small h and 1e-3 should be usable; 1e-2 should degrade by locality. | P0 |
| 2 | Clean BF16-or-FP16 training validation | BF16 if matching modern probe; FP16 if paper wording chooses FP16 | RoBERTa-large / SST-5 | standard MeZO | dense | 1e-5, 1e-3, 1e-2 | 2k or 5k | seed0 | Removes BF16-vs-FP16 mismatch in current evidence. | Low precision should prefer h around 1e-3 over too-small h. | P0 |
| 3 | Dense self-consistency probe | FP32, BF16, INT8 | RoBERTa-large / SST-5 | none | dense | representative small/window/large h, especially 1e-3, 3e-3, 1e-2 | no training | probe seeds only | Strengthens locality claim using d_h vs d_h/2. | h=1e-2 should fail consistency while visible geometrically. | P1 |
| 4 | Direct INT8 update baseline | INT8 | RoBERTa-large / SST-5 | direct_int8 | dense | 3e-3 | 100-300 | seed0 or seed16 | Replaces 2-step dryrun with requested supplement. | Low cos / inflated norm ratio should expose commit snapping. | P1 |
| 5 | Optional sparse training appendix | INT8 | RoBERTa-large / SST-5 | fp16_master | sparse p=0.01 | h_active=0.006, lr=1e-5 | 2k or 5k | seed0 | Only if sparse appendix needs a training curve. | Confirms sparse is auxiliary, not main. | P2 |

## Experiments Not To Run Now

- Do not run full 20k sweeps over all h values.
- Do not expand sparse into a main method section.
- Do not promote residual-grid into the central contribution.
- Do not rerun the full dense probe grid unless adding the missing self-consistency metric.
- Do not spend runs on unrelated OPT/MNLI/smoke settings for this minimal RoBERTa-large/SST-5 paper.
