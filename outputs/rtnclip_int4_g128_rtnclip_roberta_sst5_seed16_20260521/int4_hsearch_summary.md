# INT4 RTNClip H-Search Summary

For low-bit settings, the canonical nMSE in this runner is `lowbit_true_nmse`, an alias of the dequantized effective-displacement nMSE `delta_visibility_nmse`: MSE of `Q_t(w_t+h u)-Q_t(w_t-h u)` against `2h u` after dequantization to floating point. Legacy `nMSE_fd_true`, if present, is reported only for reference and is not used for selecting h.

| h | status | steps | best_acc | last_acc | best_loss | last_loss | run_dir |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| 1e-5 | failed | 5485 | 0.25995316159250587 | 0.25995316159250587 | 3.076780225409836 | 51.313597775175644 | `/scratch/jy03364/MeZO_/outputs/rtnclip_int4_g128_rtnclip_roberta_sst5_seed16_20260521/int4_hsearch/int4_g128_rtnclip_k1_h1e-5_seed16_bs64_ckpt1k` |
| 1e-3 | complete | 20000 | 0.47892271662763464 | 0.47892271662763464 | 1.355313231850117 | 1.355313231850117 | `/scratch/jy03364/MeZO_/outputs/rtnclip_int4_g128_rtnclip_roberta_sst5_seed16_20260521/int4_hsearch/int4_g128_rtnclip_k1_h1e-3_seed16_bs64_ckpt1k` |
| 3e-5 | complete | 20000 | 0.28337236533957844 | 0.25995316159250587 | 1.5931073441159251 | 14.038842944964872 | `/scratch/jy03364/MeZO_/outputs/rtnclip_int4_g128_rtnclip_roberta_sst5_seed16_20260521/int4_hsearch/int4_g128_rtnclip_k1_h3e-5_seed16_bs64_ckpt1k` |
| 1p5e-3 | complete | 20000 | 0.4168618266978923 | 0.3747072599531616 | 1.5062070038056206 | 1.510467743706089 | `/scratch/jy03364/MeZO_/outputs/rtnclip_int4_g128_rtnclip_roberta_sst5_seed16_20260521/int4_hsearch/int4_g128_rtnclip_k1_h1p5e-3_seed16_bs64_ckpt1k` |
| 1e-4 | complete | 20000 | 0.3231850117096019 | 0.20140515222482436 | 1.5468292593676816 | 1.6361492791276346 | `/scratch/jy03364/MeZO_/outputs/rtnclip_int4_g128_rtnclip_roberta_sst5_seed16_20260521/int4_hsearch/int4_g128_rtnclip_k1_h1e-4_seed16_bs64_ckpt1k` |
| 2e-3 | complete | 20000 | 0.45550351288056207 | 0.4519906323185012 | 1.4521896040690867 | 1.4521896040690867 | `/scratch/jy03364/MeZO_/outputs/rtnclip_int4_g128_rtnclip_roberta_sst5_seed16_20260521/int4_hsearch/int4_g128_rtnclip_k1_h2e-3_seed16_bs64_ckpt1k` |
| 3e-4 | complete | 20000 | 0.45784543325526933 | 0.36533957845433257 | 1.363192055766979 | 1.455624725556206 | `/scratch/jy03364/MeZO_/outputs/rtnclip_int4_g128_rtnclip_roberta_sst5_seed16_20260521/int4_hsearch/int4_g128_rtnclip_k1_h3e-4_seed16_bs64_ckpt1k` |
| 3e-3 | complete | 20000 | 0.3430913348946136 | 0.32786885245901637 | 1.5639591261709602 | 1.5760689585772834 | `/scratch/jy03364/MeZO_/outputs/rtnclip_int4_g128_rtnclip_roberta_sst5_seed16_20260521/int4_hsearch/int4_g128_rtnclip_k1_h3e-3_seed16_bs64_ckpt1k` |
| 4e-3 | complete | 20000 | 0.3149882903981265 | 0.275175644028103 | 1.5574159287177987 | 1.5693999743852458 | `/scratch/jy03364/MeZO_/outputs/rtnclip_int4_g128_rtnclip_roberta_sst5_seed16_20260521/int4_hsearch/int4_g128_rtnclip_k1_h4e-3_seed16_bs64_ckpt1k` |
| 5e-3 | complete | 20000 | 0.27049180327868855 | 0.25995316159250587 | 1.6029507281908666 | 1.7209089578454333 | `/scratch/jy03364/MeZO_/outputs/rtnclip_int4_g128_rtnclip_roberta_sst5_seed16_20260521/int4_hsearch/int4_g128_rtnclip_k1_h5e-3_seed16_bs64_ckpt1k` |
| 1e-2 | complete | 20000 | 0.2576112412177986 | 0.24004683840749413 | 1.610170887002342 | 1.6612677473653397 | `/scratch/jy03364/MeZO_/outputs/rtnclip_int4_g128_rtnclip_roberta_sst5_seed16_20260521/int4_hsearch/int4_g128_rtnclip_k1_h1e-2_seed16_bs64_ckpt1k` |
