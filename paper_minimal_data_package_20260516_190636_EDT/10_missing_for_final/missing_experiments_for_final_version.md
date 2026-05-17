# Missing Experiments For Final Clean Version

## P0

| experiment name | precision | model/dataset | update backend | h values | steps | seeds | why needed | blocks current package | blocks final paper submission |
|---|---|---|---|---|---:|---|---|---|---|
| Clean modern FP32 training validation | FP32 | RoBERTa-large / SST-5 | standard MeZO | 1e-5, 1e-3, 1e-2 | 2k or 5k | seed0 minimal | Replaces historical seed13 FP32 with clean best/last metrics. | no | yes if training validation is main-text required |
| Clean modern BF16 or FP16 training validation | BF16 or FP16, but label exactly | RoBERTa-large / SST-5 | standard MeZO | 1e-5, 1e-3, 1e-2 | 2k or 5k | seed0 minimal | Removes BF16-vs-FP16 mismatch and 300-step limitation. | no | yes for a clean low-precision training claim |

## P1

| experiment name | precision | model/dataset | update backend | h values | steps | seeds | why needed | blocks current package | blocks final paper submission |
|---|---|---|---|---|---:|---|---|---|---|
| d_h vs d_h/2 self-consistency diagnostic | FP32, BF16, INT8 | RoBERTa-large / SST-5 | none | representative small/window/large h, especially 1e-3, 3e-3, 1e-2 | no training | probe seeds only | Strengthens locality/nonlocal finite-difference claim. | no | no, but useful |
| Paper-clean direct INT8 baseline | INT8 | RoBERTa-large / SST-5 | direct_int8 | 3e-3 | 100-300 | seed0 or seed16 | Current direct baseline is only a 2-step diagnostic. | no | no for main story; yes if update supplement must be formal |
| Longer/symmetric INT8 anchors | INT8-forward | RoBERTa-large / SST-5 | fp16_master | 1e-3 and 1e-2 | 5k | seed0, optional seeds 1/2 | h=2e-3/3e-3 are strong; anchors are shorter. | no | optional/conditional |

## P2

| experiment name | precision | model/dataset | update backend | h values | steps | seeds | why needed | blocks current package | blocks final paper submission |
|---|---|---|---|---|---:|---|---|---|---|
| Sparse training appendix | INT8-forward | RoBERTa-large / SST-5 | fp16_master | p=0.01, h_active=0.006, lr=1e-5 | 2k or 5k | seed0 | Only if sparse appendix needs a training curve. | no | no |
| Residual-grid stronger supplement | INT8 | RoBERTa-large / SST-5 | residual_grid | h=3e-3 | 5k or 10k | seeds 0/1/2 | Only if residual-grid is promoted beyond diagnostic appendix. | no | no for main paper |
| Additional dataset/task | mixed | e.g. MNLI or SST-2 | matching main setup | selected h only | selected | selected | Generality check, not needed for minimal version. | no | no for minimal version |
