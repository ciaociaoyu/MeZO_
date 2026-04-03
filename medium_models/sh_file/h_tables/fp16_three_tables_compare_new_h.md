# FP16 Comparison of Computed h vs `1e-6` vs `1e-3`

Notes:
- `computed h` refers to the new two-point method `h_two_point`.
- `probe_mae` is the directional-derivative error.
- `dev_loss` and `dev_acc` are used below.
- The sweep did not run the exact `computed h` values, so the `@computed_h` columns are log-h interpolation estimates from neighboring sweep points.
- Red text marks the best value in each row: minimum for `probe_mae` and `dev_loss`, maximum for `dev_acc`.

## Probe MAE Table

| Task  | computed_h | probe_mae@`1e-6` | probe_mae@`1e-3` | probe_mae@computed_h                      |
| ----- | ---------- | ---------------- | ---------------- | ----------------------------------------- |
| SST-2 | `1.52e-4`  | `2621.839217`    | `1.650007`       | <span style="color:red">`0.863718`</span> |
| SST-5 | `2.322e-4` | `2723.675187`    | `2.489144`       | <span style="color:red">`1.074537`</span> |
| MNLI  | `2.614e-4` | `2397.036376`    | `1.717849`       | <span style="color:red">`0.631269`</span> |
| RTE   | `2.104e-4` | `1471.069503`    | `2.085303`       | <span style="color:red">`0.542843`</span> |

## Dev Loss Table

| Task  | computed_h | dev_loss@`1e-6` | dev_loss@`1e-3`                           | dev_loss@computed_h                       |
| ----- | ---------- | --------------- | ----------------------------------------- | ----------------------------------------- |
| SST-2 | `1.52e-4`  | `0.601987`      | <span style="color:red">`0.213370`</span> | `0.213709`                                |
| SST-5 | `2.322e-4` | `1.567158`      | <span style="color:red">`1.047110`</span> | `1.053525`                                |
| MNLI  | `2.614e-4` | `1.106385`      | `0.717478`                                | <span style="color:red">`0.710635`</span> |
| RTE   | `2.104e-4` | `0.681614`      | `0.526021`                                | <span style="color:red">`0.522060`</span> |

## Dev Acc Table

| Task  | computed_h | dev_acc@`1e-6` | dev_acc@`1e-3`                            | dev_acc@computed_h                        |
| ----- | ---------- | -------------- | ----------------------------------------- | ----------------------------------------- |
| SST-2 | `1.52e-4`  | `0.707350`     | <span style="color:red">`0.925909`</span> | `0.924884`                                |
| SST-5 | `2.322e-4` | `0.299766`     | <span style="color:red">`0.531616`</span> | `0.517837`                                |
| MNLI  | `2.614e-4` | `0.365495`     | `0.713445`                                | <span style="color:red">`0.715228`</span> |
| RTE   | `2.104e-4` | `0.582329`     | <span style="color:red">`0.779116`</span> | `0.769787`                                |

|      |      |      |      |      |      |
| --- | --- | --- | --- | --- | --- |
|      |      |      |      |      |      |
|      |      |      |      |      |      |
|      |      |      |      |      |      |
|      |      |      |      |      |      |
