# Recommended paper table

| Precision | Direction | Estimated window | Selected h | Default h valid? | Failure at small h | Failure at large h |
| --- | --- | --- | --- | --- | --- | --- |
| int8 | dense hist groupwise256 | [0.001, 0.004] | 0.002 | yes | yes | yes |
| int8 | sparse p=0.003 hist groupwise256 | [0.012, 0.024] | 0.024 | missing | yes | no |
| int8 | sparse p=0.01 hist groupwise256 | [0.003, 0.024] | 0.006 | missing | no | no |
| fp16 | dense | [3e-05, 0.004] | 0.001 | yes | no | yes |
| fp32 | dense | [3e-05, 0.004] | 0.001 | yes | no | yes |
| int4 | dense G128 RTNClip | none | NA | no | yes | yes |
| int8 | dense G128 RTNClip | none | NA | missing | yes | yes |
