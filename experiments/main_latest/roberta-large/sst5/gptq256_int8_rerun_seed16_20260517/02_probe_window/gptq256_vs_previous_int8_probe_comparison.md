# GPTQ-256 vs Previous INT8 Summary

Exact GPTQ used: `False`.
Actual quantizer labels observed: `groupwise_int8_block256`.

This rerun should be interpreted as exact GPTQ only if the quantizer report says so. In the current code path, the expected fallback is `groupwise_int8_block256`.

## Probe Window

Best observed probe h by corr_fd_true: `0.0015` with corr `0.990373`.

Previous INT8 expectation: useful signal around h=2e-3 to 3e-3; too-small h distorted; h=1e-2 can look geometrically active but fail locality.

## Training