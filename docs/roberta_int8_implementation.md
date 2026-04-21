# RoBERTa Int8 Implementation

This document describes the current `medium_models` implementation for
`roberta-large` with QuZO low-bit probing, i.e. `--zo_quantization int8` or
`--zo_quantization int4`.

Effective date of the current behavior:

- `2026-04-21`

Scope:

- code path: `medium_models/`
- model family: RoBERTa-style masked-LM prompt training
- precisions covered here: QuZO `int8` and `int4`
- not covered here: the legacy `fp16/quzo16` path, `large_models/`, or
  `load_int8`

## 1. Parameter Representation

The current RoBERTa QuZO low-bit path does **not** store `nn.Parameter` tensors
as true `torch.int8` tensors.

Instead:

- model parameters are loaded as floating-point tensors
- `quantize_model_in_place(...)` snaps each parameter tensor onto an `int8` or
  `int4` quantization grid
- the snapped value is written back as a float tensor

So the runtime parameter state is:

- "float storage on a low-bit grid"

not:

- "true integer-code storage with separate scales"

This is the current practical design because the existing Hugging Face /
PyTorch training path expects trainable parameters to remain floating-point.

## 2. Low-Bit Probe Semantics

For RoBERTa QuZO `int8/int4`, the finite-difference probe path now uses one
canonical implementation only.

The probe state is defined as:

- `w_probe(scale) = w_base + Q(scale * eps * u_raw)`

where:

- `w_base` is the already-snapped floating-point base parameter tensor at the
  start of the step
- `u_raw` is the raw Gaussian direction reconstructed from the saved per-tensor
  `gaussian_seed`
- `Q(...)` is the low-bit perturbation quantizer using the perturbation seed
  and the perturbation tensor's own scale
- `scale` is one of `{+1, -1, 0}` for the two-sided probe transition

The key implementation rule is:

- **do not resnap the whole parameter tensor after adding the probe
  perturbation**

In other words, the current low-bit probe path is:

- `w_base + Q(scale * eps * u_raw)`

and **not**:

- `Q_all(w_base + Q(scale * eps * u_raw))`

This change removes the old full-parameter post-perturbation resnap that was
distorting the two-point probe state.

## 3. Transition Rule During `+1 / -2 / +1`

`medium_models` uses incremental perturbation calls during ZO probing, so the
implementation keeps a small per-step probe-scale state for low-bit QuZO.

The transition is handled as:

- previous target delta: `Q(prev_scale * eps * u_raw)`
- next target delta: `Q(next_scale * eps * u_raw)`
- applied in-place increment: `Q(next_scale * eps * u_raw) - Q(prev_scale * eps * u_raw)`

This ensures that the standard probe sequence:

- `+1`
- `-2`
- `+1`

lands on:

- `+eps`
- `-eps`
- back to `0`

relative to the same snapped base weights, instead of accumulating extra
projection error from repeated full-parameter resnaps.

## 4. Update Path

The low-bit probe change above only fixes the finite-difference state used for
`loss1 / loss2`.

The parameter update path still uses the QuZO update direction `u2` and then
projects the updated parameter tensor back onto the low-bit grid. Concretely:

- probe direction for `loss1/loss2`: raw Gaussian `u_raw` -> quantized target
  delta `Q(eps * u_raw)`
- update direction for the parameter step: quantized `u2`
- post-update parameter state: snapped back onto the low-bit parameter grid

So the current low-bit RoBERTa path is:

- clean low-bit probe
- quantized update direction
- snapped-float parameter state after updates

## 5. Historical Note

Results produced before `2026-04-21` may have used the legacy low-bit probe
implementation that resnapped the whole parameter tensor after applying the
probe perturbation.

Those artifacts should be treated as:

- historical debugging data
- not directly comparable to the current canonical RoBERTa QuZO `int8/int4`
  implementation unless explicitly labeled

## 6. Source Pointers

Current implementation anchors:

- low-bit parameter snapping:
  - [medium_models/src/quzo.py](/scratch/jy03364/MeZO_/medium_models/src/quzo.py)
- trainer probe/update logic:
  - [medium_models/src/trainer.py](/scratch/jy03364/MeZO_/medium_models/src/trainer.py)
- training argument / run metadata wiring:
  - [medium_models/run.py](/scratch/jy03364/MeZO_/medium_models/run.py)
