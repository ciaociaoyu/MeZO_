# Paper Window Result Summary

## fp32

- Default h=1e-3 inside W1: True.
- Theoretical W1: [8.74e-09, 0.01].
- Empirical accuracy good set: [1e-07, 0.003].
- Interpretation: broad default-safe.
- Wording: treat this as a guardrail certificate/probe diagnostic, not an accuracy optimum.

## fp16

- Default h=1e-3 inside W1: None.
- Theoretical W1: no tau=1 certificate.
- Empirical accuracy good set: [3e-06, 0.0015].
- Interpretation: no stable fit.
- Wording: treat this as a guardrail certificate/probe diagnostic, not an accuracy optimum.

## int8

- Default h=1e-3 inside W1: True.
- Theoretical W1: [0.000122, 0.0151].
- Empirical accuracy good set: [3e-05, 0.001].
- Interpretation: broad default-safe.
- Wording: treat this as a guardrail certificate/probe diagnostic, not an accuracy optimum.

## int4

- Default h=1e-3 inside W1: None.
- Theoretical W1: no tau=1 certificate.
- Empirical accuracy good set: [0.001, 0.001].
- Interpretation: no stable fit.
- Wording: treat this as a guardrail certificate/probe diagnostic, not an accuracy optimum.
