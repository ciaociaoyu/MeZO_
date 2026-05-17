# GPTQ-256 Quantizer Report

Requested experiment label: GPTQ-256 INT8 rerun.

Exact GPTQ available in medium_models: `False`.

Actual quantizer used here: `groupwise_int8_block256` fallback, not exact GPTQ.

Bits: `8`; group/block size: `256`; zero point: `none_symmetric`.

Calibration: no Hessian/second-order calibration is run because exact GPTQ is not implemented in this code path.

Activation quantization: not added.

Storage/forward path: fake-quantized dequantized weights are written back to model parameters for the existing QuZO forward/probe path.

Quantized parameters: current `quantize_model_in_place(..., include_frozen=True)` quantizes all floating model parameters in the QuZO path, including LayerNorm, bias, and classifier parameters.

Representative tensors loaded from `roberta-large`.

Roundtrip dequant equality across checked tensors: `False`.

Median of per-tensor median scales: `0.00113576`.

| name | shape | groups | scale_min | scale_median | scale_max | rel_error | roundtrip_max_abs |
| --- | --- | --- | --- | --- | --- | --- | --- |
| roberta.embeddings.word_embeddings.weight | 50265x1024 | 201060 | 0.000424603 | 0.00225301 | 0.00530189 | 0.00488608 | 2.98023e-08 |
| roberta.encoder.layer.0.attention.self.query.weight | 1024x1024 | 4096 | 0.000485638 | 0.00090928 | 0.00239335 | 0.0073716 | 1.49012e-08 |
| roberta.encoder.layer.0.attention.self.value.weight | 1024x1024 | 4096 | 0.000427246 | 0.000772311 | 0.00200503 | 0.00719485 | 1.49012e-08 |
| roberta.encoder.layer.0.attention.output.dense.weight | 1024x1024 | 4096 | 0.000362606 | 0.000687246 | 0.0055095 | 0.00714743 | 1.49012e-08 |
| roberta.encoder.layer.0.intermediate.dense.weight | 4096x1024 | 16384 | 0.000545472 | 0.00127068 | 0.00437146 | 0.00746236 | 2.98023e-08 |
| roberta.encoder.layer.0.output.dense.weight | 1024x4096 | 16384 | 0.000682921 | 0.00123608 | 0.00787402 | 0.00729487 | 5.96046e-08 |
| roberta.encoder.layer.1.attention.self.query.weight | 1024x1024 | 4096 | 0.000668023 | 0.00128799 | 0.00372555 | 0.00782792 | 2.98023e-08 |
| roberta.encoder.layer.1.attention.self.value.weight | 1024x1024 | 4096 | 0.000315268 | 0.000782403 | 0.00208961 | 0.00714732 | 1.49012e-08 |
| roberta.encoder.layer.1.attention.output.dense.weight | 1024x1024 | 4096 | 0.000454159 | 0.000672828 | 0.0059901 | 0.00750115 | 1.49012e-08 |
| roberta.encoder.layer.1.intermediate.dense.weight | 4096x1024 | 16384 | 0.000519039 | 0.00134373 | 0.00438684 | 0.00710463 | 2.98023e-08 |
| roberta.encoder.layer.1.output.dense.weight | 1024x4096 | 16384 | 0.000757893 | 0.00117361 | 0.00787402 | 0.00775428 | 5.96046e-08 |
| roberta.encoder.layer.2.attention.self.query.weight | 1024x1024 | 4096 | 0.000642071 | 0.00123993 | 0.00358714 | 0.00747067 | 2.98023e-08 |
