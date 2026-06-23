# GPTQ-Initialized Low-Bit Update Probe

This mini-repo tests whether a GPTQ-initialized low-bit OPT model can accept an
effective low-bit-only weight update from a real backprop gradient.

The first version intentionally does **not** use zeroth-order perturbations,
LoRA/adapters, full-precision master weights for the updated tensor, or
residual/error-feedback buffers. The only high-precision object used during the
update experiment is the gradient used to compute an intended SGD update. The
committed parameter state is represented as integer codes plus fixed or
recomputed group scales.

## What Is Being Tested

For a selected OPT Linear weight, the experiment:

1. Quantizes the model with Hugging Face `GPTQConfig` when a compatible GPTQ
   backend is installed.
2. Reads the dequantized GPTQ weight for a target layer.
3. Builds a surrogate explicit groupwise signed integer lattice from that
   GPTQ-dequantized weight.
4. Computes a real cross-entropy gradient by backpropagation.
5. Applies low-bit commit rules directly to the lattice.
6. Replaces the layer with the dequantized committed low-bit weight.
7. Measures same-batch and heldout loss changes.

The required path is labeled:

`surrogate explicit lattice initialized from GPTQ-dequantized weights`

It is **not** exact mutation of packed GPTQ internals. The optional
`--backend exact_gptq_packed` currently reports a skip unless an exact
unpack/mutate/repack path is added and verified.

## Installation

```bash
cd lowbit_update_experiment
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On the existing cluster, using the project conda environment is also fine:

```bash
cd /scratch/jy03364/MeZO_/lowbit_update_experiment
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate mezo-mistral
pip install -r requirements.txt
```

## Unit Tests

```bash
cd lowbit_update_experiment
PYTHONPATH=. pytest -q
```

## Smoke Test

```bash
python scripts/run_lowbit_update_experiment.py \
  --model facebook/opt-125m \
  --smoke \
  --bits 4 \
  --group_size 128 \
  --seq_len 256 \
  --calib_samples 16 \
  --eval_samples 8 \
  --target_layers last_mlp \
  --num_steps 1 \
  --output_dir results/smoke_opt125m_b4
```

If GPTQ dependencies are not installed, the script fails clearly by default. For
debugging the lattice/update code without GPTQ, add:

```bash
  --allow_surrogate_without_gptq --allow_synthetic_data
```

Do not use that fallback for paper claims.

## Full OPT-1.3B Run

```bash
python scripts/run_lowbit_update_experiment.py \
  --model facebook/opt-1.3b \
  --bits 4 \
  --group_size 128 \
  --seq_len 512 \
  --calib_samples 128 \
  --eval_samples 64 \
  --target_layers last_mlp \
  --num_steps 1 \
  --relative_update_norms 1e-6 3e-6 1e-5 3e-5 1e-4 3e-4 1e-3 3e-3 \
  --update_rules nearest_requant_fixed_grid stochastic_round_fixed_grid topk_code_flip topk_code_flip_plus_stochastic_tail dense_stochastic_code_flip \
  --k_fracs 1e-5 3e-5 1e-4 3e-4 1e-3 3e-3 1e-2 \
  --output_dir results/opt13b_b4_lowbit_update
```

Summarize:

```bash
python scripts/summarize_results.py --input_dir results/opt13b_b4_lowbit_update
```

## Output Files

Each experiment writes:

- `results.jsonl`: one record per target layer / rule / update magnitude.
- `summary.csv`: all records in tabular form.
- `best_by_rule.csv`: best same-batch loss decrease by update rule.
- `env.json`: model, bitwidth, target layers, backend, and environment metadata.
- `plots/loss_delta_vs_active_fraction.png`
- `plots/cosine_vs_norm_ratio.png`
- `plots/train_delta_by_rule.png`

## Interpretation

An effective low-bit update is counted when:

- at least one code changes;
- actual low-bit update norm is nonzero;
- cosine between intended SGD update and committed update is greater than 0.05;
- first-order predicted change is negative;
- same-batch loss decreases.

A heldout-effective update additionally requires heldout loss decrease.

The summary distinguishes:

- descent-aligned actual update;
- same-batch loss decrease;
- heldout loss decrease;
- multi-step sustainability when `--num_steps > 1`.

For `--num_steps > 1`, the runner keeps the selected layer as low-bit codes
plus scales across steps. To avoid accidentally mixing a sweep with a single
state trajectory, multi-step mode requires exactly one learning-rate or
relative-norm setting and exactly one concrete low-bit commit rule.

## Known Limitations

- The required path mutates a surrogate explicit lattice initialized from
  GPTQ-dequantized weights, not native packed GPTQ codes.
- Only selected Linear weights are tested.
- Same-batch improvement is not evidence of sustained training.
- Heldout loss changes from a single update can be noisy.
- CPU fallback is for debugging only; OPT-1.3B should use CUDA.
