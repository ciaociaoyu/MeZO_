#!/usr/bin/env python3
import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from contextlib import nullcontext
from torch.utils.data import Dataset


REPO_ROOT = Path(__file__).resolve().parents[2]
LARGE_ROOT = REPO_ROOT / "large_models"
if str(LARGE_ROOT) not in sys.path:
    sys.path.insert(0, str(LARGE_ROOT))

import run as large_run  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="Run 100-step OPT-1.3B/SST5 training and estimate h by two methods.")
    parser.add_argument("--output-root", type=str, default=str(REPO_ROOT / "experiments" / "h_estimation_h100"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-seed", type=int, default=42)
    parser.add_argument("--train-set-seed", type=int, default=42)
    parser.add_argument("--num-k", type=int, default=16)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--zo-eps", type=float, default=1e-4)
    parser.add_argument("--learning-rate", type=float, default=1e-6)
    parser.add_argument("--per-device-train-batch-size", type=int, default=16)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--adaptive-num-batches", type=int, default=4)
    parser.add_argument("--adaptive-num-directions", type=int, default=3)
    parser.add_argument("--two-point-num-directions-g", type=int, default=4)
    parser.add_argument("--two-point-num-directions-l", type=int, default=4)
    return parser.parse_args()


class HFDataset(Dataset):
    def __init__(self, samples, convert_one_fn):
        self.samples = samples
        self.convert_one_fn = convert_one_fn

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.convert_one_fn(self.samples[idx])


def build_args(cli_args, output_dir: str):
    args = large_run.OurArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        seed=cli_args.seed,
        data_seed=cli_args.data_seed,
        train_set_seed=cli_args.train_set_seed,
        model_name="facebook/opt-1.3b",
        task_name="SST5",
        dataset_mode="full",
        num_k=cli_args.num_k,
        trainer="zo",
        train_as_classification=True,
        load_float16=True,
        zo_quantization_bits=16,
        learning_rate=cli_args.learning_rate,
        zo_eps=cli_args.zo_eps,
        num_train_epochs=1,
        max_steps=cli_args.max_steps,
        per_device_train_batch_size=cli_args.per_device_train_batch_size,
        gradient_accumulation_steps=cli_args.gradient_accumulation_steps,
        lr_scheduler_type="constant",
        evaluation_strategy="no",
        save_strategy="no",
        logging_steps=10,
        zo_probe_every=0,
        no_eval=True,
        report_to=[],
        disable_tqdm=True,
        tag="hdiag-opt13b-sst5-fp16",
    )
    args = large_run.normalize_model_args(args)
    args = large_run.normalize_data_args(args)
    return args


def build_train_and_eval_samples(args, task):
    train_sets = task.sample_train_sets(
        num_train=args.num_train,
        num_dev=args.num_dev,
        num_eval=args.num_eval,
        num_train_sets=args.num_train_sets,
        seed=args.train_set_seed,
        dataset_mode=args.dataset_mode,
        num_k=args.num_k,
    )
    train_samples = train_sets[0]
    eval_split_samples = large_run.get_eval_split_samples(task, num_eval=args.num_eval, seed=args.train_set_seed)
    primary_eval_split = "valid" if "valid" in eval_split_samples else list(eval_split_samples.keys())[0]
    eval_samples = eval_split_samples[primary_eval_split]

    if args.dataset_mode == "full":
        if args.num_dev is not None and args.num_dev > 0:
            train_samples, dev_samples = large_run.split_train_dev_samples(
                train_samples, num_dev=args.num_dev, seed=args.data_seed
            )
        elif args.num_dev == 0:
            dev_samples = None
        else:
            train_samples, dev_samples = large_run.split_train_dev_samples(
                train_samples, dev_ratio=args.full_dev_ratio, seed=args.data_seed
            )
    elif args.num_dev is not None and args.num_dev > 0:
        train_samples, dev_samples = large_run.split_train_dev_samples(
            train_samples, num_dev=args.num_dev, seed=args.data_seed
        )
    elif args.num_dev == -1:
        train_samples, dev_samples = large_run.split_train_dev_samples(
            train_samples, dev_ratio=0.25, seed=args.data_seed
        )
    else:
        dev_samples = None

    return train_samples, (dev_samples if dev_samples is not None else eval_samples)


def build_datasets(framework, train_samples, eval_samples):
    task_template = framework.task.get_template()

    def _convert_one(sample):
        encoded_candidates, option_lens = large_run.encode_prompt(
            framework.task,
            task_template,
            [],
            sample,
            framework.tokenizer,
            max_length=framework.args.max_length,
            generation=framework.task.generation,
            generation_with_gold=True,
            max_new_tokens=framework.args.max_new_tokens,
        )

        if framework.task.generation:
            correct_candidate_id = 0
        elif isinstance(sample.correct_candidate, list):
            correct_candidate_id = sample.candidates.index(sample.correct_candidate[0])
        else:
            correct_candidate_id = sample.candidates.index(sample.correct_candidate)

        if framework.args.train_as_classification:
            return [
                {
                    "input_ids": encoded_candidates[i],
                    "labels": correct_candidate_id,
                    "option_len": option_lens[i],
                    "num_options": len(sample.candidates),
                }
                for i in range(len(encoded_candidates))
            ]

        if framework.args.only_train_option:
            return {
                "input_ids": encoded_candidates[correct_candidate_id],
                "labels": encoded_candidates[correct_candidate_id],
                "option_len": option_lens[correct_candidate_id],
            }
        return {
            "input_ids": encoded_candidates[correct_candidate_id],
            "labels": encoded_candidates[correct_candidate_id],
        }

    train_dataset = HFDataset(train_samples, _convert_one)
    eval_dataset = HFDataset(eval_samples, _convert_one)
    return train_dataset, eval_dataset


def build_trainer(framework, train_dataset, eval_dataset):
    if framework.args.only_train_option and not framework.args.non_diff:
        framework.model.original_forward = framework.model.forward
        framework.model.forward = large_run.forward_wrap_with_option_len.__get__(
            framework.model, type(framework.model)
        )

    if framework.args.train_as_classification:
        collator = large_run.DataCollatorWithPaddingAndNesting(framework.tokenizer, pad_to_multiple_of=8)
    else:
        collator = large_run.DataCollatorForTokenClassification(framework.tokenizer, pad_to_multiple_of=8)

    return large_run.OurTrainer(
        model=framework.model,
        args=framework.args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=framework.tokenizer,
        data_collator=collator,
    )


def build_named_parameters_to_optim(model):
    return [(name, param) for name, param in model.named_parameters() if param.requires_grad]


def fp16_two_point_forward(trainer, model, inputs):
    use_cuda_amp = torch.cuda.is_available()
    autocast_ctx = (
        torch.amp.autocast("cuda", dtype=torch.float16)
        if use_cuda_amp
        else nullcontext()
    )
    with autocast_ctx:
        return trainer.zo_forward(model, inputs)


def apply_seeded_direction_from_originals(named_params, originals, alpha, seed):
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    with torch.no_grad():
        for (_, param), original in zip(named_params, originals):
            z = torch.normal(mean=0, std=1, size=param.data.size(), device=param.data.device, dtype=param.data.dtype)
            param.data.copy_(original + float(alpha) * z)


def restore_originals(named_params, originals):
    with torch.no_grad():
        for (_, param), original in zip(named_params, originals):
            param.data.copy_(original)


def estimate_noise(trainer, model, inputs, seed, q=8, delta=1e-6):
    named_params = build_named_parameters_to_optim(model)
    originals = [param.data.detach().clone() for _, param in named_params]
    f_vals = []
    try:
        for i in range(int(q) + 1):
            apply_seeded_direction_from_originals(named_params, originals, i * float(delta), seed)
            with torch.no_grad():
                f_vals.append(float(fp16_two_point_forward(trainer, model, inputs)))
    finally:
        restore_originals(named_params, originals)

    T = [[0.0] * (q + 1) for _ in range(q + 1)]
    for i in range(q + 1):
        T[i][0] = f_vals[i]
    for j in range(1, q + 1):
        for i in range(q + 1 - j):
            T[i][j] = T[i + 1][j - 1] - T[i][j - 1]
    j = 3
    gamma = (math.factorial(j) ** 2) / math.factorial(2 * j)
    s_j_sq = gamma / (q + 1 - j) * sum(T[i][j] ** 2 for i in range(q + 1 - j))
    return float(math.sqrt(max(s_j_sq, 0.0)))


def estimate_nu3(trainer, model, inputs, seed, eps_f, tau1=10.0, tau2=0.1):
    named_params = build_named_parameters_to_optim(model)
    originals = [param.data.detach().clone() for _, param in named_params]
    tiny = 1e-30
    h_min, h_max = 1e-8, 5e-2
    eps_train = float(getattr(trainer.args, "zo_eps", 1e-4))
    h_theory = float(max(eps_f, tiny) ** 0.2)
    h_start = float(min(eps_train, h_theory))
    h_start = float(min(h_max, max(h_min, h_start)))
    growth = 2.0
    max_trials = 10

    def eval_at(alpha):
        try:
            apply_seeded_direction_from_originals(named_params, originals, alpha, seed)
            with torch.no_grad():
                return float(fp16_two_point_forward(trainer, model, inputs))
        finally:
            restore_originals(named_params, originals)

    with torch.no_grad():
        restore_originals(named_params, originals)
        f0 = float(fp16_two_point_forward(trainer, model, inputs))

    def dh_tests_on(h_local):
        f1 = eval_at(1.0 * h_local)
        fm1 = eval_at(-1.0 * h_local)
        delta2 = abs(fm1 - 2.0 * f0 + f1)
        snr_val = delta2 / max(eps_f, tiny)
        snr_ok = snr_val >= tau1
        prox_plus = abs(f1 - f0) / max(abs(f0), abs(f1), tiny)
        prox_minus = abs(fm1 - f0) / max(abs(f0), abs(fm1), tiny)
        prox_ok = (prox_plus <= tau2) and (prox_minus <= tau2)
        return snr_ok, prox_ok, delta2, snr_val, prox_plus, prox_minus

    def nu3_hat_at(h_local):
        f2 = eval_at(2.0 * h_local)
        f1 = eval_at(1.0 * h_local)
        fm1 = eval_at(-1.0 * h_local)
        fm2 = eval_at(-2.0 * h_local)
        delta3 = abs(-f2 + 2.0 * f1 - 2.0 * fm1 + fm2)
        nu3_hat = delta3 / (2.0 * (h_local ** 3 + tiny))
        return float(nu3_hat), float(delta3)

    chosen_h = None
    snr0, prox0, _, _, _, _ = dh_tests_on(h_start)
    if snr0 and prox0:
        chosen_h = h_start
    else:
        mode = "down" if (not prox0 or (not snr0 and not prox0)) else "up"
        for i in range(1, max_trials):
            h_i = h_start / (growth ** i) if mode == "down" else h_start * (growth ** i)
            h_i = float(min(h_max, max(h_min, h_i)))
            snr_ok, prox_ok, _, _, _, _ = dh_tests_on(h_i)
            if snr_ok and prox_ok:
                chosen_h = h_i
                break

    if chosen_h is None:
        return float("nan"), float("nan"), float("nan")

    nu3_accept, delta3_accept = nu3_hat_at(chosen_h)
    if (not math.isfinite(nu3_accept)) or nu3_accept <= 0.0 or delta3_accept == 0.0:
        return float("nan"), chosen_h, delta3_accept
    return float(nu3_accept), chosen_h, delta3_accept


def estimate_additive_h(trainer, model, inputs_list, num_directions):
    gamma = 3 ** (1 / 3)
    h_vals, eps_vals, nu3_vals = [], [], []
    seed_base = int(getattr(trainer.args, "seed", 42)) * 1000003 + int(getattr(trainer.state, "global_step", 0))
    for batch_idx, inputs in enumerate(inputs_list):
        for dir_idx in range(max(1, int(num_directions))):
            seed = seed_base + batch_idx * 97 + dir_idx * 17 + 11
            eps_i = estimate_noise(trainer, model, inputs, seed=seed)
            if (not math.isfinite(eps_i)) or eps_i <= 0.0:
                continue
            nu3_i, chosen_h, delta3 = estimate_nu3(trainer, model, inputs, seed=seed, eps_f=eps_i)
            if (not math.isfinite(nu3_i)) or nu3_i <= 0.0:
                continue
            h_i = (eps_i / nu3_i) ** (1 / 3) * gamma
            if (not math.isfinite(h_i)) or h_i <= 0.0:
                continue
            h_i = float(min(0.5, max(1e-5, h_i)))
            h_vals.append(h_i)
            eps_vals.append(eps_i)
            nu3_vals.append(nu3_i)
    if len(h_vals) == 0:
        return {
            "h_additive": float("nan"),
            "eps_est": float("nan"),
            "nu3_est": float("nan"),
            "num_valid_estimates": 0,
        }
    return {
        "h_additive": float(np.mean(np.asarray(h_vals, dtype=np.float64))),
        "eps_est": float(np.mean(np.asarray(eps_vals, dtype=np.float64))),
        "nu3_est": float(np.mean(np.asarray(nu3_vals, dtype=np.float64))),
        "num_valid_estimates": int(len(h_vals)),
    }


def quantize_delta_tensor_fp16(delta, target_dtype):
    return delta.detach().to(dtype=torch.float16).to(dtype=target_dtype)


def sample_direction_and_delta(named_params, h):
    delta_list = []
    norm_sq = 0.0
    for _, param in named_params:
        z = torch.normal(mean=0, std=1, size=param.data.size(), device=param.data.device, dtype=param.data.dtype)
        norm_sq += float(torch.sum(z.detach().float() * z.detach().float()).item())
        delta_list.append(quantize_delta_tensor_fp16(z * float(h), target_dtype=param.data.dtype))
    return delta_list, float(norm_sq)


def apply_delta_list(named_params, delta_list, multiplier):
    with torch.no_grad():
        for (_, param), delta in zip(named_params, delta_list):
            param.data.add_(float(multiplier) * delta)


def estimate_delta_rms_sampled(named_params, sample_size=4096):
    total_numel = int(sum(param.data.numel() for _, param in named_params))
    if total_numel <= 0:
        return None
    sample_size = max(1, min(int(sample_size), total_numel))
    cums = np.cumsum([int(param.data.numel()) for _, param in named_params])
    picks = np.random.randint(0, total_numel, size=sample_size)
    vals = []
    for flat_idx in picks:
        param_idx = int(np.searchsorted(cums, int(flat_idx), side="right"))
        prev = 0 if param_idx == 0 else int(cums[param_idx - 1])
        local_idx = int(flat_idx) - prev
        tensor = named_params[param_idx][1].data.detach().view(-1)[local_idx].float().cpu()
        vals.append(float(tensor.item()))
    sample = torch.tensor(vals, dtype=torch.float32)
    sample_low = sample.to(dtype=torch.float16)
    sample_next = torch.nextafter(sample_low, torch.full_like(sample_low, float("inf")))
    delta_i = (sample_next - sample_low).abs().to(dtype=torch.float32)
    delta_rms = torch.sqrt(torch.mean(delta_i * delta_i))
    val = float(delta_rms.item())
    if (not math.isfinite(val)) or val <= 0.0:
        return None
    return val


def estimate_two_point_h(trainer, model, probe_inputs, current_h, num_directions_g, num_directions_l):
    named_params = build_named_parameters_to_optim(model)
    delta_tilde = estimate_delta_rms_sampled(named_params)

    vals = []
    for _ in range(max(1, int(num_directions_g))):
        delta_list, _ = sample_direction_and_delta(named_params, float(current_h))
        try:
            apply_delta_list(named_params, delta_list, +1.0)
            loss_plus = float(fp16_two_point_forward(trainer, model, probe_inputs).item())
        finally:
            apply_delta_list(named_params, delta_list, -1.0)
        try:
            apply_delta_list(named_params, delta_list, -1.0)
            loss_minus = float(fp16_two_point_forward(trainer, model, probe_inputs).item())
        finally:
            apply_delta_list(named_params, delta_list, +1.0)
        d_hat = (loss_plus - loss_minus) / (2.0 * float(current_h))
        if math.isfinite(d_hat):
            vals.append(abs(float(d_hat)))
    g_tilde = None if len(vals) == 0 else float(math.sqrt(math.pi / 2.0) * (sum(vals) / len(vals)))

    delta_hat = delta_tilde
    l_tilde = None
    h2 = None
    if delta_hat is not None and math.isfinite(delta_hat) and delta_hat > 0.0:
        c2 = 1.0
        eps_num = 1e-12
        q_l = 0.5
        h2 = float(max(float(delta_hat), c2 * math.sqrt(float(delta_hat))))
        base_loss = float(fp16_two_point_forward(trainer, model, probe_inputs).item())
        if math.isfinite(base_loss):
            lambdas = []
            for _ in range(max(1, int(num_directions_l))):
                delta_list, norm_sq = sample_direction_and_delta(named_params, h2)
                try:
                    apply_delta_list(named_params, delta_list, +1.0)
                    loss1 = float(fp16_two_point_forward(trainer, model, probe_inputs).item())
                    apply_delta_list(named_params, delta_list, +1.0)
                    loss2 = float(fp16_two_point_forward(trainer, model, probe_inputs).item())
                finally:
                    apply_delta_list(named_params, delta_list, -2.0)
                k_hat = (loss2 - 2.0 * loss1 + base_loss) / max(h2 ** 2, 1e-30)
                lam = abs(float(k_hat)) / (float(norm_sq) + eps_num)
                if math.isfinite(lam):
                    lambdas.append(float(lam))
            if len(lambdas) > 0:
                l_tilde = float(np.quantile(np.asarray(lambdas, dtype=np.float64), q_l))

    h_two_point_tilde = None
    h_two_point = float("nan")
    if (
        delta_hat is not None and g_tilde is not None and l_tilde is not None
        and delta_hat > 0.0 and g_tilde > 0.0 and l_tilde > 0.0
    ):
        d_dim = max(1, int(sum(param.data.numel() for _, param in named_params)))
        h_two_point_tilde = (
            (float(delta_hat) ** 2 * float(g_tilde) ** 2)
            / (16.0 * (float(l_tilde) ** 2) * float(d_dim) * float(d_dim + 2))
        ) ** 0.25
        h_two_point_tilde = float(min(0.5, max(1e-5, h_two_point_tilde)))
        h_two_point = h_two_point_tilde

    return {
        "h_two_point": h_two_point,
        "h_two_point_tilde": float(h_two_point_tilde) if h_two_point_tilde is not None else float("nan"),
        "delta_tilde": float(delta_tilde) if delta_tilde is not None else float("nan"),
        "g_tilde": float(g_tilde) if g_tilde is not None else float("nan"),
        "l_tilde": float(l_tilde) if l_tilde is not None else float("nan"),
        "delta_hat": float(delta_hat) if delta_hat is not None else float("nan"),
        "g_hat": float(g_tilde) if g_tilde is not None else float("nan"),
        "l_hat": float(l_tilde) if l_tilde is not None else float("nan"),
        "h2": float(h2) if h2 is not None else float("nan"),
    }


def main():
    cli_args = parse_args()
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir = Path(cli_args.output_root) / f"opt13b_sst5_hdiag_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    args = build_args(cli_args, str(output_dir / "run"))
    large_run.set_seed(args.seed)
    task = large_run.get_task(args.task_name)
    framework = large_run.Framework(args, task)
    framework.tokenizer.padding_side = "left"
    train_samples, eval_samples = build_train_and_eval_samples(args, task)
    train_dataset, eval_dataset = build_datasets(framework, train_samples, eval_samples)
    trainer = build_trainer(framework, train_dataset, eval_dataset)

    train_dataloader = trainer.get_train_dataloader()
    batch_iter = iter(train_dataloader)
    inputs_list = []
    for _ in range(max(1, int(cli_args.adaptive_num_batches))):
        try:
            inputs_list.append(next(batch_iter))
        except StopIteration:
            batch_iter = iter(train_dataloader)
            inputs_list.append(next(batch_iter))
    probe_inputs = inputs_list[0]
    pre_train_probe_loss = float(fp16_two_point_forward(trainer, framework.model, probe_inputs).item())

    train_start = time.time()
    train_result = trainer.train()
    train_wall_seconds = time.time() - train_start
    framework.model = trainer.model

    debug_seed = int(args.seed) * 1000003 + int(getattr(trainer.state, "global_step", 0)) + 11
    probe_loss = float(fp16_two_point_forward(trainer, framework.model, probe_inputs).item())
    eps_debug = estimate_noise(trainer, framework.model, probe_inputs, seed=debug_seed)
    nu3_debug, nu3_chosen_h_debug, delta3_debug = (
        estimate_nu3(trainer, framework.model, probe_inputs, seed=debug_seed, eps_f=eps_debug)
        if math.isfinite(eps_debug) and eps_debug > 0.0
        else (float("nan"), float("nan"), float("nan"))
    )

    additive = estimate_additive_h(
        trainer,
        framework.model,
        inputs_list,
        num_directions=cli_args.adaptive_num_directions,
    )
    two_point = estimate_two_point_h(
        trainer,
        framework.model,
        probe_inputs,
        current_h=float(args.zo_eps),
        num_directions_g=cli_args.two_point_num_directions_g,
        num_directions_l=cli_args.two_point_num_directions_l,
    )

    result = {
        "task": "SST5",
        "model_name": "facebook/opt-1.3b",
        "precision": "fp16",
        "zo_quantization_bits": 16,
        "config": {
            "dataset_mode": args.dataset_mode,
            "num_k": int(args.num_k),
            "seed": int(args.seed),
            "data_seed": int(args.data_seed),
            "train_set_seed": int(args.train_set_seed),
            "learning_rate": float(args.learning_rate),
            "zo_eps": float(args.zo_eps),
            "max_steps": int(args.max_steps),
            "per_device_train_batch_size": int(args.per_device_train_batch_size),
            "gradient_accumulation_steps": int(args.gradient_accumulation_steps),
        },
        "train_split_size": int(len(train_samples)),
        "eval_split_size": int(len(eval_samples)),
        "train_metrics": getattr(train_result, "metrics", None),
        "train_wall_seconds": float(train_wall_seconds),
        "post_train_global_step": int(getattr(trainer.state, "global_step", 0)),
        "pre_train_probe_loss": pre_train_probe_loss,
        "post_train_probe_loss": probe_loss,
        "debug": {
            "probe_loss": probe_loss,
            "single_seed": debug_seed,
            "single_seed_eps_est": eps_debug,
            "single_seed_nu3_est": nu3_debug,
            "single_seed_nu3_chosen_h": nu3_chosen_h_debug,
            "single_seed_delta3": delta3_debug,
        },
        "additive_h_estimation": additive,
        "two_point_h_estimation": two_point,
    }

    result_path = output_dir / "result.json"
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(json.dumps({"output_dir": str(output_dir), "result_path": str(result_path)}, ensure_ascii=False))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
