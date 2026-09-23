#!/usr/bin/env python
"""Resumable OPT-2.7B MeZO radius sweep with a separate gradient audit."""
from __future__ import annotations

import argparse
import contextlib
import csv
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
import traceback
import uuid

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import opt_sst5_guardrail_probe as prior
import smoke_rtnclip_roberta_sst5 as rtn
from opt_mezo_option_tasks import get_option_task

TRAIN_H = [1e-5, 1e-4, 1e-3, 3e-3, 1e-2, 3e-2]
PROBE_H = [1e-7, 3e-7, 1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4,
           1e-3, 1.5e-3, 2e-3, 3e-3, 5e-3, 1e-2, 3e-2, 1e-1]
STOP = False


def request_stop(signum, frame):
    global STOP
    STOP = True


def stamp():
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(value, indent=2, default=str, allow_nan=False) + "\n")
    os.replace(tmp, path)


def read_json(path, default=None):
    return json.loads(Path(path).read_text()) if Path(path).exists() else default


def append(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as stream:
        stream.write(json.dumps(value, default=str, allow_nan=False) + "\n")
        stream.flush()


def records(path):
    if not Path(path).exists():
        return []
    rows = []
    for line in Path(path).read_text().splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def csv_write(path, rows, fields=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = fields or list(dict.fromkeys(k for row in rows for k in row))
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    with tmp.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, path)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def config_hash(config):
    return hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()


def direction(master, seed):
    # Generate FP32 normals in the identical parameter order for every precision.
    gen = torch.Generator(device="cuda").manual_seed(int(seed))
    for name, w in master.items():
        yield name, torch.randn(w.shape, dtype=torch.float32, device=w.device, generator=gen)


@torch.no_grad()
def grids(master, names, precision):
    if precision not in ("int4", "int8"):
        return {}
    bits = int(precision[3:])
    return {name: rtn.compute_rtnclip_state(name, master[name], bits, 128, collect_stats=False)[0]
            for name in names}


@torch.no_grad()
def apply(params, master, states, seed=None, h=0.0, sign=0.0):
    iterator = direction(master, seed) if seed is not None else ((name, None) for name in master)
    for name, u in iterator:
        v = master[name].float()
        if u is not None:
            v = v.add(u, alpha=sign * h)
        if name in states:
            v = rtn.quantize_with_state(v, states[name])
        params[name].copy_(v)


@torch.no_grad()
def restore(params, master):
    for name, p in params.items():
        p.copy_(master[name])


def option_loss(model, batch):
    """Same MeZO mean-token option likelihood CE, selecting tokens before FP32 softmax."""
    out = model(input_ids=batch["input_ids"], attention_mask=batch.get("attention_mask"),
                use_cache=False, return_dict=True)
    scores = []
    for i, length in enumerate(batch["option_len"].tolist()):
        length = int(length)
        if length <= 0:
            raise ValueError("Empty verbalizer after tokenization")
        logits = out.logits[i, -length-1:-1, :].float()
        labels = batch["input_ids"][i, -length:]
        mask = labels != model.config.pad_token_id
        logp = logits.log_softmax(-1).gather(-1, labels[:, None]).squeeze(-1)
        scores.append((logp * mask).sum() / mask.sum().clamp_min(1))
    scores = torch.stack(scores)
    losses, correct = [], []
    start = 0
    while start < len(scores):
        end = start + int(batch["num_options"][start])
        label = batch["labels"][start].long()
        losses.append(F.cross_entropy(scores[start:end][None], label[None]))
        correct.append((scores[start:end].argmax() == label).detach())
        start = end
    return torch.stack(losses).mean(), torch.stack(correct).sum(), len(losses)


def batch_loss(model, batch, microbatch, backward=False):
    spans = prior.batch_group_ranges(batch)
    total, correct = 0.0, 0
    context = contextlib.nullcontext if backward else torch.no_grad
    with context():
        for start in range(0, len(spans), microbatch):
            end = min(start + microbatch, len(spans))
            sub = prior.slice_batch(batch, spans[start][0], spans[end-1][1])
            loss, hit, n = option_loss(model, sub)
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite option loss")
            weight = n / len(spans)
            total += loss.detach().item() * weight
            correct += int(hit.item())
            if backward:
                (weight * loss).backward()
    return total, correct, len(spans)


def load_context(config, outdir, probe=False):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformers.utils import logging as hf_logging
    hf_logging.set_verbosity_error()
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "1")))
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(config["train_seed"])
    tokenizer = AutoTokenizer.from_pretrained(config["model"], use_fast=False, local_files_only=True)
    tokenizer.padding_side = "left"
    tokenizer.bos_token_id = 0
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = torch.float32 if probe or config["precision"] == "fp32" else torch.float16
    model = AutoModelForCausalLM.from_pretrained(config["model"], torch_dtype=torch.float16,
        local_files_only=True, low_cpu_mem_usage=True, attn_implementation="eager")
    model.config.use_cache = False
    model.config.pad_token_id = tokenizer.pad_token_id
    model.eval().to(device="cuda", dtype=dtype)
    params = {n: p for n, p in model.named_parameters() if p.is_floating_point()}
    for p in params.values():
        p.requires_grad_(probe)
    # The common probe center is the exact FP16 checkpoint promoted to FP32.
    master = {n: p.detach().clone().to(torch.float16) for n, p in params.items()}
    q_names = prior.linear_weight_names(model, params)
    task = get_option_task(config["task"])
    train = list(task.samples["train"])
    valid = list(task.samples["valid"])
    datasets = [prior.OptionDataset(samples, task, tokenizer, config["max_seq_len"])
                for samples in (train, valid)]
    collate = prior.DataCollatorWithPaddingAndNesting(tokenizer, pad_to_multiple_of=8)
    info = {"train_examples": len(train), "eval_examples": len(valid), "dataset_mode": "full",
            "template_class": type(task.get_template()).__name__, "model_commit": model.config._commit_hash,
            "d": sum(p.numel() for p in params.values()),
            "d_quantized_linear": sum(params[n].numel() for n in q_names),
            "parameters": [{"name": n, "shape": list(p.shape), "quantized": n in q_names} for n, p in params.items()],
            "train_sample_ids": [prior.stable_sample_id(x, i) for i, x in enumerate(train)],
            "eval_sample_ids": [prior.stable_sample_id(x, i) for i, x in enumerate(valid)]}
    write_json(outdir / "dataset_and_scope.json", info)
    template = task.get_template()
    write_json(outdir / "prompt_examples.json", [
        {"id": prior.stable_sample_id(x, i), "data": x.data, "prompt": template.encode(x),
         "candidates": [template.verbalize(x, c) for c in x.candidates],
         "correct_candidate": x.correct_candidate, "encoded": datasets[0][i]}
        for i, x in enumerate(train[:16])])
    return model, params, master, q_names, datasets, collate, info


def make_batch(dataset, ids, collate):
    batch = collate([dataset[int(i)] for i in ids])
    return {k: v.cuda() if torch.is_tensor(v) else v for k, v in batch.items()}


class PairedOrder:
    def __init__(self, count, seed, bs):
        self.count, self.seed, self.bs = count, seed, bs
        self.epoch, self.order = None, None

    def at(self, step):
        ids = []
        for offset in range((step-1)*self.bs, step*self.bs):
            epoch, pos = divmod(offset, self.count)
            if self.epoch != epoch:
                self.order = np.random.default_rng(self.seed + 1000003*epoch).permutation(self.count)
                self.epoch = epoch
            ids.append(int(self.order[pos]))
        return ids


def evaluate(model, params, master, names, config, dataset, collate, limit=0):
    states = grids(master, names, config["precision"])
    try:
        apply(params, master, states)
        total_loss, correct, n = 0.0, 0, 0
        count = min(len(dataset), limit) if limit else len(dataset)
        for start in range(0, count, config["microbatch"]):
            batch = make_batch(dataset, range(start, min(count, start + config["microbatch"])), collate)
            loss, hit, size = batch_loss(model, batch, config["microbatch"])
            total_loss += loss * size
            correct += hit
            n += size
        return {"eval_acc": correct / n, "eval_loss": total_loss / n, "eval_examples": n}
    finally:
        restore(params, master)


def checkpoint(outdir, master, step, config, elapsed):
    path = outdir / "checkpoint.pt"
    tmp = outdir / "checkpoint.tmp"
    torch.save({"master": {n: w.cpu() for n, w in master.items()}, "step": step,
                "config_hash": config_hash(config), "runtime_sec": elapsed}, tmp)
    os.replace(tmp, path)


def recover_training_logs(outdir, start_step):
    """Archive uncommitted records before resuming the exact checkpoint state."""
    archive = outdir / "interrupted_records" / (str(time.time_ns()) + "_" + uuid.uuid4().hex[:8])
    for filename in ("train.jsonl", "eval.jsonl"):
        path = outdir / filename
        if not path.exists():
            continue
        old = records(path)
        kept = [r for r in old if start_step > 1 and r["step"] < start_step]
        if len(kept) != len(old):
            archive.mkdir(parents=True, exist_ok=True)
            path.rename(archive / filename)
            path.write_text("".join(json.dumps(r) + "\n" for r in kept))
    if archive.exists():
        append(outdir / "resume_audit.jsonl", {"time": stamp(), "start_step": start_step,
            "archived_records": str(archive), "reason": "rollback to committed master weights"})


def train(config, outdir, smoke=False):
    started = time.time()
    model, params, master, q_names, datasets, collate, info = load_context(config, outdir)
    if smoke:
        check_batch = make_batch(datasets[0], range(2), collate)
        with torch.no_grad():
            model.original_forward = model.forward
            expected = prior.stable_option_forward(model, **check_batch, return_dict=True).loss.item()
            actual = batch_loss(model, check_batch, 2)[0]
            micro = batch_loss(model, check_batch, 1)[0]
        if not np.isclose(expected, actual, rtol=2e-5, atol=2e-5):
            raise AssertionError(f"Official option loss mismatch: {expected} != {actual}")
        if not np.isclose(actual, micro, rtol=2e-3, atol=2e-3):
            raise AssertionError(f"Microbatch loss mismatch: {actual} != {micro}")
        write_json(outdir / "loss_equivalence.json", {"official_stable_loss": expected,
            "selected_token_loss": actual, "microbatch1_loss": micro, "passed": True})
        del check_batch
    start_step, previous_runtime = 1, 0.0
    if (outdir / "checkpoint.pt").exists():
        saved = torch.load(outdir / "checkpoint.pt", map_location="cpu", weights_only=False)
        if saved["config_hash"] != config_hash(config):
            raise RuntimeError("Resume config hash mismatch")
        for n, w in saved["master"].items():
            master[n].copy_(w)
        start_step = int(saved["step"]) + 1
        previous_runtime = saved["runtime_sec"]
        del saved
    recover_training_logs(outdir, start_step)
    restore(params, master)
    limit = 16 if smoke else 0
    if start_step == 1:
        ev = evaluate(model, params, master, q_names, config, datasets[1], collate, limit)
        append(outdir / "eval.jsonl", {"step": 0, **ev})
        print(f"initial eval {ev}", flush=True)
    order = PairedOrder(len(datasets[0]), config["data_seed"], config["batch_size"])
    step = start_step - 1
    step_times = []
    for step in range(start_step, config["steps"] + 1):
        t0 = time.time()
        ids = order.at(step)
        batch = make_batch(datasets[0], ids, collate)
        states = grids(master, q_names, config["precision"])
        seed = config["direction_seed"] * 1000003 + step * 1009
        try:
            apply(params, master, states, seed, config["h"], 1)
            lp = batch_loss(model, batch, config["microbatch"])[0]
            apply(params, master, states, seed, config["h"], -1)
            lm = batch_loss(model, batch, config["microbatch"])[0]
        finally:
            restore(params, master)
        dh = (lp-lm)/(2*config["h"])
        if not math.isfinite(dh):
            raise FloatingPointError(f"Nonfinite d_h at step {step}")
        with torch.no_grad():
            # Match the project's FP16-master MeZO update rounding convention.
            for name, u in direction(master, seed):
                master[name].add_(u.to(master[name].dtype), alpha=-config["lr"]*dh)
        del states, batch
        restore(params, master)
        elapsed = time.time()-t0
        step_times.append(elapsed)
        row = {"step": step, "h": config["h"], "direction_seed": seed, "sample_indices": ids,
               "loss_plus": lp, "loss_minus": lm, "d_h": dh, "step_sec": elapsed}
        append(outdir / "train.jsonl", row)
        if step == 1 or step % 25 == 0:
            print(f"{config['id']} step={step} loss+={lp:.6g} loss-={lm:.6g} dh={dh:.6g} sec={elapsed:.3f}", flush=True)
        if step % config["eval_every"] == 0 or step == config["steps"]:
            ev = evaluate(model, params, master, q_names, config, datasets[1], collate, limit)
            append(outdir / "eval.jsonl", {"step": step, **ev})
            print(f"eval step={step} {ev}", flush=True)
        if step % config["checkpoint_every"] == 0 or step == config["steps"] or STOP:
            checkpoint(outdir, master, step, config, previous_runtime + time.time()-started)
        if STOP:
            break
    evals = records(outdir / "eval.jsonl")
    trained_evals = [r for r in evals if r["step"] > 0]
    best = max(trained_evals or evals, key=lambda r: r["eval_acc"])
    summary = {**config, "status": "complete" if step == config["steps"] else "pending",
               "run_type": "smoke" if smoke else ("full" if step == 20000 else "partial"),
               "steps_completed": step, "initial_eval_acc": evals[0]["eval_acc"],
               "best_eval_acc": best["eval_acc"], "best_eval_step": best["step"],
               "final_eval_acc": evals[-1]["eval_acc"], "final_eval_step": evals[-1]["step"],
               "runtime_sec": previous_runtime + time.time()-started,
               "median_step_sec": float(np.median(step_times)) if step_times else None,
               "peak_gpu_memory_gib": torch.cuda.max_memory_allocated()/1024**3,
               "train_examples": len(datasets[0]), "source_path": str(outdir / "train.jsonl")}
    write_json(outdir / "summary.json", summary)
    print(json.dumps(summary), flush=True)
    return summary


def prepare(root):
    root.mkdir(parents=True, exist_ok=True)
    for folder in ("configs", "logs", "runs", "probes", "smoke", "summaries", "figures", "code", "locks"):
        (root / folder).mkdir(exist_ok=True)
    common = {"model": "facebook/opt-2.7b", "mode": "dense", "task_path": "mezo_option",
              "dataset_mode": "full", "group_size": 128, "master_dtype": "fp16",
              "direction_distribution": "torch.randn float32 Gaussian; unnormalized", "q": 1,
              "lr": 3e-7, "lr_source": "slurm/opt13b_int4_roberta_matched_h100.sbatch",
              "steps": 20000, "eval_every": 500, "checkpoint_every": 500,
              "batch_size": 16, "microbatch": 2, "train_seed": 16, "data_seed": 16, "direction_seed": 16}
    jobs = []
    for task in ("sst-2", "rte"):
        for precision in ("fp16", "int8", "int4"):
            for h in TRAIN_H:
                jobs.append({**common, "id": f"{task}_{precision}_h{h:g}_seed16", "kind": "train",
                    "task": task, "precision": precision, "quantizer": "none" if precision == "fp16" else "RTNClip_shared_grid_K1",
                    "h": h, "h_policy": "predeclared_grid", "max_seq_len": 128 if task == "sst-2" else 256})
    for seed in (32, 64):
        for h in TRAIN_H:
            jobs.append({**common, "id": f"sst-2_fp16_h{h:g}_seed{seed}", "kind": "train", "task": "sst-2",
                "precision": "fp16", "quantizer": "none", "h": h, "h_policy": "predeclared_grid", "max_seq_len": 128,
                "train_seed": seed, "data_seed": seed, "direction_seed": seed})
    # One task job shares the same true reference and calibration over precisions.
    for task in ("sst-2", "rte"):
        jobs.append({**common, "id": f"{task}_mult precision_probe".replace(" ", ""), "kind": "probe", "task": task,
                     "precision": "all", "h_grid": PROBE_H, "max_seq_len": 128 if task == "sst-2" else 256,
                     "batches": 4, "directions": 64, "l_directions": 16, "gradient_microbatch": 1,
                     "l_seeds": [710000+1009*i for i in range(16)],
                     "audit_seeds": [910000+1009*i for i in range(64)]})
    existing = read_json(root / "jobs.json")
    if existing is not None and existing != jobs:
        raise RuntimeError("Existing locked jobs differ; choose a new output directory")
    write_json(root / "jobs.json", jobs)
    csv_write(root / "run_manifest.csv", [{**j, "status": "pending"} for j in jobs])
    for j in jobs:
        write_json(root / "configs" / f"{j['id']}.json", j)
    code_paths = [Path(__file__), ROOT / "tools/smoke_rtnclip_roberta_sst5.py", ROOT / "tools/opt_sst5_guardrail_probe.py",
                  ROOT / "tools/opt_mezo_option_tasks.py", ROOT / "large_models/templates.py", ROOT / "large_models/tasks.py",
                  ROOT / "large_models/utils.py"]
    code_paths.extend([ROOT / "tools/opt27b_radius_monitor.py", ROOT / "slurm/opt27b_radius_extension_l4.sbatch"])
    import shutil
    for path in code_paths:
        shutil.copy2(path, root / "code" / path.name)
    write_json(root / "metadata.json", {"created_at": stamp(), "git_commit": prior.git_commit(),
        "code_sha256": {str(p.relative_to(ROOT)): sha(p) for p in code_paths}, "training_runs": 48,
        "local_allocation": os.environ.get("SLURM_JOB_ID"), "accuracy_good_set": "mean final dev accuracy >= max_h(mean final dev accuracy)-0.01",
        "loss": "CE over mean token option log-likelihoods in float32", "no_accuracy_selected_h": True})
    (root / "METHOD_LOCK.md").write_text("""# OPT-2.7B radius extension method lock

Two full-data tasks: SST-2 and RTE, official MeZO templates, candidate verbalizers,
mean-token likelihood classification CE. Training is a full-data extension, not
an exact original few-shot benchmark reproduction. No chat template or classifier head.

Dense MeZO q=1. All unique floating model parameters are perturbed. FP16 master,
FP32 Gaussian directions sampled identically across h and precision; each side
is generated from the restored center. FP16 forward or INT8/INT4 RTNClip G128
linear-weight forward. RTNClip grid is refreshed from center once per step and
shared across both signs. No adapters, residual state, or gradient training.
The established master update casts the direction to FP16 before add_.

48 predeclared full runs: two tasks x three precisions x six h x seed16;
SST-2 FP16 all six h repeated at seeds32/64. Full train split and full validation;
20k steps, effective BS16, microbatch2, eval500, lr3e-7. Dataset order and
Gaussian seeds are independent of h and exactly resumable from step index.
Training h={1e-5,1e-4,1e-3,3e-3,1e-2,3e-2}. Final dev accuracy is primary;
best dev accuracy and initial accuracy are separate. A chance-level flat curve
does not establish a learned plateau. Differences across paired seeds are retained.

Probe: common FP16 pretrained center promoted to FP32 for true clean backward,
4 disjoint fixed train batches x16 examples, 64 common FP32 Gaussian directions.
Calibration uses separate 16 directions on batch0, forward second difference
[F(w+2rho*u)-2F(w+rho*u)+F(w)]/rho^2 divided by ||u||^2.
choose_l_plateau selects clean L_q90 before audit. Delta_eff is parameter-weighted
RTNClip scale RMS/sqrt(6). G is direct clean gradient norm. No tail fitting.
h_ref=.5*sqrt(Delta_eff*G/(L*sqrt(d*(d+2)))); c_d=(d+4)/(d+1);
rho_th=c_d*((Delta_eff/(2*h))^2+(2*h*L*sqrt(d*(d+2))/G)^2).
tau={1,5,10}; no certificate if rho_min>tau. FP32/FP16 lack a uniform RTN
Delta, so plug-in theory is explicitly unavailable for these modes.

Audit saves true scalar nMSE, V_total, V_dir, V_h_dep, direct cross term and rho.
No geometry proxy is called true MSE. Quantized curves are at one common center
and paired batches/seeds. Numerical failures and non-U curves are retained.
Primary theory/audit figures use calibration batch0. Batches1-3 are separately
reported robustness checks; pooled four-batch metrics have a distinct CSV.
""")
    (root / "README.md").write_text("""# OPT-2.7B radius extension

Status: prepared; live state is in run_manifest.csv, summaries/, and logs/.
See METHOD_LOCK.md and configs/ for the predeclared protocol. Original prompts,
token ids, full split sizes and model revision are saved in each run directory.
48 full training configurations run on at most two L4 workers; the local A100
runs the two expensive multi-precision true-gradient probe jobs. Keeping the
training matrix on L4 avoids architecture-dependent CUDA RNG stream differences.
Probe batches/directions are not training seeds.

Checkpoints are atomic and replaced every500 steps. Records beyond the latest
checkpoint are rolled back on resume. Both data and direction streams use step
indexed seeds. Each run has an exclusive filesystem lock to avoid duplication.
Summaries and PDF/PNG figures are regenerated from logs by the aggregate command.
No empirical result has been substituted for a theoretical prediction.
""")
    print(f"Prepared {len(jobs)} jobs at {root}")


def execute(root, config, outdir, smoke=False):
    outdir.mkdir(parents=True, exist_ok=True)
    write_json(outdir / "config.json", config)
    write_json(outdir / "environment.json", prior.environment_info())
    append(outdir / "invocations.jsonl", {"time":stamp(),"command":[sys.executable,*sys.argv],
        "git_commit":prior.git_commit(),"script_sha256":sha(__file__),
        "quantizer_sha256":sha(ROOT/"tools/smoke_rtnclip_roberta_sst5.py")})
    write_json(outdir / "status.json", {"status": "running", "started": stamp(), "hostname": socket.gethostname(),
                                       "pid": os.getpid(), "job_id": os.environ.get("SLURM_JOB_ID")})
    try:
        result = train(config, outdir, smoke) if config["kind"] == "train" else probe(config, outdir, smoke)
        write_json(outdir / "status.json", {"status": result["status"], "updated_at": stamp()})
        return 0 if result["status"] == "complete" else 75
    except Exception as exc:
        append(outdir / "failures.jsonl", {"time": stamp(), "error": repr(exc), "traceback": traceback.format_exc()})
        write_json(outdir / "status.json", {"status": "retry", "updated_at": stamp(), "error": repr(exc)})
        raise


def finite_number(x):
    return float(x) if math.isfinite(float(x)) else None


def gradient_refs(model, params, master, batch, config):
    model.float()
    restore(params, master)
    for p in params.values():
        p.requires_grad_(True)
    model.zero_grad(set_to_none=True)
    clean_loss = batch_loss(model, batch, config["gradient_microbatch"], backward=True)[0]
    g2_tensor = torch.zeros((), device="cuda", dtype=torch.float64)
    for name, p in params.items():
        if p.grad is None or not torch.isfinite(p.grad).all():
            raise RuntimeError(f"Missing/nonfinite clean gradient: {name}")
        g2_tensor += p.grad.double().square().sum()
    g2 = g2_tensor.item()
    refs = []
    for i, seed in enumerate(config["audit_seeds"]):
        dot = torch.zeros((), device="cuda", dtype=torch.float64)
        norm2 = torch.zeros_like(dot)
        for name, u in direction(master, seed):
            dot += (params[name].grad.double() * u.double()).sum()
            norm2 += u.double().square().sum()
        refs.append({"direction_seed": seed, "d_star": dot.item(), "norm_u2": norm2.item()})
        if i % 8 == 0:
            print(f"clean gradient dots {i+1}/{len(config['audit_seeds'])}", flush=True)
    model.zero_grad(set_to_none=True)
    for p in params.values():
        p.requires_grad_(False)
    torch.cuda.empty_cache()
    return {"clean_loss": clean_loss, "G2": g2, "G": math.sqrt(g2), "refs": refs}


def calibrate_theory(model, params, master, q_names, batch, reference, config, outdir):
    path = outdir / "PREDICTED_WINDOWS_FROZEN.json"
    if path.exists():
        return read_json(path)
    model.float()
    restore(params, master)
    base_clean = batch_loss(model, batch, config["microbatch"])[0]
    raw_path = outdir / "clean_l_raw.jsonl"
    raw = records(raw_path)
    done = {(r["seed"], r["rho"]) for r in raw}
    l_grid = [1e-5,3e-5,1e-4,3e-4,1e-3,3e-3,1e-2]
    for seed in config["l_seeds"]:
        norm2 = sum(u.double().square().sum().item() for _, u in direction(master, seed))
        for h in l_grid:
            if (seed,h) in done:
                continue
            try:
                apply(params, master, {}, seed, h, 1)
                l1 = batch_loss(model, batch, config["microbatch"])[0]
                apply(params, master, {}, seed, 2*h, 1)
                l2 = batch_loss(model, batch, config["microbatch"])[0]
                k = (l2-2*l1+base_clean)/h**2
                row = {"seed": seed, "rho": h, "norm_u2": norm2, "K_raw": k,
                       "L_unit": abs(k)/norm2, "loss1": l1, "loss2": l2, "status": "complete"}
            finally:
                restore(params, master)
            raw.append(row)
            append(raw_path, row)
        print(f"clean L seed={seed} complete", flush=True)
        if STOP:
            return None
    candidates = []
    for h in l_grid:
        values = np.array([r["L_unit"] for r in raw if r["rho"] == h])
        ks = np.array([r["K_raw"] for r in raw if r["rho"] == h])
        mad = np.median(np.abs(ks-np.median(ks)))
        candidates.append({"h2": h, "lambda_q50": float(np.quantile(values,.5)),
            "lambda_q90": float(np.quantile(values,.9)), "lambda_q95": float(np.quantile(values,.95)),
            "SNR2": float(np.median(np.abs(ks))/(1.4826*mad+1e-30)), "finite_rate": 1.0})
    selected, status = prior.choose_l_plateau(candidates)
    csv_write(outdir / "clean_l_candidates.csv", candidates)
    d = sum(w.numel() for w in master.values())
    l_value = selected.get("lambda_q90")
    frozen = {"created_at": stamp(), "G": reference["G"], "L": l_value,
              "L_rho": selected.get("h2"), "L_status": status, "d": d, "precisions": {}}
    for precision in ("int8", "int4"):
        states = grids(master, q_names, precision)
        scale_stats = prior.quantizer_scale_stats(states)
        theory = (prior.theory_window(scale_stats["Delta_eff"], reference["G"], l_value, d)
                  if l_value is not None and l_value > 0 else {"status": "no_valid_L", "windows": []})
        frozen["precisions"][precision] = {**scale_stats, **theory}
        del states
    for precision in ("fp32", "fp16"):
        frozen["precisions"][precision] = {"status": "no_uniform_delta", "notes": "No empirical back-solving of Delta"}
    write_json(path, frozen)
    return frozen


@torch.no_grad()
def geometry(master, states, seed, h, dtype):
    # Deterministic stratified coordinates; full tensors still pass through the real Q.
    sq, target_sq, dot, active, count, clip, clip_count = 0.,0.,0.,0,0,0,0
    for name, u in direction(master, seed):
        plus, minus = master[name].float().add(u, alpha=h), master[name].float().add(u, alpha=-h)
        if name in states:
            state = states[name]
            plus = rtn.quantize_with_state(plus, state)
            minus = rtn.quantize_with_state(minus, state)
        plus, minus = plus.to(dtype), minus.to(dtype)
        stride = max(1, plus.numel()//4096)
        diff = (plus.flatten()[::stride].double()-minus.flatten()[::stride].double())
        target = 2*h*u.flatten()[::stride].double()
        weight = plus.numel()/diff.numel()
        sq += diff.square().sum().item()*weight
        target_sq += target.square().sum().item()*weight
        dot += (diff*target).sum().item()*weight
        active += (diff != 0).sum().item()*weight
        count += plus.numel()
        if name in states:
            s = states[name]
            # Boundary occupancy is sampled with the same represented parameter weights.
            gp = plus.float().reshape(plus.shape[0],-1,s.group_size)
            gm = minus.float().reshape(minus.shape[0],-1,s.group_size)
            boundary = s.scales * s.qmax
            mask = ((gp.abs() >= boundary*.999) | (gm.abs() >= boundary*.999)).flatten()[::stride]
            clip += mask.sum().item()*(plus.numel()/mask.numel())
            clip_count += plus.numel()
    return {"active_fraction": active/count, "visible_norm_ratio": math.sqrt(sq/(target_sq+1e-30)),
            "visible_cosine": dot/math.sqrt(sq*target_sq+1e-30),
            "p_clip": clip/clip_count if clip_count else None,
            "geometry_status": "stratified_per_tensor_4096_coordinates"}


def vector_record(dh, ds, norm2, g2):
    total = dh**2*norm2 - 2*dh*ds + g2
    floor = ds**2*norm2 - 2*ds**2 + g2
    dep = (dh-ds)**2*norm2
    cross_direct = 2*(dh-ds)*ds*(norm2-1)
    residual = total-floor-dep-cross_direct
    scale = max(1., abs(total), abs(floor), abs(dep), abs(cross_direct))
    if abs(residual)/scale > 1e-10:
        raise AssertionError("Vector MSE decomposition failed")
    return {"e_h": dh-ds, "vector_total": total, "vector_dir": floor, "vector_h_dep": dep,
            "cross_direct": cross_direct, "decomposition_rel_error": abs(residual)/scale}


def probe(config, outdir, smoke=False):
    started=time.time()
    model, params, master, q_names, datasets, collate, info = load_context(config, outdir, probe=True)
    order = np.random.default_rng(config["data_seed"]).permutation(len(datasets[0]))
    ref_path = outdir / "references.json"
    allrefs = read_json(ref_path, {})
    raw = records(outdir / "raw_probe_metrics.jsonl")
    done = {(r["batch_id"],r["precision"],r["direction_seed"],r["h"]) for r in raw if r.get("status") == "complete"}
    failures = {(r["batch_id"],r["precision"],r["direction_seed"],r["h"]): r for r in raw if r.get("status") == "failed_final"}
    for bid in range(config["batches"]):
        ids = [int(i) for i in order[bid*16:(bid+1)*16]]
        write_json(outdir / f"batch_{bid}.json", {"indices": ids, "samples": [
            {"id": prior.stable_sample_id(datasets[0].samples[i],i), "data": datasets[0].samples[i].data} for i in ids]})
        batch = make_batch(datasets[0], ids, collate)
        if str(bid) not in allrefs:
            allrefs[str(bid)] = gradient_refs(model, params, master, batch, config)
            write_json(ref_path, allrefs)
        ref = allrefs[str(bid)]
        if bid == 0:
            frozen = calibrate_theory(model, params, master, q_names, batch, ref, config, outdir)
            if frozen is None:
                return {"status": "pending"}
        for precision in ("fp32", "fp16", "int8", "int4"):
            if all((bid,precision,r["direction_seed"],h) in done or (bid,precision,r["direction_seed"],h) in failures
                   for r in ref["refs"] for h in config["h_grid"]):
                continue
            model.to(dtype=torch.float32 if precision == "fp32" else torch.float16)
            restore(params, master)
            states = grids(master, q_names, precision)
            apply(params, master, states)
            base_loss = batch_loss(model, batch, config["microbatch"])[0]
            for ri, r in enumerate(ref["refs"]):
                seed = r["direction_seed"]
                for h in config["h_grid"]:
                    key = (bid,precision,seed,h)
                    if key in done or key in failures:
                        continue
                    base = {"model": config["model"], "task": config["task"], "checkpoint_step": 0,
                            "batch_id": bid, "precision": precision, "h": h, **r, "G2": ref["G2"], "d": info["d"]}
                    try:
                        apply(params, master, states, seed, h, 1)
                        lp = batch_loss(model, batch, config["microbatch"])[0]
                        apply(params, master, states, seed, h, -1)
                        lm = batch_loss(model, batch, config["microbatch"])[0]
                        dh = (lp-lm)/(2*h)
                        row = {**base, "d_h": dh, "loss_plus": lp, "loss_minus": lm, "loss_base": base_loss,
                               **vector_record(dh,r["d_star"],r["norm_u2"],ref["G2"]), "status": "complete"}
                        if bid == 0 and ri < 4:
                            row.update(geometry(master,states,seed,h,next(iter(params.values())).dtype))
                        append(outdir / "raw_probe_metrics.jsonl", row)
                        done.add(key)
                    except (FloatingPointError, torch.cuda.OutOfMemoryError) as exc:
                        row = {**base,"status": "failed_final", "error": repr(exc)}
                        append(outdir / "raw_probe_metrics.jsonl",row)
                        failures[key] = row
                        torch.cuda.empty_cache()
                    finally:
                        restore(params, master)
                    if STOP:
                        return {"status": "pending"}
                print(f"{config['task']} batch={bid} {precision} direction={ri+1}/{config['directions']}",flush=True)
            del states
    summary = {**config,"status": "complete", "num_complete":len(done), "num_failed":len(failures),
               "run_type": "smoke" if smoke else "probe_only", "source_path":str(outdir/"raw_probe_metrics.jsonl"),
               "runtime_sec":time.time()-started,"peak_gpu_memory_gib":torch.cuda.max_memory_allocated()/1024**3}
    if smoke:
        check=[r for r in records(outdir/"raw_probe_metrics.jsonl")
               if r.get("status")=="complete" and r["precision"]=="fp32" and r["h"]==1e-4]
        nmse=sum(r["e_h"]**2 for r in check)/(sum(r["d_star"]**2 for r in check)+1e-30)
        passed=len(check)==config["directions"] and nmse<.05 and not failures
        write_json(outdir/"clean_reference_check.json",{"fp32_h1e4_nmse":nmse,"n_directions":len(check),"passed":passed})
        if not passed:
            raise AssertionError(f"Clean reference smoke failed: nMSE={nmse}, failed points={len(failures)}")
    write_json(outdir / "summary.json",summary)
    return summary


def selftest(root):
    out = root / "selftests"
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(991)
    results = {}
    for bits in (4,8):
        w = torch.randn(7,259,device="cuda",dtype=torch.float16)*.03
        old, _ = rtn.compute_rtnclip_state("w",w,bits,128)
        fast, _ = rtn.compute_rtnclip_state("w",w,bits,128,collect_stats=False)
        assert torch.equal(old.scales,fast.scales) and torch.equal(old.alpha_idx,fast.alpha_idx)
        for h in TRAIN_H:
            u = torch.randn_like(w,dtype=torch.float32)
            assert torch.equal(rtn.quantize_with_state(w.float()+h*u,old),rtn.quantize_with_state(w.float()+h*u,fast))
        results[f"int{bits}_quantizer_equivalence"] = True
    master = {"w":torch.randn(7,259,device="cuda",dtype=torch.float16), "b":torch.randn(7,device="cuda",dtype=torch.float16)}
    first = dict(direction(master,99))
    assert all(torch.equal(first[n],u) for n,u in direction(master,99))
    params = {n:torch.nn.Parameter(w.clone()) for n,w in master.items()}
    apply(params,master,{},99,1e-3,1)
    apply(params,master,{},99,1e-3,-1)
    restore(params,master)
    assert all(torch.equal(params[n],w) for n,w in master.items())
    results["direction_replay_and_restore"] = True
    paired = PairedOrder(37,16,16)
    ids = [paired.at(i) for i in range(1,11)]
    resumed = PairedOrder(37,16,16)
    assert [resumed.at(i) for i in range(6,11)] == ids[5:]
    results["resumed_data_order"] = True
    u=torch.randn(64,17,dtype=torch.float64); g=torch.randn(17,dtype=torch.float64)
    ds=u@g; dh=ds+torch.randn(64,dtype=torch.float64)*.2
    measured=[]
    for i in range(64):
        r=vector_record(dh[i].item(),ds[i].item(),u[i].square().sum().item(),g.square().sum().item())
        assert abs(r["vector_total"]-(dh[i]*u[i]-g).square().sum().item()) < 1e-9
        measured.append(r)
    assert np.isclose(np.sum([(r["e_h"])**2 for r in measured])/ds.square().sum().item(),
                      ((dh-ds).square().sum()/ds.square().sum()).item())
    results["vector_identity_and_pooled_nmse"] = True
    write_json(out / "unit_results.json",results)
    print(results,flush=True)


@contextlib.contextmanager
def distributed_claim(root, run_id):
    """Lustre cross-node exclusion uses atomic mkdir, not node-local flock."""
    path=root/"locks"/(run_id+".claim")
    if os.environ.get("SLURM_ARRAY_JOB_ID"):
        job_id=os.environ["SLURM_ARRAY_JOB_ID"]+"_"+os.environ["SLURM_ARRAY_TASK_ID"]
    else:
        job_id=os.environ.get("SLURM_JOB_ID","")
    restart=int(os.environ.get("SLURM_RESTART_COUNT","0"))
    if path.exists():
        recovery=root/"locks/recovery.claim"
        acquired=False
        try:
            recovery.mkdir()
            acquired=True
            owner=read_json(path/"owner.json",{})
            stale=False
            if owner.get("job_id"):
                q=subprocess.run(["squeue","-h","-j",owner["job_id"],"-o","%T"],capture_output=True,text=True)
                stale=(q.returncode==0 and not q.stdout.strip()) or "Invalid job id" in q.stderr
                if owner["job_id"]==job_id and restart>owner.get("restart_count",0):
                    stale=True
            if stale and path.exists():
                archived=root/"locks"/(run_id+".stale."+uuid.uuid4().hex)
                path.rename(archived)
                append(root/"logs/claim_recovery.jsonl",{"run_id":run_id,"old_owner":owner,"time":stamp()})
        except FileExistsError:
            pass
        finally:
            if acquired:recovery.rmdir()
    try:
        path.mkdir()
    except FileExistsError:
        yield False
        return
    token=uuid.uuid4().hex
    write_json(path/"owner.json",{"token":token,"job_id":job_id,"restart_count":restart,
                                  "host":socket.gethostname(),"pid":os.getpid(),"time":stamp()})
    try:
        yield True
    finally:
        if read_json(path/"owner.json",{}).get("token")==token:
            (path/"owner.json").unlink()
            path.rmdir()


def worker(root, kind, once):
    jobs=read_json(root/"jobs.json")
    priority = [j for j in jobs if kind in ("all",j["kind"])]
    if kind == "train":
        priority.sort(key=lambda j: (j["train_seed"] != 16, ("sst-2","rte").index(j["task"]),
            ("fp16","int8","int4").index(j["precision"]), j["train_seed"],
            -1 if j["h"] == 1e-3 else TRAIN_H.index(j["h"])))
    if kind == "all":
        priority.sort(key=lambda j: 0 if j["kind"]=="probe" else 1)
    did_work=False
    for config in priority:
        if STOP:
            break
        outdir=root/("runs" if config["kind"]=="train" else "probes")/config["id"]
        outdir.mkdir(parents=True,exist_ok=True)
        with distributed_claim(root,config["id"]) as claimed:
            if not claimed:
                continue
            status=read_json(outdir/"status.json",{})
            if status.get("status") in ("complete","failed_final"):
                continue
            did_work=True
            attempts=len(records(outdir/"attempts.jsonl"))
            failures=len(records(outdir/"worker_failures.jsonl"))
            while failures < 3 and not STOP:
                attempts+=1
                append(outdir/"attempts.jsonl",{"attempt":attempts,"time":stamp(),"job_id":os.environ.get("SLURM_JOB_ID"),"host":socket.gethostname()})
                log=outdir/f"attempt_{attempts}.stdout.log"
                err=outdir/f"attempt_{attempts}.stderr.log"
                with log.open("a") as stdout,err.open("a") as stderr:
                    child=subprocess.Popen([sys.executable,"-u",str(Path(__file__).resolve()),"run","--root",str(root.resolve()),"--id",config["id"]],
                                           stdout=stdout,stderr=stderr)
                    print(f"launched {config['id']} pid={child.pid} log={log}",flush=True)
                    last_report=time.time()
                    while child.poll() is None:
                        if STOP:
                            child.send_signal(signal.SIGTERM)
                        time.sleep(2)
                        if time.time()-last_report>300:
                            safe_aggregate(root)
                            last_report=time.time()
                    rc=child.returncode
                safe_aggregate(root)
                if rc==0:
                    break
                if rc==75 or STOP:
                    write_json(outdir/"status.json",{"status":"pending","reason":"checkpointed interruption","updated_at":stamp()})
                    return 75
                failures+=1
                append(outdir/"worker_failures.jsonl",{"attempt":attempts,"returncode":rc,"time":stamp()})
                if failures>=3:
                    write_json(outdir/"status.json",{"status":"failed_final","attempts":attempts,"returncode":rc,"updated_at":stamp()})
                else:
                    print(f"retry {config['id']} returncode={rc}",flush=True)
            if once:
                return 0
    safe_aggregate(root)
    return 0


def safe_aggregate(root):
    """Reporting failures must not terminate an expensive child experiment."""
    try:
        aggregate(root)
        return True
    except Exception as exc:
        append(root / "logs/aggregation_failures.jsonl", {"time": stamp(),
            "error": repr(exc), "traceback": traceback.format_exc(), "pid": os.getpid()})
        print(f"Aggregation failed; experiments continue: {exc!r}", file=sys.stderr, flush=True)
        return False


def smoke_suite(root):
    jobs=read_json(root/"jobs.json")
    cases=[j for j in jobs if j["kind"]=="train" and j["h"]==1e-3 and j["train_seed"]==16]
    commands=[]
    for cfg in cases:
        # Include a real 100-step timing/memory test for the longest task + quantizer.
        steps=100 if cfg["task"]=="rte" and cfg["precision"]=="int4" else 2
        out=root/"smoke"/cfg["id"]
        summary=read_json(out/"summary.json",{})
        if summary.get("status")=="complete" and summary.get("steps_completed")==steps and (out/"loss_equivalence.json").exists():
            continue
        if out.exists():
            # Keep earlier partial tests, including failures, under a separate attempt directory.
            out.rename(out.with_name(out.name+"_previous_"+str(int(time.time()))))
        cmd=[sys.executable,"-u",str(Path(__file__).resolve()),"smoke","--root",str(root.resolve()),"--id",cfg["id"],"--steps",str(steps)]
        commands.append(cmd)
        with (root/"logs"/f"smoke_{cfg['id']}.stdout.log").open("a") as stdout, (root/"logs"/f"smoke_{cfg['id']}.stderr.log").open("a") as stderr:
            print(f"smoke {cfg['id']} steps={steps}",flush=True)
            subprocess.run(cmd,stdout=stdout,stderr=stderr,check=True)
    probe_cfg=next(j for j in jobs if j["kind"]=="probe" and j["task"]=="sst-2")
    cmd=[sys.executable,"-u",str(Path(__file__).resolve()),"smoke","--root",str(root.resolve()),"--id",probe_cfg["id"]]
    with (root/"logs/probe_smoke.stdout.log").open("a") as stdout, (root/"logs/probe_smoke.stderr.log").open("a") as stderr:
        subprocess.run(cmd,stdout=stdout,stderr=stderr,check=True)
    commands.append(cmd)
    write_json(root/"SMOKE_PASSED.json",{"time":stamp(),"commands":commands,"unit_results":read_json(root/"selftests/unit_results.json"),
               "script_sha256":sha(__file__),"quantizer_sha256":sha(ROOT/"tools/smoke_rtnclip_roberta_sst5.py")})


def aggregate(root):
    jobs=read_json(root/"jobs.json",[])
    manifest=[]; runs=[]; eval_rows=[]; probe_rows=[]; theory=[]; raw_probe=[]
    for cfg in jobs:
        out=root/("runs" if cfg["kind"]=="train" else "probes")/cfg["id"]
        status=read_json(out/"status.json",{"status":"pending"})
        manifest.append({**cfg,**status,"source_path":str(out)})
        summary=read_json(out/"summary.json")
        if summary:
            runs.append(summary)
        for ev in records(out/"eval.jsonl"):
            eval_rows.append({"id":cfg["id"],"task":cfg["task"],"precision":cfg["precision"],"h":cfg.get("h"),
                              "seed":cfg["train_seed"],**ev})
        raw=records(out/"raw_probe_metrics.jsonl")
        successful={}
        for r in raw:
            if r.get("status")=="complete":
                successful[(r["batch_id"],r["precision"],r["direction_seed"],r["h"])]=r
        groups={}
        for r in successful.values():
            groups.setdefault((r["precision"],r["h"],"pooled"),[]).append(r)
            groups.setdefault((r["precision"],r["h"],r["batch_id"]),[]).append(r)
        raw_probe.extend(successful.values())
        for (prec,h,bid),rows in groups.items():
            ds=np.array([r["d_star"] for r in rows]); dh=np.array([r["d_h"] for r in rows])
            floor=np.mean([r["vector_dir"] for r in rows]); dep=np.mean([r["vector_h_dep"] for r in rows])
            correlation=float(np.corrcoef(ds,dh)[0,1]) if ds.std()>0 and dh.std()>0 else None
            row={"task":cfg["task"],"precision":prec,"batch_id":bid,"h":h,"n_pairs":len(rows),"n_batches":len(set(r["batch_id"] for r in rows)),
                 "true_directional_nmse":float(np.sum((dh-ds)**2)/(np.sum(ds**2)+1e-30)),
                 "rho_emp":float(dep/floor),"V_dir":float(floor),"V_h_dep":float(dep),
                 "V_total":float(np.mean([r["vector_total"] for r in rows])),
                 "V_cross":float(np.mean([r["cross_direct"] for r in rows])),
                 "directional_corr":correlation,"sign_agreement":float(np.mean(np.sign(dh)==np.sign(ds))),
                 "dh_std":float(dh.std()),"dh_zero_fraction":float(np.mean(dh==0)),"source_path":str(out/"raw_probe_metrics.jsonl")}
            # Resample direction clusters, retaining the same direction across batches/h.
            seeds=sorted(set(r["direction_seed"] for r in rows))
            clusters=[np.array([sum((r["d_h"]-r["d_star"])**2 for r in rows if r["direction_seed"]==s),
                               sum(r["d_star"]**2 for r in rows if r["direction_seed"]==s),
                               sum(r["vector_h_dep"] for r in rows if r["direction_seed"]==s),
                               sum(r["vector_dir"] for r in rows if r["direction_seed"]==s)]) for s in seeds]
            if len(seeds)>=2:
                rng=np.random.default_rng(8216)
                indices=rng.integers(0,len(seeds),size=(500,len(seeds)))
                pooled=np.asarray(clusters)[indices].sum(1)
                for field,values in (("nmse",pooled[:,0]/np.maximum(pooled[:,1],1e-30)),
                                     ("rho",pooled[:,2]/np.maximum(pooled[:,3],1e-30))):
                    row[field+"_ci_low"],row[field+"_ci_high"]=[float(v) for v in np.quantile(values,[.025,.975])]
            for field in ("active_fraction","visible_cosine","visible_norm_ratio","p_clip"):
                vals=[r[field] for r in rows if r.get(field) is not None]
                row[field]=float(np.mean(vals)) if vals else None
            probe_rows.append(row)
        frozen=read_json(out/"PREDICTED_WINDOWS_FROZEN.json")
        if frozen:
            for prec,p in frozen["precisions"].items():
                theory.append({"task":cfg["task"],"precision":prec,"G":frozen["G"],"L":frozen["L"],"d":frozen["d"],
                               **p,"source_path":str(out/"PREDICTED_WINDOWS_FROZEN.json")})
    csv_write(root/"run_manifest.csv",manifest)
    csv_write(root/"summaries/training_runs.csv",[r for r in runs if r["kind"]=="train"])
    csv_write(root/"summaries/eval_curves.csv",eval_rows)
    csv_write(root/"summaries/probe_mse_by_h.csv",[r for r in probe_rows if r["batch_id"]==0])
    csv_write(root/"summaries/probe_mse_by_batch_h.csv",[r for r in probe_rows if r["batch_id"]!="pooled"])
    csv_write(root/"summaries/probe_mse_pooled_by_h.csv",[r for r in probe_rows if r["batch_id"]=="pooled"])
    csv_write(root/"summaries/raw_probe_metrics.csv",raw_probe)
    csv_write(root/"summaries/theory_windows.csv",theory)
    counts={s:sum(r["status"]==s for r in manifest) for s in ("pending","running","retry","complete","failed_final")}
    write_json(root/"summaries/progress.json",{"updated_at":stamp(),"counts":counts})
    completed=[r for r in runs if r["kind"]=="train" and r["status"]=="complete"]
    if completed or probe_rows:
        make_plots(root,completed,[r for r in probe_rows if r["batch_id"]==0],theory)
    (root/"STATUS.md").write_text(f"# Live status\n\nUpdated {stamp()}\n\n{json.dumps(counts,indent=2)}\n\n"
        "Pending and running jobs are not results. Primary training metric is final full dev accuracy.\n")
    report=["# OPT-2.7B radius extension results", "", f"Updated: {stamp()}", "",
            f"Complete jobs: {counts['complete']}/{len(jobs)}. Failed-final: {counts['failed_final']}.",
            "Results remain provisional until each paired seed/h matrix is complete.", "",
            "The full-data SST-2/RTE experiments use official MeZO prompts. They are an architecture/precision extension, not an exact original benchmark reproduction.",
            "Final dev accuracy is primary. Best and initial dev accuracy are saved separately. A chance-level plateau is not evidence of successful training.", ""]
    for r in completed:
        report.append(f"- {r['task']} {r['precision']} h={r['h']:g} seed={r['train_seed']}: final={r['final_eval_acc']:.4f}, initial={r['initial_eval_acc']:.4f}, best={r['best_eval_acc']:.4f} (20k full).")
    report.extend(["", "Probe metrics use clean FP32 true gradients at the exact common FP16 pretrained center. Scalar nMSE, vector V_total and radius ratio rho are distinct columns.",
                   "No claim of a U-shape or an accuracy plateau is made from incomplete curves. Failed points and missing certificates are retained.",
                   "Direction-cluster bootstrap intervals describe direction variation conditional on the fixed batches, not uncertainty over the dataset."])
    (root/"RESULTS_REPORT.md").write_text("\n".join(report)+"\n")


def make_plots(root,runs,probes,theory):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    def save(fig,name):
        for ext in ("pdf","png"):
            tmp=root/"figures"/f"{name}.{os.getpid()}.{ext}"
            fig.savefig(tmp,bbox_inches="tight",dpi=160)
            os.replace(tmp,root/"figures"/f"{name}.{ext}")
        plt.close(fig)
    if runs:
        fig,axes=plt.subplots(1,2,figsize=(10,4))
        groups={}; summaries=[]
        for r in runs:groups.setdefault((r["task"],r["precision"],r["h"]),[]).append(r)
        for (task,precision,h),rows in groups.items():
            values=[r["final_eval_acc"] for r in rows]
            summaries.append({"task":task,"precision":precision,"h":h,"n_seeds":len(rows),
                              "mean_final_acc":float(np.mean(values)),"std_final_acc":float(np.std(values,ddof=1)) if len(values)>1 else None})
        csv_write(root/"summaries/accuracy_by_h.csv",summaries)
        good=[]
        for task,ax in zip(("sst-2","rte"),axes):
            for precision in ("fp16","int8","int4"):
                rr=sorted([r for r in summaries if r["task"]==task and r["precision"]==precision],key=lambda r:r["h"])
                if not rr:continue
                ax.errorbar([r["h"] for r in rr],[r["mean_final_acc"] for r in rr],
                            yerr=[r["std_final_acc"] or 0 for r in rr],marker="o",label=precision)
                best=max(r["mean_final_acc"] for r in rr)
                good.append({"task":task,"precision":precision,"threshold":.01,"best_mean_final_acc":best,
                             "good_h_values":[r["h"] for r in rr if r["mean_final_acc"]>=best-.01],
                             "status":"complete_grid" if len(rr)==6 else "partial_grid"})
            ax.set(xscale="log",xlabel="h",ylabel="Final dev accuracy",title=task)
            ax.axvline(1e-3,color="gray",ls="--");ax.legend()
        csv_write(root/"summaries/empirical_accuracy_good_sets.csv",good)
        save(fig,"accuracy_vs_h")
        # Balanced paired seed design; do not estimate variance components with missing cells.
        hp=[r for r in runs if r["task"]=="sst-2" and r["precision"]=="fp16"]
        cells={(r["h"],r["train_seed"]):r["final_eval_acc"] for r in hp}
        if all((h,s) in cells for h in TRAIN_H for s in (16,32,64)):
            matrix=np.array([[cells[h,s] for s in (16,32,64)] for h in TRAIN_H])
            mean=matrix.mean(); a=matrix.mean(1)-mean; b=matrix.mean(0)-mean
            resid=matrix-mean-a[:,None]-b[None,:]
            csv_write(root/"summaries/paired_variance.csv",[{"component":"h_policy","descriptive_variance":float(np.mean(a*a))},
                {"component":"seed","descriptive_variance":float(np.mean(b*b))},
                {"component":"interaction_residual","descriptive_variance":float(np.mean(resid*resid))}])
            fig,ax=plt.subplots(figsize=(6,4))
            for idx,seed in enumerate((16,32,64)):
                ax.semilogx(TRAIN_H,matrix[:,idx],"o-",label=f"seed {seed}")
            ax.set(xlabel="h",ylabel="Final dev accuracy",title="SST-2 FP16 paired seeds")
            ax.legend();save(fig,"fp16_paired_seed_lines")
            contrasts=[{"seed":s,"h":h,"delta_vs_default":cells[h,s]-cells[1e-3,s]} for h in TRAIN_H for s in (16,32,64)]
            csv_write(root/"summaries/paired_contrasts.csv",contrasts)
            core=matrix[:4]
            core_mean=core.mean();ca=core.mean(1)-core_mean;cb=core.mean(0)-core_mean
            cr=core-core_mean-ca[:,None]-cb[None,:]
            csv_write(root/"summaries/core_h_paired_variance.csv",[{"h_grid":TRAIN_H[:4],"component":"h_policy","descriptive_variance":float(np.mean(ca*ca))},
                {"h_grid":TRAIN_H[:4],"component":"seed","descriptive_variance":float(np.mean(cb*cb))},
                {"h_grid":TRAIN_H[:4],"component":"interaction_residual","descriptive_variance":float(np.mean(cr*cr))}])
    if probes:
        for field,label in (("true_directional_nmse","True directional nMSE"),("rho_emp","Vector error ratio rho")):
            fig,axes=plt.subplots(1,2,figsize=(10,4))
            for task,ax in zip(("sst-2","rte"),axes):
                for precision in ("fp32","fp16","int8","int4"):
                    rr=sorted([r for r in probes if r["task"]==task and r["precision"]==precision],key=lambda r:r["h"])
                    if rr:ax.loglog([r["h"] for r in rr],[max(r[field],1e-20) for r in rr],"o-",label=precision)
                task_rows=[r for r in probes if r["task"]==task]
                title=task if task_rows and all(r["n_pairs"]>=64 for r in task_rows) else task+" (partial probe)"
                ax.axvline(1e-3,color="gray",ls="--");ax.set(xscale="log",xlabel="h",ylabel=label,title=title)
                if task_rows:ax.legend()
                else:ax.text(.5,.5,"Pending: no measurements yet",ha="center",transform=ax.transAxes)
                if field=="rho_emp":ax.axhline(1,color="black",ls=":")
            save(fig,field+"_vs_h")
        fig,axes=plt.subplots(1,2,figsize=(10,4))
        for task,ax in zip(("sst-2","rte"),axes):
            for precision,marker in (("int8","s"),("int4","o")):
                rr=sorted([r for r in probes if r["task"]==task and r["precision"]==precision],key=lambda r:r["h"])
                if not rr:continue
                line=ax.loglog([r["h"] for r in rr],[r["rho_emp"] for r in rr],marker+"-",label=precision+" measured")[0]
                pred=next((r for r in theory if r["task"]==task and r["precision"]==precision),None)
                if pred and "h_q" in pred:
                    xx=np.geomspace(min(r["h"] for r in rr),max(r["h"] for r in rr),200)
                    yy=pred["c_d"]*((pred["h_q"]/xx)**2+(xx/pred["h_loc"])**2)
                    ax.loglog(xx,yy,"--",color=line.get_color(),label=precision+" plug-in theory")
            ax.axhline(1,color="black",ls=":");ax.axvline(1e-3,color="gray",ls="--")
            ax.set(xscale="log",xlabel="h",ylabel="Vector error ratio rho",title=task)
            if ax.get_legend_handles_labels()[0]:ax.legend(fontsize=8)
            else:ax.text(.5,.5,"Pending: no low-bit measurements yet",ha="center",transform=ax.transAxes)
        save(fig,"theory_vs_empirical_rho")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["prepare", "run", "smoke", "smoke-suite", "aggregate", "worker", "selftest"])
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--id")
    parser.add_argument("--kind", choices=["train", "probe", "all"], default="train")
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGUSR1):
        signal.signal(sig, request_stop)
    if args.command == "prepare":
        prepare(args.root)
        return 0
    if args.command == "selftest":
        selftest(args.root)
        return 0
    if args.command == "smoke-suite":
        smoke_suite(args.root)
        return 0
    if args.command == "aggregate":
        aggregate(args.root)
        return 0
    if args.command == "worker":
        return worker(args.root, args.kind, args.once)
    jobs = read_json(args.root / "jobs.json")
    config = next(j for j in jobs if j["id"] == args.id)
    smoke = args.command == "smoke"
    if smoke:
        config = {**config, "steps": args.steps, "eval_every": args.steps, "checkpoint_every": args.steps}
        if config["kind"] == "probe":
            config.update(batches=1, directions=2, l_directions=2, audit_seeds=config["audit_seeds"][:2],
                          l_seeds=config["l_seeds"][:2], h_grid=[1e-4,1e-3], smoke=True)
        folder = "smoke"
    else:
        folder = "runs" if config["kind"] == "train" else "probes"
    return execute(args.root, config, args.root / folder / config["id"], smoke)


if __name__ == "__main__":
    raise SystemExit(main())
