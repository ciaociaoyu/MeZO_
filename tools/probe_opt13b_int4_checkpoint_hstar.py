#!/usr/bin/env python
"""Re-estimate OPT INT4 h-star components from saved FP16-master checkpoints."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import platform
import socket
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Sequence

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
TOOLS_DIR = REPO_ROOT / "tools"
LARGE_MODELS_DIR = REPO_ROOT / "large_models"
for path in (TOOLS_DIR, LARGE_MODELS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import probe_opt13b_int4_task_grid as optprobe  # noqa: E402
import train_opt13b_int4_dense_smoke as opttrain  # noqa: E402


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: List[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def append_jsonl(path: Path, row: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, sort_keys=True, default=str) + "\n")


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(REPO_ROOT), text=True).strip()
    except Exception:
        return ""


def env_info() -> Dict[str, object]:
    out = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "hostname": socket.gethostname(),
        "python": sys.executable,
        "python_version": platform.python_version(),
        "conda_env": os.environ.get("CONDA_DEFAULT_ENV", ""),
        "git_commit": git_commit(),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
    }
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(torch.cuda.current_device())
        out["gpu_name"] = props.name
        out["gpu_total_memory_mb"] = int(props.total_memory / 1024 / 1024)
    return out


def parse_float_list(raw: Sequence[str]) -> List[float]:
    vals: List[float] = []
    for item in raw:
        for part in str(item).replace(",", " ").split():
            vals.append(float(part))
    return vals


def load_checkpoint_master(checkpoint_path: Path, master: Dict[str, torch.Tensor], device: torch.device) -> int:
    state = torch.load(checkpoint_path, map_location="cpu")
    saved = state["master"]
    for name, tensor in saved.items():
        if name in master:
            master[name] = tensor.to(device=device, dtype=master[name].dtype)
    return int(state.get("step", 0))


def refresh_master32(master: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {name: tensor.detach().clone().to(dtype=torch.float32) for name, tensor in master.items()}


def finite_float(value: object) -> float | None:
    try:
        out = float(value)  # type: ignore[arg-type]
    except Exception:
        return None
    return out if math.isfinite(out) else None


def mean(vals: List[float]) -> float:
    return sum(vals) / max(len(vals), 1)


def corr(xs: List[float], ys: List[float]) -> float | None:
    pairs = [(x, y) for x, y in zip(xs, ys) if math.isfinite(x) and math.isfinite(y)]
    if len(pairs) < 2:
        return None
    mx = mean([x for x, _ in pairs])
    my = mean([y for _, y in pairs])
    vx = sum((x - mx) ** 2 for x, _ in pairs)
    vy = sum((y - my) ** 2 for _, y in pairs)
    if vx <= 0 or vy <= 0:
        return None
    return sum((x - mx) * (y - my) for x, y in pairs) / math.sqrt(vx * vy)


def summarize_records(records: List[Dict[str, object]], h_values: Sequence[float]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for h in h_values:
        group = [r for r in records if abs(float(r["h"]) - float(h)) <= 1e-15]
        dh = [float(r["d_h"]) for r in group]
        dt = [float(r["d_true"]) for r in group]
        mse = mean([(a - b) ** 2 for a, b in zip(dh, dt)])
        ref = mean([b * b for b in dt])
        rows.append(
            {
                "h": float(h),
                "k_dirs": len(group),
                "default_fd_true_nmse": mse / max(ref, 1e-30),
                "default_corr_fd_true": corr(dh, dt),
                "d_h_abs_mean": mean([abs(x) for x in dh]),
                "d_true_abs_mean": mean([abs(x) for x in dt]),
                "d_h_rms": math.sqrt(mean([x * x for x in dh])),
                "d_true_rms": math.sqrt(mean([x * x for x in dt])),
            }
        )
    return rows


def formula_h(delta: float, g: float, lval: float, d: int) -> float:
    if not (delta > 0 and g > 0 and lval > 0 and d > 0):
        return float("nan")
    return math.sqrt(delta * g / (4.0 * lval * math.sqrt(float(d) * float(d + 2))))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--checkpoint_tags", nargs="+", default=["best_acc", "final"])
    parser.add_argument("--task", default="sst-2", choices=["sst-2", "sst-5", "rte", "mnli", "trec"])
    parser.add_argument("--model_id", default="facebook/opt-1.3b")
    parser.add_argument("--task_path", choices=["mezo_option"], default="mezo_option")
    parser.add_argument("--dataset_mode", choices=["full", "fewshot", "auto"], default="full")
    parser.add_argument("--num_train", type=int, default=-1)
    parser.add_argument("--num_k", type=int, default=16)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--eval_batch_size", type=int, default=16)
    parser.add_argument("--eval_samples", type=int, default=0)
    parser.add_argument("--max_seq_len", type=int, default=128)
    parser.add_argument("--bitwidth", type=int, default=4)
    parser.add_argument("--group_size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=16)
    parser.add_argument("--data_seed", type=int, default=16)
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument("--h_values", nargs="+", default=["1e-4", "3e-4", "1e-3"])
    parser.add_argument("--hstar_l_grid", nargs="+", default=["1e-5", "3e-5", "1e-4", "3e-4", "1e-3", "3e-3", "1e-2"])
    parser.add_argument("--k_dirs", type=int, default=16)
    parser.add_argument("--m_l", type=int, default=4)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for OPT-1.3B checkpoint h-star probe.")
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    h_values = parse_float_list(args.h_values)
    hstar_l_grid = parse_float_list(args.hstar_l_grid)
    write_json(output_root / "run_config.json", {**vars(args), "h_values": h_values, "hstar_l_grid": hstar_l_grid})
    write_json(output_root / "env.json", env_info())

    device = torch.device("cuda")
    model, tokenizer = optprobe.load_model_and_tokenizer(args, device)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    if model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.pad_token_id
    opttrain.patch_mezo_option_loss(model)
    params = optprobe.params_map(model)
    q_names = optprobe.linear_weight_names(model, params)
    base_master = optprobe.make_master(params, torch.float16)
    _task, train_loader, _eval_loader, train_count, _eval_count = opttrain.load_mezo_option_loaders(args, tokenizer)
    batch = opttrain.prepare_batch(next(iter(train_loader)), device)

    summary_rows: List[Dict[str, object]] = []
    all_l_rows: List[Dict[str, object]] = []
    all_probe_rows: List[Dict[str, object]] = []
    run_dir = Path(args.run_dir)
    for tag in args.checkpoint_tags:
        master = {name: tensor.detach().clone() for name, tensor in base_master.items()}
        checkpoint_path = run_dir / "checkpoints" / tag / "master.pt"
        step = load_checkpoint_master(checkpoint_path, master, device)
        optprobe.restore_master(params, master)
        states, _q_rows = optprobe.refresh_states(master, q_names, int(args.bitwidth), int(args.group_size))
        delta_stats = optprobe.weighted_delta_with_optional_masks(states, None)
        clean_loss = optprobe.compute_true_gradient(model, params, master, batch, states=None)
        records: List[Dict[str, object]] = []
        perturb_names = list(master.keys())
        for direction_id in range(int(args.k_dirs)):
            directions = optprobe.sample_direction(master, perturb_names, int(args.seed) + direction_id * 1009 + 5000, masks=None)
            d_true = optprobe.grad_dot_direction(params, directions)
            for h in h_values:
                loss_plus, loss_minus, d_h = optprobe.finite_difference(model, params, master, batch, states, directions, float(h))
                row = {
                    "checkpoint_tag": tag,
                    "checkpoint_step": step,
                    "direction_id": direction_id,
                    "h": float(h),
                    "loss_plus": loss_plus,
                    "loss_minus": loss_minus,
                    "d_h": d_h,
                    "d_true": d_true,
                    "clean_loss": clean_loss,
                }
                records.append(row)
                all_probe_rows.append(row)
                append_jsonl(output_root / "probe_records.jsonl", row)
            del directions
        probe_summary = summarize_records(records, h_values)
        for row in probe_summary:
            row.update({"checkpoint_tag": tag, "checkpoint_step": step})

        clean_master32 = refresh_master32(master)
        clean_selected, clean_status, clean_rows = optprobe.clean_second_diff_l(
            model,
            params,
            clean_master32,
            batch,
            perturb_names,
            None,
            seed_base=int(args.seed),
            h2_grid=hstar_l_grid,
            m_l=int(args.m_l),
        )
        optprobe.restore_master(params, master)
        states, _ = optprobe.refresh_states(master, q_names, int(args.bitwidth), int(args.group_size))
        low_selected, low_status, low_rows = optprobe.lowbit_second_diff_l(
            model,
            params,
            master,
            batch,
            perturb_names,
            None,
            states,
            seed_base=int(args.seed),
            h2_grid=hstar_l_grid,
            m_l=int(args.m_l),
        )
        for kind, rows, selected, status in (
            ("clean32", clean_rows, clean_selected, clean_status),
            ("lowbit_rtnclip", low_rows, low_selected, low_status),
        ):
            for row in rows:
                new = dict(row)
                new.update(
                    {
                        "checkpoint_tag": tag,
                        "checkpoint_step": step,
                        "kind": kind,
                        "selected_h2": selected.get("h2"),
                        "selected_lambda_q90": selected.get("lambda_q90"),
                        "selection_status": status,
                    }
                )
                all_l_rows.append(new)

        by_h = {float(r["h"]): r for r in probe_summary}
        lowbit_g_vals = [
            math.sqrt(math.pi / 2.0) * float(by_h[h]["d_h_abs_mean"])
            for h in (1e-4, 3e-4, 1e-3)
            if h in by_h and finite_float(by_h[h].get("d_h_abs_mean")) is not None
        ]
        clean_g_vals = [
            math.sqrt(math.pi / 2.0) * float(by_h[h]["d_true_abs_mean"])
            for h in (1e-4, 3e-4, 1e-3)
            if h in by_h and finite_float(by_h[h].get("d_true_abs_mean")) is not None
        ]
        lowbit_g = sorted(lowbit_g_vals)[len(lowbit_g_vals) // 2] if lowbit_g_vals else float("nan")
        clean_g = sorted(clean_g_vals)[len(clean_g_vals) // 2] if clean_g_vals else float("nan")
        delta = float(delta_stats["delta_int4_rtnclip_scale_rms"]) / math.sqrt(6.0)
        d_trainable = sum(int(t.numel()) for t in master.values())
        l_clean = float(clean_selected.get("lambda_q90", float("nan")))
        l_low = float(low_selected.get("lambda_q90", float("nan")))
        summary = {
            "checkpoint_tag": tag,
            "checkpoint_step": step,
            "clean_loss": clean_loss,
            "Delta_scale_rms_over_sqrt6": delta,
            "delta_int4_rtnclip_scale_rms": delta_stats["delta_int4_rtnclip_scale_rms"],
            "G_clean_abs": clean_g,
            "G_lowbit_abs_median_1e-4_3e-4_1e-3": lowbit_g,
            "L_clean_q90": l_clean,
            "L_clean_h2": clean_selected.get("h2"),
            "L_clean_status": clean_status,
            "L_lowbit_q90": l_low,
            "L_lowbit_h2": low_selected.get("h2"),
            "L_lowbit_status": low_status,
            "d_trainable": d_trainable,
            "hstar_cleanG_cleanL": formula_h(delta, clean_g, l_clean, d_trainable),
            "hstar_lowbitG_cleanL": formula_h(delta, lowbit_g, l_clean, d_trainable),
            "hstar_cleanG_lowbitL": formula_h(delta, clean_g, l_low, d_trainable),
            "hstar_lowbitG_lowbitL": formula_h(delta, lowbit_g, l_low, d_trainable),
        }
        for row in probe_summary:
            prefix = f"h{row['h']:g}".replace(".", "p")
            summary[f"{prefix}_nmse"] = row["default_fd_true_nmse"]
            summary[f"{prefix}_corr"] = row["default_corr_fd_true"]
            summary[f"{prefix}_d_h_abs_mean"] = row["d_h_abs_mean"]
            summary[f"{prefix}_d_true_abs_mean"] = row["d_true_abs_mean"]
        summary_rows.append(summary)
        print(
            f"{tag}@{step}: cleanG={clean_g:.6g} lowG={lowbit_g:.6g} "
            f"cleanL={l_clean:.6g} lowL={l_low:.6g} "
            f"h_lowG_cleanL={summary['hstar_lowbitG_cleanL']:.6g}",
            flush=True,
        )
        write_csv(output_root / "checkpoint_hstar_summary.csv", summary_rows)
        write_csv(output_root / "L_candidates_by_checkpoint.csv", all_l_rows)
        write_csv(output_root / "probe_summary_by_checkpoint.csv", probe_summary)
        del master, states
        torch.cuda.empty_cache()

    write_csv(output_root / "checkpoint_hstar_summary.csv", summary_rows)
    write_csv(output_root / "L_candidates_by_checkpoint.csv", all_l_rows)
    write_json(
        output_root / "run_summary.json",
        {
            "status": "complete",
            "rows": len(summary_rows),
            "train_sample_count": train_count,
            "summary_csv": str(output_root / "checkpoint_hstar_summary.csv"),
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
