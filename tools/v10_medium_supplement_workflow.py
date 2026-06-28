#!/usr/bin/env python3
"""Prepare the V10 medium supplement probe/training workflow.

This script is intentionally orchestration-only. It does not change the
training semantics, quantizer, sparse mask convention, prefix path, or h
selection rules. It reuses the V10 seed-fixed family where possible and only
launches missing seeds.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import subprocess
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATE = os.environ.get("V10_MEDIUM_DATE") or "20260627"
OUT_DEFAULT = ROOT / f"v10_medium_supplement_results_{DATE}"

V10_AUDIT = ROOT / "v10_supplement_results_20260626" / "v10_table_values_audit.csv"
V10_VIS = ROOT / "v10_supplement_results_20260626" / "sparse_prefix_true_mse_visibility.csv"
V10_MISMATCH = ROOT / "v10_supplement_results_20260626" / "probe_update_mismatch_diagnostics.csv"
LOWBIT_FAMILY = ROOT / "outputs/int4_sparse_prefix_seedfixed_int4fd_20k_20260523_171841"
LOWBIT_SUMMARY = LOWBIT_FAMILY / "int4_hsearch_summary.csv"
HP_SOURCE = ROOT / "experiments/main_latest/mezo/roberta-large/sst5/fp32_fp16_h_sweep_11h_seed16_bs64_ckpt1k_20260517"
HP_PROBE = HP_SOURCE / "plots" / "plot_probe_vs_h.csv"


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return "unknown"


def write_csv(path: Path, rows: list[dict], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        keys: list[str] = []
        for row in rows:
            for key in row:
                if key not in keys:
                    keys.append(key)
        fieldnames = keys
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def h_label(h: float) -> str:
    known = {
        1e-5: "1e-5",
        1e-4: "1e-4",
        3e-4: "3e-4",
        1e-3: "1e-3",
        3e-3: "3e-3",
        1e-2: "1e-2",
    }
    for value, label in known.items():
        if abs(float(h) - value) <= max(abs(value) * 1e-9, 1e-15):
            return label
    return f"{float(h):.14g}".replace(".", "p").replace("-", "m")


def policy_run_name(policy: str) -> str:
    return {
        "fixed-small": "fixed_small",
        "fixed_small": "fixed_small",
        "default": "mezo_default",
        "mezo_default": "mezo_default",
        "reference": "hstar",
        "hstar_cleanGL": "hstar_cleanGL",
        "hstar_lowbitL": "hstar_lowbitL",
    }.get(policy, policy)


def load_v10_rows() -> pd.DataFrame:
    if not V10_AUDIT.exists():
        raise FileNotFoundError(V10_AUDIT)
    df = pd.read_csv(V10_AUDIT)
    df["task"] = df["task"].astype(str).str.lower()
    df["mode"] = df["mode"].astype(str)
    df["raw_h_policy"] = df["raw_h_policy"].astype(str)
    return df


def make_audit(out: Path, v10: pd.DataFrame) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    phase1 = out / "phase1_probe"
    raw = out / "raw_runs"
    ag = out / "aggregates"
    fig = out / "figures"
    scripts = out / "scripts"
    manifests = out / "manifests"
    jobs = out / "jobs"
    for p in [phase1, raw, ag, fig, scripts, manifests, jobs]:
        p.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    # Exact config names and h_ref sources requested by the user.
    configs = [
        ("high_precision_plateau", "sst-5", "fp32", "dense", "existing latest_main FP32 seed16 sweep", str(HP_SOURCE)),
        ("prefix_int4", "sst-2", "int4", "prefix", "hstar_cleanGL", str(V10_AUDIT)),
        ("prefix_int4", "trec", "int4", "prefix", "hstar_cleanGL", str(V10_AUDIT)),
        ("prefix_int4_optional", "sst-5", "int4", "prefix", "hstar_cleanGL", str(V10_AUDIT)),
        ("sparse_p0p1_int4", "sst-5", "int4", "sparse_p0p1", "hstar_lowbitL", str(V10_AUDIT)),
        ("sparse_p0p1_int4", "rte", "int4", "sparse_p0p1", "hstar_lowbitL", str(V10_AUDIT)),
        ("sparse_p0p1_int4_optional", "sst-2", "int4", "sparse_p0p1", "hstar_lowbitL", str(V10_AUDIT)),
    ]
    for family, task, precision, mode, ref_policy, source in configs:
        if family == "high_precision_plateau":
            match = pd.DataFrame()
            h_ref = math.nan
            run_name = "fp32 high-precision plateau manifests from latest_main"
        else:
            match = v10[(v10["task"] == task) & (v10["mode"] == mode)]
            if ref_policy.startswith("hstar"):
                match = match[match["raw_h_policy"] == ref_policy]
            h_ref = float(match["h_value"].iloc[0]) if not match.empty else math.nan
            run_name = match["run_name"].iloc[0] if not match.empty and "run_name" in match else ""
        rows.append(
            {
                "family": family,
                "task": task,
                "precision": precision,
                "mode": mode,
                "reference_policy": ref_policy,
                "h_ref": h_ref,
                "source_path": source,
                "run_name_or_config": run_name,
                "comparable_v10_family": bool(not match.empty) if family != "high_precision_plateau" else True,
                "notes": "V10 seed-fixed family; no legacy highest-abs sparse mask." if "int4" in family else "FP32 high-precision plateau chosen from existing main latest sweep.",
            }
        )
    write_csv(out / "config_audit.csv", rows)

    audit_md = [
        "# V10 Medium Supplement Config Audit",
        "",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        f"Git commit: `{git_commit()}`",
        "",
        "## Decisions",
        "",
        "- High precision plateau uses FP32 RoBERTa-large/SST-5 from the existing `latest_main` family.",
        "- Sparse p=0.1 INT4 uses the seed-fixed task-gradient mask family: `outputs/int4_sparse_prefix_seedfixed_int4fd_20k_20260523_171841`.",
        "- Prefix INT4 uses the same seed-fixed prefix-quantized family.",
        "- Required sparse reference policy is `hstar_lowbitL`; required prefix reference policy is `hstar_cleanGL`, as in V10 tables.",
        "- Prefix RTE is not used for required runs because the V10 audit marks it incomplete/not comparable.",
        "- Sparse TREC is not used for required runs because the prompt restricts required sparse tasks to SST-5/RTE and warns not to use sparse TREC unless fully audited.",
        "- Existing seed16 full runs are reused; newly submitted runs fill missing seeds only.",
        "- Direction-stream pairing is limited by existing runners. Base train/data seeds are paired by h; the low-bit runner's internal direction seed includes h, so exact identical direction streams across h are not guaranteed.",
        "",
        "## Config Rows",
        "",
    ]
    audit_md.extend(["```", pd.DataFrame(rows).to_string(index=False), "```"])
    (out / "README.md").write_text("\n".join(audit_md) + "\n", encoding="utf-8")

    return {
        "phase1": phase1,
        "raw": raw,
        "aggregates": ag,
        "figures": fig,
        "scripts": scripts,
        "manifests": manifests,
        "jobs": jobs,
    }


def _plot_metric(df: pd.DataFrame, out: Path, metric: str, ylabel: str) -> None:
    import matplotlib.pyplot as plt

    cols = ["model", "task", "precision", "mode", "quantizer"]
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    plotted = False
    for key, group in df.dropna(subset=[metric]).groupby(cols, dropna=False):
        group = group.sort_values("h")
        if group.empty:
            continue
        label = "/".join(str(x) for x in key if str(x) and str(x) != "nan")
        ax.plot(group["h"], group[metric], marker="o", linewidth=1.5, label=label[:80])
        plotted = True
    ax.set_xscale("log")
    ax.set_xlabel("h")
    ax.set_ylabel(ylabel)
    ax.grid(True, which="both", alpha=0.25)
    if plotted:
        ax.legend(fontsize=7, loc="best")
    else:
        ax.text(0.5, 0.5, f"No finite {metric}", ha="center", va="center", transform=ax.transAxes)
    out.parent.mkdir(parents=True, exist_ok=True)
    for ext in ["pdf", "png"]:
        fig.savefig(out.with_suffix(f".{ext}"), bbox_inches="tight", dpi=180)
    plt.close(fig)


def build_phase1_probe(out: Path, dirs: dict) -> pd.DataFrame:
    rows: list[dict] = []
    if V10_VIS.exists():
        vis = pd.read_csv(V10_VIS)
        for _, r in vis.iterrows():
            mode = str(r.get("mode", ""))
            task = str(r.get("task", "")).lower()
            # Keep requested mechanism rows and useful prefix fallback rows.
            keep = False
            if mode == "sparse_p0p1" and task in {"sst-5", "rte", "sst-2", "mnli"}:
                keep = True
            if mode in {"prefix_int4", "prefix_fp32_or_mixed"} and task in {"sst-2", "trec", "sst-5", "mnli"}:
                keep = True
            if not keep:
                continue
            rows.append(
                {
                    "model": r.get("model", "roberta-large"),
                    "task": task,
                    "precision": r.get("precision", ""),
                    "mode": mode,
                    "quantizer": r.get("quantizer", ""),
                    "checkpoint": "existing_probe_checkpoint_or_init",
                    "batch_source": "existing_probe_batch",
                    "h": float(r.get("h")),
                    "h_policy": r.get("h_policy", ""),
                    "true_directional_nmse": r.get("true_directional_nmse"),
                    "corr_dq_dstar": r.get("directional_corr"),
                    "sign_agreement": r.get("sign_agreement"),
                    "active_frac": r.get("active_frac"),
                    "norm_ratio": r.get("norm_ratio"),
                    "visible_direction_cos": r.get("visible_direction_cos"),
                    "d_Q_mean": np.nan,
                    "d_Q_std": np.nan,
                    "d_star_mean": np.nan,
                    "d_star_std": np.nan,
                    "n_directions": r.get("n_directions"),
                    "direction_seed": "existing",
                    "source_path": r.get("source_path"),
                    "notes": r.get("notes", ""),
                }
            )

    if V10_MISMATCH.exists():
        mm = pd.read_csv(V10_MISMATCH)
        for _, r in mm.iterrows():
            task = str(r.get("task", "")).lower()
            mode = str(r.get("mode", ""))
            precision = str(r.get("precision", ""))
            if task != "sst-5":
                continue
            if not ((mode == "dense" and precision in {"int4", "int8"}) or (mode == "sparse_p0p1" and precision == "int4")):
                continue
            rows.append(
                {
                    "model": r.get("model", "roberta-large"),
                    "task": task,
                    "precision": precision,
                    "mode": mode,
                    "quantizer": r.get("quantizer", ""),
                    "checkpoint": "existing_probe_checkpoint_or_init",
                    "batch_source": "existing_probe_batch",
                    "h": float(r.get("h")),
                    "h_policy": "fixed-small" if abs(float(r.get("h")) - 1e-5) < 1e-12 else ("default" if abs(float(r.get("h")) - 1e-3) < 1e-12 else "grid"),
                    "true_directional_nmse": r.get("directional_nmse"),
                    "corr_dq_dstar": r.get("directional_corr"),
                    "sign_agreement": np.nan,
                    "active_frac": r.get("active_frac"),
                    "norm_ratio": r.get("norm_ratio"),
                    "visible_direction_cos": r.get("cos_vh_u"),
                    "d_Q_mean": np.nan,
                    "d_Q_std": r.get("d_Q_scale_or_variance"),
                    "d_star_mean": np.nan,
                    "d_star_std": np.nan,
                    "n_directions": "existing",
                    "direction_seed": "existing",
                    "source_path": r.get("source_path"),
                    "notes": "Existing probe/update mismatch diagnostic.",
                }
            )

    if HP_PROBE.exists():
        hp = pd.read_csv(HP_PROBE)
        for _, r in hp.iterrows():
            precision = str(r.get("precision_mode", ""))
            if precision not in {"fp32", "fp16", "bf16"}:
                continue
            h = float(r.get("h"))
            if h not in {1e-5, 1e-4, 1e-3, 3e-3, 1e-2}:
                continue
            rows.append(
                {
                    "model": "roberta-large",
                    "task": "sst-5",
                    "precision": precision,
                    "mode": "dense",
                    "quantizer": "none_or_native",
                    "checkpoint": "existing_seed16_high_precision_checkpoint",
                    "batch_source": "existing_checkpoint_probe",
                    "h": h,
                    "h_policy": "fixed-small" if abs(h - 1e-5) < 1e-12 else ("default" if abs(h - 1e-3) < 1e-12 else "grid"),
                    "true_directional_nmse": r.get("probe_nMSE_fd_true"),
                    "corr_dq_dstar": r.get("probe_corr_fd_true"),
                    "sign_agreement": np.nan,
                    "active_frac": 1.0,
                    "norm_ratio": r.get("probe_norm_ratio"),
                    "visible_direction_cos": r.get("probe_alignment"),
                    "d_Q_mean": np.nan,
                    "d_Q_std": np.nan,
                    "d_star_mean": np.nan,
                    "d_star_std": np.nan,
                    "n_directions": "existing",
                    "direction_seed": "existing",
                    "source_path": str(HP_PROBE.relative_to(ROOT)),
                    "notes": "High-precision dense checkpoint probe; true nMSE/corr unavailable in this summary.",
                }
            )

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("No Phase 1 probe diagnostics were found.")
    df = df.drop_duplicates(subset=["model", "task", "precision", "mode", "quantizer", "h", "source_path"]).sort_values(["mode", "task", "precision", "h", "source_path"])
    df.to_csv(dirs["phase1"] / "probe_diagnostics.csv", index=False)
    df.to_csv(out / "probe_diagnostics.csv", index=False)

    agg_cols = ["model", "task", "precision", "mode", "quantizer", "h"]
    summary = (
        df.groupby(agg_cols, dropna=False)
        .agg(
            true_directional_nmse=("true_directional_nmse", "mean"),
            corr_dq_dstar=("corr_dq_dstar", "mean"),
            sign_agreement=("sign_agreement", "mean"),
            active_frac=("active_frac", "mean"),
            norm_ratio=("norm_ratio", "mean"),
            visible_direction_cos=("visible_direction_cos", "mean"),
            num_sources=("source_path", "nunique"),
        )
        .reset_index()
        .sort_values(agg_cols)
    )
    summary.to_csv(dirs["phase1"] / "probe_diagnostics_summary.csv", index=False)
    summary.to_csv(dirs["aggregates"] / "probe_diagnostics_summary.csv", index=False)

    _plot_metric(summary, dirs["figures"] / "probe_true_nmse_vs_h", "true_directional_nmse", "true directional nMSE")
    _plot_metric(summary, dirs["figures"] / "probe_corr_vs_h", "corr_dq_dstar", "corr(d_Q, d*)")
    _plot_metric(summary, dirs["figures"] / "probe_active_fraction_vs_h", "active_frac", "active fraction")
    _plot_metric(summary, dirs["figures"] / "probe_norm_ratio_vs_h", "norm_ratio", "norm ratio")
    _plot_metric(summary, dirs["figures"] / "probe_visible_direction_cos_vs_h", "visible_direction_cos", "cos(v_h, u)")

    readme = [
        "# Phase 1 Probe Diagnostics",
        "",
        "This phase reuses existing fixed-checkpoint probe outputs. It does not run new full training.",
        "",
        "- True directional nMSE is copied only from sources that explicitly recorded finite-difference-vs-reference probe metrics.",
        "- Geometry-only fields are kept separate as active fraction, norm ratio, and visible-direction cosine.",
        "- Prefix active/norm geometry is incomplete in existing summaries and is left blank rather than fabricated.",
        "- High-precision dense probe summaries provide alignment/norm-ratio only; true nMSE/corr are unavailable there.",
        "",
        f"Rows: {len(df)}",
    ]
    (dirs["phase1"] / "README_probe.md").write_text("\n".join(readme) + "\n", encoding="utf-8")
    shutil.copyfile(dirs["phase1"] / "README_probe.md", out / "README_probe.md")
    return df


def base_lowbit_manifest_row(src: pd.Series, seed: int, out_root: Path) -> dict:
    task = str(src["task_name"])
    mode = str(src["direction_mode"])
    raw_policy = str(src["h_policy"])
    h = float(src["h"])
    label = str(src.get("h_label") or h_label(h))
    policy_dir = raw_policy
    task_compact = task.replace("-", "")

    if mode == "sparse":
        subdir = "sparse_p0p1_taskgrad"
        policy_prefix = f"int4_sparsep0p1_taskgrad_seedfixed_{task_compact}_{raw_policy}_h{label}_seed{seed}_full_bs64_step20k"
        lr = 1e-6
    elif mode == "prefix":
        subdir = "prefix_quantized"
        policy_prefix = f"int4_prefix_quantized_seedfixed_{task_compact}_{raw_policy}_h{label}_seed{seed}_full_bs64_step20k"
        lr = 0.01
    else:
        raise ValueError(f"unexpected low-bit mode: {mode}")

    run_dir = out_root / "int4_hsearch" / subdir / policy_dir / policy_prefix
    row: dict[str, object] = {}
    for key, val in src.to_dict().items():
        if isinstance(val, float) and math.isnan(val):
            val = ""
        row[key] = val
    result_keys = {
        "status",
        "steps_completed",
        "gpu_name",
        "gpu_type_requested",
        "fallback_used",
        "best_eval_acc",
        "best_eval_step",
        "last_eval_acc",
        "last_eval_step",
        "best_eval_loss",
        "best_eval_loss_step",
        "last_eval_loss",
        "last_eval_loss_step",
        "final_train_loss",
        "d_h_last",
        "d_g_last",
        "fd_true_error_last",
        "active_frac",
        "alignment",
        "norm_ratio",
        "delta_q_norm",
        "ideal_displacement_norm",
        "code_change_frac",
        "delta_visibility_mse",
        "delta_visibility_nmse",
        "delta_visibility_rel_l2",
        "lowbit_true_nmse",
        "lowbit_true_corr",
        "saturation_frac_w",
        "saturation_frac_w_plus",
        "saturation_frac_w_minus",
        "weight_recon_mse",
        "weight_recon_rel_mse",
        "weight_recon_sqnr_db",
        "corr_fd_true",
        "nMSE_fd_true",
        "fd_true_available",
        "fd_true_mse",
        "fd_true_nmse",
        "fd_true_rmse",
        "fd_true_bias",
        "resume_command",
        "warnings",
    }
    for key in result_keys:
        if key in row:
            row[key] = ""
    row.update(
        {
            "phase": "int4_hsearch",
            "run_name": policy_prefix,
            "run_dir": str(run_dir),
            "dataset": task,
            "task_name": task,
            "dataset_mode": "full",
            "data_dir": "",
            "num_k": 16,
            "seed": seed,
            "data_seed": seed,
            "batch_size": 64,
            "eval_batch_size": 64,
            "eval_batches": -1,
            "bitwidth": 4,
            "h": h,
            "h_label": label,
            "h_policy": raw_policy,
            "max_steps": 20000,
            "scale_refresh_k": 1,
            "eval_every": 1000,
            "checkpoint_steps": 1000,
            "diag_every": 100,
            "quant_log_every": 1000,
            "log_every": 100,
            "lr": lr,
            "update_scalar_source": "finite_difference",
            "update_backend": "fp16_master",
            "master_dtype": "fp16",
            "quantizer": "G128_groupwise_RTNClip_fake_quant",
            "pair_shared_grid": True,
            "fresh_round_codes": True,
            "grid_source": "unperturbed_fp16_master_weight",
            "scale_refresh_k": 1,
            "seed_mask_fix_applied": True,
            "seed_reset_before_model_load_required": True,
            "source_seed16_run_name": src["run_name"],
            "source_seed16_summary": str(LOWBIT_SUMMARY.relative_to(ROOT)),
            "status": "pending",
            "steps_completed": 0,
        }
    )
    if mode == "sparse":
        row.update(
            {
                "direction_mode": "sparse",
                "sparse_ratio": 0.1,
                "sparse_p": 0.1,
                "sparse_mask_strategy": "task_grad_static",
                "mask_strategy": "task_grad_static",
                "sparse_mask_batches": 1,
                "sparse_mask_scope": "linear_weight",
                "sparse_rescale": "none",
                "sparse_scaled_by_inv_sqrt_p": False,
                "sparse_mask_saved_in_checkpoint_required": True,
                "notes": "V10 medium supplement: sparse p=0.1 task_grad_static, unscaled mask, seed-fixed family.",
            }
        )
    else:
        row.update(
            {
                "direction_mode": "prefix",
                "prefix_precision": "fp16",
                "prefix_init_strategy": "real_act_with_random_fallback",
                "prefix_quantize": True,
                "prefix_num": 5,
                "perturbed_parameter_scope": "prefix_parameters_only",
                "quantized_forward_scope": "base_Linear.weight_plus_prefix_params_int4",
                "excluded_methods": "GPTQ;residual_grid;direct_int_update;sparse;LoRA;OPT;Mistral",
                "notes": "V10 medium supplement: prefix parameters only, prefix tensors quantized in INT4 forward/probe, seed-fixed family.",
            }
        )
    return row


def build_training_manifests(out: Path, dirs: dict, v10: pd.DataFrame) -> dict:
    hp_root = dirs["raw"] / "high_precision_sst5_fp32_multiseed"
    hp_result = hp_root / "fp32" / "multiseed_plateau" / "results"
    low_root = dirs["raw"] / "roberta_int4_sparse_prefix_multiseed"
    hp_rows: list[dict] = []
    low_rows: list[dict] = []

    h_values = [1e-5, 1e-4, 1e-3, 3e-3]
    hp_seeds_to_launch = [32, 64, 128, 256]  # seed16 already exists in HP_SOURCE.
    for seed in hp_seeds_to_launch:
        for h in h_values:
            lab = h_label(h)
            hp_rows.append(
                {
                    "lane_id": 0,
                    "gpu_type": "H100",
                    "precision_mode": "fp32",
                    "h": f"{h:.12g}",
                    "h_label": lab,
                    "run_name": f"fp32_h{lab}_seed{seed}_bs64_ckpt1k",
                    "result_root": str(hp_result),
                    "max_steps": 20000,
                    "eval_steps": 1000,
                    "checkpoint_steps": 1000,
                    "seed": seed,
                    "data_seed": seed,
                    "batch_size": 64,
                    "lr": "1e-6",
                }
            )

    summary = pd.read_csv(LOWBIT_SUMMARY)
    summary["task_name"] = summary["task_name"].astype(str)
    summary["h_policy"] = summary["h_policy"].astype(str)
    summary["direction_mode"] = summary["direction_mode"].astype(str)
    required_specs = []
    for task in ["sst-2", "trec"]:
        for policy in ["fixed_small", "mezo_default", "hstar_cleanGL"]:
            required_specs.append(("prefix", task, policy))
    for task in ["sst-5", "rte"]:
        for policy in ["fixed_small", "mezo_default", "hstar_lowbitL"]:
            required_specs.append(("sparse", task, policy))

    source_rows: list[pd.Series] = []
    for mode, task, policy in required_specs:
        match = summary[
            (summary["direction_mode"] == mode)
            & (summary["task_name"] == task)
            & (summary["h_policy"] == policy)
            & (summary["status"].astype(str).str.contains("complete", case=False, na=False))
        ]
        if match.empty:
            raise RuntimeError(f"Missing V10 seed16 complete source row for {mode} {task} {policy}")
        source_rows.append(match.iloc[0])

    for seed in [32, 64]:
        for src in source_rows:
            low_rows.append(base_lowbit_manifest_row(src, seed, low_root))

    all_jobs: list[tuple[str, dict]] = [("high_precision", r) for r in hp_rows] + [("lowbit", r) for r in low_rows]
    lanes: dict[int, list[tuple[str, dict]]] = {i: [] for i in range(6)}
    for idx, item in enumerate(all_jobs):
        lanes[idx % 6].append(item)

    hp_fields = ["lane_id", "gpu_type", "precision_mode", "h", "h_label", "run_name", "result_root", "max_steps", "eval_steps", "checkpoint_steps", "seed", "data_seed", "batch_size", "lr"]
    low_fields = list(dict.fromkeys(k for _, row in [("x", r) for r in low_rows] for k in row.keys()))
    for lane_id, items in lanes.items():
        hp_lane = []
        low_lane = []
        for kind, row in items:
            if kind == "high_precision":
                r = dict(row)
                r["lane_id"] = lane_id
                hp_lane.append(r)
            else:
                low_lane.append(dict(row))
        write_csv(dirs["manifests"] / f"high_precision_lane{lane_id}.csv", hp_lane, hp_fields)
        write_csv(dirs["manifests"] / f"lowbit_lane{lane_id}.csv", low_lane, low_fields)

    write_csv(out / "high_precision_sst5_multiseed_launch_manifest.csv", hp_rows, hp_fields)
    write_csv(out / "lowbit_sparse_prefix_multiseed_launch_manifest.csv", low_rows, low_fields)
    write_csv(
        out / "all_required_training_launch_manifest.csv",
        [
            {
                "job_kind": kind,
                "lane_id": i % 6,
                **row,
            }
            for i, (kind, row) in enumerate(all_jobs)
        ],
    )

    existing = []
    # Preserve seed16 provenance rows for required tables.
    for _, r in pd.read_csv(LOWBIT_SUMMARY).iterrows():
        if (
            (r.get("direction_mode") == "prefix" and r.get("task_name") in {"sst-2", "trec"} and r.get("h_policy") in {"fixed_small", "mezo_default", "hstar_cleanGL"})
            or (r.get("direction_mode") == "sparse" and r.get("task_name") in {"sst-5", "rte"} and r.get("h_policy") in {"fixed_small", "mezo_default", "hstar_lowbitL"})
        ):
            existing.append({"source": "existing_seed16_lowbit", **r.to_dict()})
    write_csv(out / "existing_seed16_reused_runs.csv", existing)

    metadata = {
        "high_precision_new_runs": len(hp_rows),
        "lowbit_new_runs": len(low_rows),
        "existing_seed16_reused_lowbit_rows": len(existing),
        "max_concurrent_lanes": 6,
        "target_gpu": "H100",
        "target_time": "72:00:00",
        "git_commit": git_commit(),
    }
    write_json(out / "metadata.json", metadata)
    return metadata


def write_sbatch_scripts(out: Path, dirs: dict) -> None:
    lane_script = dirs["scripts"] / "run_v10_medium_lane.sbatch"
    lane_script.write_text(
        f"""#!/bin/bash
#SBATCH --job-name=v10-med
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=72:00:00
#SBATCH --gres=gpu:H100:1
#SBATCH --array=0-5%6
#SBATCH --output={out}/jobs/%x_%A_%a.out
#SBATCH --error={out}/jobs/%x_%A_%a.err

set -euo pipefail

REPO_ROOT="${{REPO_ROOT:-{ROOT}}}"
OUT_DIR="${{OUT_DIR:-{out}}}"
LANE_ID="${{SLURM_ARRAY_TASK_ID:-0}}"
CONDA_ENV_HP="${{CONDA_ENV_HP:-ciao}}"
CONDA_ENV_LB="${{CONDA_ENV_LB:-mezo-env}}"

cd "$REPO_ROOT"
mkdir -p "$OUT_DIR/jobs" "$OUT_DIR/raw_runs/slurm_environment"

echo "job_id=${{SLURM_JOB_ID:-local}} lane=${{LANE_ID}} host=$(hostname) start=$(date -Is)" | tee "$OUT_DIR/jobs/lane${{LANE_ID}}_start.txt"
nvidia-smi | tee "$OUT_DIR/raw_runs/slurm_environment/lane${{LANE_ID}}_nvidia_smi.txt" || true
git rev-parse HEAD | tee "$OUT_DIR/raw_runs/slurm_environment/lane${{LANE_ID}}_git_commit.txt" || true
env | sort > "$OUT_DIR/raw_runs/slurm_environment/lane${{LANE_ID}}_env.txt"

source_conda() {{
  if [[ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
  elif [[ -f "$HOME/.conda/etc/profile.d/conda.sh" ]]; then
    source "$HOME/.conda/etc/profile.d/conda.sh"
  elif [[ -f "/apps/conda/etc/profile.d/conda.sh" ]]; then
    source "/apps/conda/etc/profile.d/conda.sh"
  fi
}}

HP_MANIFEST="$OUT_DIR/manifests/high_precision_lane${{LANE_ID}}.csv"
if [[ -s "$HP_MANIFEST" ]] && [[ "$(wc -l < "$HP_MANIFEST")" -gt 1 ]]; then
  source_conda
  conda activate "$CONDA_ENV_HP"
  echo "[lane $LANE_ID] running high-precision manifest $HP_MANIFEST"
  EXP_ROOT="$OUT_DIR/raw_runs/high_precision_sst5_fp32_multiseed" \\
  LANE_ID="$LANE_ID" \\
  LANE_MANIFEST="$HP_MANIFEST" \\
  CONDA_ENV="$CONDA_ENV_HP" \\
  PROBE_DIRECTIONS="${{PROBE_DIRECTIONS:-4}}" \\
  bash scripts/run_latest_main_lane.sh
else
  echo "[lane $LANE_ID] no high-precision rows"
fi

LB_MANIFEST="$OUT_DIR/manifests/lowbit_lane${{LANE_ID}}.csv"
if [[ -s "$LB_MANIFEST" ]] && [[ "$(wc -l < "$LB_MANIFEST")" -gt 1 ]]; then
  source_conda
  conda activate "$CONDA_ENV_LB"
  export DATALOADER_SHUFFLE=True
  export REQUESTED_GPU_TYPE=H100
  export FALLBACK_USED=0
  echo "[lane $LANE_ID] running low-bit manifest $LB_MANIFEST"
  python tools/rtnclip_roberta_sst5_batch.py \\
    --output_root "$OUT_DIR/raw_runs/roberta_int4_sparse_prefix_multiseed" \\
    --manifest "$LB_MANIFEST" \\
    run-manifest
else
  echo "[lane $LANE_ID] no low-bit rows"
fi

echo "job_id=${{SLURM_JOB_ID:-local}} lane=${{LANE_ID}} end=$(date -Is)" | tee "$OUT_DIR/jobs/lane${{LANE_ID}}_end.txt"
""",
        encoding="utf-8",
    )
    lane_script.chmod(0o755)

    submit = out / "submit_v10_medium_supplement.sh"
    submit.write_text(
        f"""#!/bin/bash
set -euo pipefail

ACCOUNT="${{ACCOUNT:-}}"
PARTITION="${{PARTITION:-gpu_p}}"
TIME="${{TIME:-72:00:00}}"
GPU_TYPE="${{GPU_TYPE:-H100}}"
CONDA_ENV_HP="${{CONDA_ENV_HP:-ciao}}"
CONDA_ENV_LB="${{CONDA_ENV_LB:-mezo-env}}"
OUT_DIR="${{OUT_DIR:-{out}}}"
REPO_ROOT="${{REPO_ROOT:-{ROOT}}}"

mkdir -p "$OUT_DIR/jobs"
args=(
  --job-name=v10-med
  --nodes=1
  --ntasks=1
  --cpus-per-task=8
  --mem=96G
  --time="$TIME"
  --gres="gpu:${{GPU_TYPE}}:1"
  --array=0-5%6
  --output="$OUT_DIR/jobs/%x_%A_%a.out"
  --error="$OUT_DIR/jobs/%x_%A_%a.err"
  --export=ALL,OUT_DIR="$OUT_DIR",REPO_ROOT="$REPO_ROOT",CONDA_ENV_HP="$CONDA_ENV_HP",CONDA_ENV_LB="$CONDA_ENV_LB"
)
if [[ -n "$ACCOUNT" ]]; then args+=(--account="$ACCOUNT"); fi
if [[ -n "$PARTITION" ]]; then args+=(--partition="$PARTITION"); fi

echo "Submitting V10 medium supplement 6-lane array at $(date -Is)" | tee -a "$OUT_DIR/jobs/job_ids.txt"
sbatch "${{args[@]}}" "$OUT_DIR/scripts/run_v10_medium_lane.sbatch" | tee -a "$OUT_DIR/jobs/job_ids.txt"
""",
        encoding="utf-8",
    )
    submit.chmod(0o755)

    monitor = out / "monitor_v10_medium_supplement.sh"
    monitor.write_text(
        """#!/bin/bash
set -euo pipefail
OUT_DIR="${OUT_DIR:-$(pwd)}"
echo "Recent jobs:"
squeue -u "$USER" -o "%.18i %.9P %.30j %.8u %.2t %.10M %.6D %R" | grep -E 'v10-med|JOBID' || true
echo
echo "Job ids:"
cat "$OUT_DIR/jobs/job_ids.txt" 2>/dev/null || true
echo
echo "Completed run summaries so far:"
find "$OUT_DIR/raw_runs" -name run_summary.json | wc -l
""",
        encoding="utf-8",
    )
    monitor.chmod(0o755)


def write_postprocess_script(out: Path, dirs: dict) -> None:
    script = dirs["scripts"] / "summarize_v10_medium_results.py"
    script.write_text(
        r'''#!/usr/bin/env python3
from __future__ import annotations
import csv, json, math, sys
from pathlib import Path
import pandas as pd
import numpy as np

OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.cwd()
AG = OUT / "aggregates"
FIG = OUT / "figures"
AG.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)

def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))

def collect_lowbit():
    rows = []
    root = OUT / "raw_runs" / "roberta_int4_sparse_prefix_multiseed"
    for p in root.glob("int4_hsearch/**/run_summary.json"):
        try:
            d = read_json(p)
        except Exception:
            continue
        cfg = {}
        cfgp = p.parent / "run_config.json"
        if cfgp.exists():
            try: cfg = read_json(cfgp)
            except Exception: cfg = {}
        rows.append({**cfg, **d, "source_path": str(p)})
    return pd.DataFrame(rows)

def collect_hp():
    rows = []
    root = OUT / "raw_runs" / "high_precision_sst5_fp32_multiseed"
    for p in root.glob("fp32/multiseed_plateau/results/**/run_summary.json"):
        try: rows.append({**read_json(p), "source_path": str(p)})
        except Exception: pass
    return pd.DataFrame(rows)

hp = collect_hp()
lb = collect_lowbit()
if not hp.empty:
    hp.to_csv(OUT / "raw_runs" / "high_precision_sst5_multiseed_runs.csv", index=False)
    summ = hp.groupby(["h"]).agg(
        n=("best_eval_acc","count"),
        best_dev_acc_mean=("best_eval_acc","mean"),
        best_dev_acc_std=("best_eval_acc","std"),
        final_dev_acc_mean=("last_eval_acc","mean"),
        final_dev_acc_std=("last_eval_acc","std"),
    ).reset_index()
    summ.to_csv(AG / "high_precision_sst5_multiseed_summary.csv", index=False)
    # Simple two-way variance components.
    tab = hp.dropna(subset=["h","seed","best_eval_acc"]).copy()
    if tab["h"].nunique() > 1 and tab["seed"].nunique() > 1:
        grand = tab["best_eval_acc"].mean()
        hm = tab.groupby("h")["best_eval_acc"].mean()
        sm = tab.groupby("seed")["best_eval_acc"].mean()
        resid = tab.apply(lambda r: r["best_eval_acc"] - hm.loc[r["h"]] - sm.loc[r["seed"]] + grand, axis=1)
        var = pd.DataFrame([{
            "metric": "best_eval_acc",
            "num_runs": len(tab),
            "num_h": tab["h"].nunique(),
            "num_seed": tab["seed"].nunique(),
            "var_h_policy": float(np.var(hm - grand)),
            "var_seed_direction": float(np.var(sm - grand)),
            "var_residual_interaction": float(np.var(resid)),
        }])
    else:
        var = pd.DataFrame([{"metric":"best_eval_acc","status":"insufficient_completed_runs"}])
    var.to_csv(AG / "high_precision_variance_components.csv", index=False)
if not lb.empty:
    lb.to_csv(OUT / "raw_runs" / "lowbit_sparse_prefix_multiseed_runs.csv", index=False)
    prefix = lb[lb.get("direction_mode","").astype(str).eq("prefix")].copy()
    sparse = lb[lb.get("direction_mode","").astype(str).eq("sparse")].copy()
    for name, df in [("prefix_int4_multiseed_summary.csv", prefix), ("sparse_int4_multiseed_summary.csv", sparse)]:
        if df.empty:
            pd.DataFrame([{"status":"no_completed_runs"}]).to_csv(AG / name, index=False)
            continue
        group_cols = ["task_name","h_policy","h"]
        s = df.groupby(group_cols).agg(
            n=("best_eval_acc","count"),
            best_dev_acc_mean=("best_eval_acc","mean"),
            best_dev_acc_std=("best_eval_acc","std"),
            final_dev_acc_mean=("last_eval_acc","mean"),
            final_dev_acc_std=("last_eval_acc","std"),
        ).reset_index()
        s.to_csv(AG / name, index=False)

try:
    import matplotlib.pyplot as plt
    if not hp.empty:
        fig, ax = plt.subplots(figsize=(6,4))
        for seed, g in hp.groupby("seed"):
            g = g.sort_values("h")
            ax.plot(g["h"], g["best_eval_acc"], marker="o", label=f"seed{seed}")
        ax.set_xscale("log"); ax.set_xlabel("h"); ax.set_ylabel("best dev acc"); ax.grid(True, alpha=.25); ax.legend(fontsize=7)
        for ext in ["pdf","png"]: fig.savefig(FIG / f"high_precision_seed_lines_accuracy_vs_h.{ext}", bbox_inches="tight", dpi=180)
        plt.close(fig)
    for mode_name, filename in [("prefix", "prefix_int4_policy_bars_with_seed_dots"), ("sparse", "sparse_int4_policy_bars_with_seed_dots")]:
        df = lb[lb.get("direction_mode","").astype(str).eq(mode_name)] if not lb.empty else pd.DataFrame()
        if df.empty: continue
        fig, ax = plt.subplots(figsize=(8,4))
        labels=[]; vals=[]; xs=[]; i=0
        for (task, pol), g in df.groupby(["task_name","h_policy"]):
            labels.append(f"{task}\n{pol}")
            vals.append(g["best_eval_acc"].mean())
            xs.append(i)
            ax.scatter([i]*len(g), g["best_eval_acc"], color="black", s=14, zorder=3)
            i += 1
        ax.bar(xs, vals, alpha=.75)
        ax.set_xticks(xs); ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
        ax.set_ylabel("best dev acc"); ax.grid(True, axis="y", alpha=.25)
        for ext in ["pdf","png"]: fig.savefig(FIG / f"{filename}.{ext}", bbox_inches="tight", dpi=180)
        plt.close(fig)
except Exception as exc:
    (OUT / "figure_generation_error.txt").write_text(str(exc)+"\n")

notes = ["# V10 Medium Supplement Paper Update Notes", ""]
notes.append("Run this summarizer after Slurm jobs finish. Completed rows are aggregated only from run_summary.json files.")
notes.append("")
notes.append(f"High precision completed rows: {len(hp)}")
notes.append(f"Low-bit completed rows: {len(lb)}")
(OUT / "paper_update_notes.md").write_text("\n".join(notes)+"\n", encoding="utf-8")
print(f"wrote summaries under {OUT}")
''',
        encoding="utf-8",
    )
    script.chmod(0o755)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default=str(OUT_DEFAULT))
    parser.add_argument("--no_phase1", action="store_true")
    args = parser.parse_args()
    out = Path(args.output_dir)
    v10 = load_v10_rows()
    dirs = make_audit(out, v10)
    if not args.no_phase1:
        build_phase1_probe(out, dirs)
    metadata = build_training_manifests(out, dirs, v10)
    write_sbatch_scripts(out, dirs)
    write_postprocess_script(out, dirs)
    missing = [
        "# Missing / Limited Items",
        "",
        "- Phase 1 prefix visibility geometry is incomplete in existing summaries; true nMSE/corr rows are retained where available.",
        "- Prefix RTE is not in required training because V10 audit marked it incomplete/not comparable.",
        "- Sparse TREC is not in required training; V10 required sparse tasks are SST-5 and RTE.",
        "- Existing low-bit seed16 runs are reused; new jobs fill seeds 32/64.",
        "- Existing high-precision seed16 FP32 runs are reused; new jobs fill seeds 32/64/128/256.",
        "- Exact direction stream matching across h is limited by existing runner internals; base train/data seeds are paired.",
    ]
    (out / "missing_items.md").write_text("\n".join(missing) + "\n", encoding="utf-8")
    with (out / "README.md").open("a", encoding="utf-8") as f:
        f.write("\n## Launch Summary\n\n")
        f.write("```\n")
        f.write(pd.DataFrame([metadata]).to_string(index=False) + "\n")
        f.write("```\n\n")
        f.write("Submit with:\n\n")
        f.write("```bash\n")
        f.write(f"bash {out}/submit_v10_medium_supplement.sh\n")
        f.write("```\n\n")
        f.write("Monitor with:\n\n")
        f.write("```bash\n")
        f.write(f"OUT_DIR={out} bash {out}/monitor_v10_medium_supplement.sh\n")
        f.write("```\n\n")
        f.write("Summarize completed jobs with:\n\n")
        f.write("```bash\n")
        f.write(f"python {out}/scripts/summarize_v10_medium_results.py {out}\n")
        f.write("```\n")
    print(json.dumps({"output_dir": str(out), **metadata}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
