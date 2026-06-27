#!/usr/bin/env python
"""Package lightweight experiment results, provenance, and paper artifacts.

The repository contains many terabytes-scale experiment directories once model
checkpoints are included.  This packager intentionally archives analysis-ready
results only: configs, summaries, metrics, CSV/JSON/Markdown docs, logs, plots,
and generation scripts.  It excludes raw model weights/checkpoints while
recording every excluded large/binary file in a manifest.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
import shutil
import socket
import subprocess
import sys
import textwrap
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = "all_experiment_results_package_20260626"
MAX_FILE_MB = 50

INCLUDE_SUFFIXES = {
    ".csv",
    ".tsv",
    ".json",
    ".jsonl",
    ".md",
    ".txt",
    ".tex",
    ".pdf",
    ".png",
    ".svg",
    ".py",
    ".sh",
    ".sbatch",
    ".yaml",
    ".yml",
}

EXCLUDE_SUFFIXES = {
    ".pt",
    ".pth",
    ".bin",
    ".safetensors",
    ".ckpt",
    ".npy",
    ".npz",
    ".pkl",
    ".pickle",
    ".parquet",
    ".arrow",
    ".zip",
    ".tar",
    ".gz",
    ".xz",
    ".log",
    ".out",
    ".err",
}

EXCLUDE_DIR_NAMES = {
    ".git",
    "__pycache__",
    ".pytest_cache",
    ".cache",
    "cache",
    "checkpoints",
    "checkpoint",
    "quantized",
    "awq_tmp",
    "gptq_tmp",
    "wandb",
    "node_modules",
}

SEARCH_ROOTS = [
    "outputs",
    "experiments",
    "paper_artifacts_final",
    "hwindow_final_experiments_bundle_v2",
    "hwindow_final_experiments_bundle",
    "hwindow_12h_highdim_bundle",
    "hwindow_12h_highdim_extra_g64",
    "hwindow_12h_highdim_extra_g256",
    "hwindow_12h_highdim_extra_midp",
    "interval_aware_h_probe",
    "interval_h_selection_8h_bundle",
    "safe_override_6h_a100_bundle",
    "sharp_interval_fit_and_roberta_int4_eval",
    "synthetic_fit_repair",
    "synthetic_no_leakage_interval",
    "synthetic_no_leakage_interval_smoke",
    "lowbit_update_experiment",
    "tools",
    "scripts",
    "slurm",
]

ROOT_DOCS = [
    "README.md",
    "README_int4_window_preflight.md",
    "main.tex",
    "updated_experiment_plan_int8_breadth_v7_mse_required.md",
    "updated_experiment_plan_int8_breadth_v6.md",
    "updated_experiment_plan_rtnclip_v4.md",
    "main_experiment_plan_revised_with_tables.md",
    "pilot_experiments_20260419.md",
]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()
    except Exception:
        return ""


def write_csv(path: Path, rows: List[Dict[str, object]], fieldnames: Optional[List[str]] = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = []
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def safe_load_json(path: Path) -> Optional[Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def flatten_json(data: Any, prefix: str = "", limit: int = 80) -> Dict[str, object]:
    out: Dict[str, object] = {}
    if isinstance(data, dict):
        for key, value in data.items():
            k = f"{prefix}.{key}" if prefix else str(key)
            if isinstance(value, (dict, list)):
                if len(out) < limit:
                    out.update(flatten_json(value, k, limit))
            else:
                out[k] = value
                if len(out) >= limit:
                    break
    elif isinstance(data, list):
        out[prefix or "list_len"] = len(data)
    return out


def should_skip_dir(path: Path) -> bool:
    return any(part in EXCLUDE_DIR_NAMES for part in path.parts)


def classify_file(path: Path) -> str:
    name = path.name.lower()
    if name in {"run_config.json", "config_manifest.json", "run_manifest_row.json"}:
        return "config"
    if name in {"run_summary.json", "summary.json"} or "summary" in name:
        return "summary"
    if "metrics" in name or "eval" in name:
        return "metrics"
    if path.suffix.lower() in {".png", ".pdf", ".svg"}:
        return "figure_or_pdf"
    if path.suffix.lower() in {".py", ".sh", ".sbatch"}:
        return "script"
    if path.suffix.lower() in {".log", ".out", ".err"}:
        return "log"
    return "artifact"


def include_decision(path: Path, max_bytes: int) -> Tuple[bool, str]:
    suffix = path.suffix.lower()
    if should_skip_dir(path.parent):
        return False, "excluded_directory"
    if suffix in EXCLUDE_SUFFIXES:
        return False, "excluded_binary_or_archive_suffix"
    if suffix not in INCLUDE_SUFFIXES:
        return False, "unsupported_suffix"
    try:
        size = path.stat().st_size
    except OSError:
        return False, "stat_failed"
    if size > max_bytes:
        return False, f"larger_than_{max_bytes}_bytes"
    return True, "included"


def iter_candidate_files() -> Iterable[Path]:
    seen = set()
    for rel in SEARCH_ROOTS:
        root = REPO_ROOT / rel
        if not root.exists():
            continue
        if root.is_file():
            files = [root]
        else:
            files = []
            for dirpath, dirnames, filenames in os.walk(root):
                dirnames[:] = [d for d in dirnames if d not in EXCLUDE_DIR_NAMES]
                for name in filenames:
                    files.append(Path(dirpath) / name)
        for path in files:
            resolved = path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            yield path
    for rel in ROOT_DOCS:
        path = REPO_ROOT / rel
        if path.exists() and path.is_file():
            resolved = path.resolve()
            if resolved not in seen:
                seen.add(resolved)
                yield path


def infer_run_row(run_dir: Path) -> Dict[str, object]:
    cfg = safe_load_json(run_dir / "run_config.json") or {}
    summ = safe_load_json(run_dir / "run_summary.json") or {}
    manifest = safe_load_json(run_dir / "run_manifest_row.json") or {}
    row: Dict[str, object] = {
        "run_dir": str(run_dir.relative_to(REPO_ROOT)),
        "has_run_config": (run_dir / "run_config.json").exists(),
        "has_run_summary": (run_dir / "run_summary.json").exists(),
        "status": summ.get("status") or cfg.get("status") or manifest.get("status") or "",
        "run_name": cfg.get("run_name") or summ.get("run_name") or manifest.get("run_name") or run_dir.name,
        "model": cfg.get("model") or cfg.get("model_id") or summ.get("model_id") or manifest.get("model") or "",
        "task": cfg.get("task") or cfg.get("task_name") or summ.get("task") or summ.get("task_name") or "",
        "dataset": cfg.get("dataset") or cfg.get("task") or cfg.get("task_name") or "",
        "dataset_mode": cfg.get("dataset_mode") or summ.get("dataset_mode") or "",
        "precision": cfg.get("precision") or cfg.get("precision_mode") or summ.get("precision") or summ.get("precision_mode") or "",
        "quantizer": cfg.get("quantizer") or cfg.get("quantizer_backend") or summ.get("quantizer") or "",
        "mode": cfg.get("mode") or cfg.get("perturbation_mode") or cfg.get("direction") or summ.get("mode") or "",
        "h": cfg.get("h") or summ.get("h") or manifest.get("h") or "",
        "h_label": cfg.get("h_label") or summ.get("h_label") or "",
        "seed": cfg.get("seed") or summ.get("seed") or "",
        "data_seed": cfg.get("data_seed") or summ.get("data_seed") or "",
        "steps": cfg.get("steps") or cfg.get("max_steps") or summ.get("steps") or "",
        "steps_completed": summ.get("steps_completed") or "",
        "best_eval_acc": summ.get("best_eval_acc") or summ.get("best_dev_acc") or summ.get("best_acc") or "",
        "last_eval_acc": summ.get("last_eval_acc") or summ.get("final_dev_acc") or "",
        "best_eval_loss": summ.get("best_eval_loss") or "",
        "last_eval_loss": summ.get("last_eval_loss") or "",
        "runtime_sec": summ.get("runtime_sec") or "",
    }
    return row


def build_run_index() -> List[Dict[str, object]]:
    run_dirs = set()
    for pattern in ("run_config.json", "run_summary.json", "run_manifest_row.json"):
        for path in (REPO_ROOT / "outputs").rglob(pattern) if (REPO_ROOT / "outputs").exists() else []:
            if should_skip_dir(path.parent):
                continue
            run_dirs.add(path.parent)
    return [infer_run_row(p) for p in sorted(run_dirs)]


def build_experiment_family_index() -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    outputs = REPO_ROOT / "outputs"
    if outputs.exists():
        for child in sorted(outputs.iterdir()):
            if not child.is_dir():
                continue
            count_configs = sum(1 for _ in child.rglob("run_config.json"))
            count_summaries = sum(1 for _ in child.rglob("run_summary.json"))
            count_csv = sum(1 for _ in child.rglob("*.csv"))
            count_md = sum(1 for _ in child.rglob("*.md"))
            rows.append(
                {
                    "experiment_root": str(child.relative_to(REPO_ROOT)),
                    "description_inferred_from_name": infer_description(child.name),
                    "run_config_count": count_configs,
                    "run_summary_count": count_summaries,
                    "csv_count": count_csv,
                    "markdown_count": count_md,
                }
            )
    return rows


def infer_description(name: str) -> str:
    n = name.lower()
    tags = []
    if "roberta" in n or "rtnclip" in n:
        tags.append("RoBERTa/RTNClip")
    if "opt13b" in n or "opt" in n:
        tags.append("OPT")
    if "int4" in n:
        tags.append("INT4")
    if "int8" in n:
        tags.append("INT8")
    if "fp16" in n:
        tags.append("FP16")
    if "fp32" in n:
        tags.append("FP32")
    if "sparse" in n:
        tags.append("sparse")
    if "prefix" in n:
        tags.append("prefix")
    if "hstar" in n or "hwindow" in n or "window" in n:
        tags.append("h-window / h-star")
    if "mse" in n or "nmse" in n:
        tags.append("directional MSE")
    if "residual" in n:
        tags.append("residual-grid")
    if "qzo" in n or "gptq" in n or "awq" in n:
        tags.append("QZO/GPTQ/AWQ")
    if "synthetic" in n:
        tags.append("synthetic")
    return ", ".join(tags) if tags else "miscellaneous experiment outputs"


def write_docs(package_dir: Path, run_count: int, artifact_count: int, excluded_count: int) -> None:
    docs = package_dir / "docs"
    docs.mkdir(parents=True, exist_ok=True)
    readme = f"""
# MeZO / Perturbation-Window Experiment Results Package

Generated: `{datetime.now().isoformat(timespec='seconds')}`

Repository: `{REPO_ROOT}`

Git commit: `{git_commit()}`

This package consolidates the lightweight results from the current MeZO /
precision-aware ZO perturbation-window project.  It is intended for analysis,
paper writing, and result sharing.  It does **not** include model checkpoints or
large quantized model files.

## What is included

- Run configs, manifests, summaries, metrics CSV/JSONL, eval logs, diagnostic
  CSVs, Markdown reports, LaTeX tables, figures, and scripts.
- A searchable run index: `indices/run_index.csv`.
- A top-level experiment-family index: `indices/experiment_family_index.csv`.
- Artifact provenance and checksums: `indices/artifact_manifest.csv`.
- Excluded large/binary files: `indices/excluded_files.csv`.
- Method, parameter, and formula documentation under `docs/`.

## Size-control policy

Included artifact count: `{artifact_count}`

Indexed run count: `{run_count}`

Excluded file count: `{excluded_count}`

Excluded by design:

- checkpoints, `master.pt`, model weights, `.bin`, `.safetensors`, `.ckpt`;
- local cache/quantized model directories;
- files larger than {MAX_FILE_MB} MB;
- nested zip/tar archives.

The excluded-file manifest records source paths and reasons.

## How to use

1. Start from `docs/EXPERIMENT_GUIDE.md` for the experiment families.
2. Use `indices/run_index.csv` to find run-level metrics.
3. Use `indices/artifact_manifest.csv` to trace each copied file back to the
   repository source path and checksum.
4. Use `docs/FORMULAS_AND_ALGORITHMS.md` for the exact h-window, MeZO, RTNClip,
   sparse, prefix, and diagnostic definitions used in these experiments.
"""
    (package_dir / "README.md").write_text(textwrap.dedent(readme).strip() + "\n", encoding="utf-8")

    guide = r"""
# Experiment Guide

This guide groups the collected results by scientific purpose.  The package is
an index of historical work, not a claim that every run is paper-ready.

## 1. RoBERTa-large / SST-5 precision h-sweeps

Purpose: measure how finite-difference radius `h` behaves under FP32, FP16,
BF16 where available, INT8, and INT4.

Typical settings:

- model: `roberta-large`;
- task/dataset: `SST-5`, full data unless stated otherwise;
- seed/data_seed: usually `16`;
- batch size: usually `64`;
- direction: dense Gaussian MeZO direction;
- estimator: two-sided finite difference;
- checkpoints/eval: usually every 1000 steps for full sweeps.

Main outputs appear under experiment roots containing names such as
`rtnclip_int4`, `fp32_fp16`, `precision_window`, and `h_sweep`.

## 2. INT8/INT4 RTNClip fake-quantized forward experiments

Purpose: test low-bit forward oracles with FP16 master updates.

Core low-bit oracle:

- groupwise RTNClip fake quantization;
- group size `G128`;
- quantized modules: `nn.Linear.weight`;
- symmetric signed quantization;
- INT8 qmax = 127, INT4 qmax = 7;
- no activation quantization, no packing;
- plus/minus probes share the grid from unperturbed master weights;
- perturbed weights are freshly rounded for each side.

## 3. RoBERTa INT4 dense / sparse / prefix multi-task runs

Tasks include SST-2, SST-5, RTE, MNLI, and TREC.

Modes:

- dense: all floating trainable parameters perturbed;
- sparse p=0.1 or p=0.01: unscaled mask directions `u=m*z`;
- prefix: prefix-only trainable parameters in several variants, including
  FP32/MeZO-style and INT4-quantized variants.

Policies compared historically:

- fixed-small, often `h=1e-5`;
- MeZO default `h=1e-3`;
- cleanGL / lowbitGL analytical h-star variants;
- minimum measured MSE candidates in diagnostic pilots;
- safe/default-aware variants in later paper packaging.

## 4. OPT-1.3B INT4 cross-architecture experiments

Purpose: sanity check whether the RoBERTa h-window findings transfer to a
decoder-only model.

Important caveat: these OPT task combinations are not all direct reproductions
of original MeZO paper benchmarks. They are project-specific sanity checks.

Recent OPT/SST-2 results include:

- seed/data_seed 16, `h=1e-3`: best acc about 0.897, last acc about 0.889;
- seed/data_seed 16, `hstar_cont≈5.06e-4`: high early peak, later collapse;
- seed/data_seed 16, `h=3e-3`: lower but stable;
- seed/data_seed 17/42 multi-seed lanes are indexed if present and may be
  incomplete depending on when this package was generated.

## 5. Directional-MSE and activation diagnostics

Purpose: separate true directional finite-difference error from quantization
geometry/visibility proxies.

Important distinction:

- true directional nMSE compares `d_Q(h,u)` with `d_star(u)=<grad,u>`;
- geometry diagnostics compare quantized displacement with `2hu` and do not
  equal loss-level directional MSE.

Recent diagnostics include:

- RoBERTa/SST-5/INT4 activation curves;
- OPT/SST-2/INT4 activation curves;
- raw-L vs unit-L dimension-correction checks.

## 6. Synthetic and analytical h-window experiments

Purpose: validate the frozen h-window theory and explore high-dimensional
effects under controlled oracles.

There are multiple generations:

- early interval-aware probes;
- no-leakage synthetic directional-MSE tests;
- final frozen-window analytical package;
- repaired synthetic fit analyses.

Only the final paper artifacts should be treated as paper-facing unless a file
explicitly says otherwise.

## 7. Residual-grid, QZO/GPTQ/AWQ, and exploratory low-bit update tests

These are exploratory and often diagnostic-only. They are included for
provenance but should not be mixed into the main h-window claim without checking
the corresponding reports and run configs.
"""
    (docs / "EXPERIMENT_GUIDE.md").write_text(textwrap.dedent(guide).strip() + "\n", encoding="utf-8")

    formulas = r"""
# Formulas And Algorithms

## Two-sided MeZO finite difference

For parameters `theta`, random direction `u`, and radius `h`:

```text
d_h(u) = [L(theta + h u) - L(theta - h u)] / (2h)
g_hat  = d_h(u) * u
theta <- theta - lr * g_hat
```

For low-bit quantized forward oracles:

```text
d_Q(h,u) = [L(Q_t(theta + h u)) - L(Q_t(theta - h u))] / (2h)
```

where `Q_t` is built from the unperturbed master weight at step `t`.

## True directional MSE

The canonical loss-level directional target is:

```text
d_star(u) = <grad L(theta), u>
A_true(h) = E[(d_Q(h,u) - d_star(u))^2] / (E[d_star(u)^2] + eps)
```

Only this, or a verified equivalent, should be labeled true directional MSE or
`fd_true_nmse`.

## Geometry / visibility diagnostics

These are not loss-level true MSE:

```text
Delta_Q(h,u) = Q_t(theta + h u) - Q_t(theta - h u)
b_h          = Delta_Q(h,u) / (2h)
A_cross      = E[||b_h - u||^2] / E[||u||^2]
active_frac  = mean_i[Q(w_i+h u_i) != Q(w_i-h u_i)]
alignment    = cos(Delta_Q, 2h u)
norm_ratio   = ||Delta_Q|| / (||2h u|| + eps)
chi          = mean_i[((Delta_Q_i)/(2h u_i) - 1)^2]
```

## RTNClip groupwise fake quantization

For each Linear weight matrix, groups are contiguous along the input dimension
with group size 128 by default. For group `B`:

```text
alpha_grid = [1.0, 0.95, 0.90, 0.85, 0.80, 0.75, 0.70]
scale_B(alpha) = alpha * max(abs(w_B)) / qmax
q_i(alpha) = clip(round(w_i / scale_B(alpha)), -qmax, qmax)
wq_i(alpha) = scale_B(alpha) * q_i(alpha)
alpha_B = argmin_alpha mean_i (w_i - wq_i(alpha))^2
```

INT8 uses `qmax=127`; INT4 uses `qmax=7`. Bias, LayerNorm, and embeddings are
kept unquantized unless a specific experiment says otherwise.

Shared-grid ZO rule:

```text
grid_t = RTNClipGrid(w_t)
Q_plus  = Q_t(w_t + h u_t)  # same grid
Q_minus = Q_t(w_t - h u_t)  # same grid, fresh rounded codes
d_Q = [L(Q_plus) - L(Q_minus)] / (2h)
w_master <- w_master - lr * d_Q * u_t
```

No direct INT lattice update is used in the standard RTNClip FP16-master runs.

## Sparse MeZO variants

Unscaled sparse direction:

```text
u = m * z,  z_i ~ N(0,1),  m_i in {0,1}
```

The package contains both legacy highest-absolute-weight sparse masks and later
task-gradient / extreme-sparse inspired variants. Check `mask_strategy`,
`sparse_p`, and `sparse_rescale` before comparing rows.

## Prefix variants

Prefix experiments differ materially:

- FP32/MeZO-style prefix: prefix parameters follow the standard high-precision
  prefix path;
- INT4 prefix-forward variants: frozen/base and sometimes prefix matrices are
  quantized depending on the run config.

Do not merge these without checking `run_config.json`.

## Frozen paper h-window quantities

The final paper method uses:

```text
h_ref = (alpha / beta)^(1/4) = sqrt(h_q * h_loc)

rho(h) = (d+4)/(d+1) * [ (h_q/h)^2 + (h/h_loc)^2 ]

rho_min = 2 * (d+4)/(d+1) * h_q / h_loc

W_tau^th = { h : rho(h) <= tau }
```

Primary paper window: `tau=1`. Strict sensitivity: `tau=0.1`.

Practical rule:

1. if default `h=1e-3` is inside the predicted window, keep default;
2. otherwise use `h_ref`;
3. if `rho_min > tau`, report no certified window.

Empirical accuracy and MSE sweeps are validation targets, not selectors.

## Legacy simple2pt h-star

Several historical experiments used:

```text
h_old = 0.5 * sqrt(Delta_eff * G_hat / (L_hat * sqrt(K_u)))
```

with:

```text
K_u = d_eff^2                  # Rademacher
K_u = d_eff * (d_eff + 2)      # Gaussian
```

Recent diagnostics distinguish:

```text
c_raw(u) = [L(theta+rho u)-2L(theta)+L(theta-rho u)] / rho^2 ~ u^T H u
L_unit   = median(|c_raw| / ||u||^2)
kappa2   = sqrt(E[c_raw^2])
```

If `L_raw` is used directly and `sqrt(K_u)` is also included, the dimension is
double-counted. The checked clean-L path divides by `||u||^2`.
"""
    (docs / "FORMULAS_AND_ALGORITHMS.md").write_text(textwrap.dedent(formulas).strip() + "\n", encoding="utf-8")

    params = r"""
# Parameter Dictionary

Common fields:

- `model`, `model_id`: model name, e.g. `roberta-large`, `facebook/opt-1.3b`.
- `task`, `task_name`, `dataset`: dataset/task, e.g. SST-2, SST-5, RTE, MNLI, TREC.
- `dataset_mode`: `full` or few-shot/k-shot mode.
- `seed`: model/random-direction/training seed.
- `data_seed`: data split and dataloader seed.
- `batch_size`, `per_device_train_batch_size`: training batch size.
- `h`: finite-difference perturbation radius.
- `h_label`: policy label, e.g. `standard_1e-3`, `hstar_cont`, `fixed_small`.
- `precision`, `precision_mode`: FP32/FP16/BF16/INT8/INT4.
- `quantizer`, `quantizer_backend`: low-bit forward oracle.
- `group_size`: quantization group size, usually 128.
- `scale_refresh_k`: how often RTNClip grids are recomputed.
- `pair_shared_grid`: plus/minus finite-difference pair shares same quantizer grid.
- `fresh_round_codes`: perturbed plus/minus weights are freshly rounded.
- `update_backend`: e.g. `fp16_master`.
- `direction`: dense or sparse direction family.
- `perturb_scope`: parameters perturbed/updated.
- `sparse_p`: sparse active fraction.
- `mask_strategy`: method for choosing sparse active parameters.
- `best_eval_acc`, `last_eval_acc`: best and last dev/eval accuracy.
- `best_eval_loss`, `last_eval_loss`: best and last eval loss.
- `run_type`: full/medium/pilot when available.

Probe fields:

- `fd_true_nmse`, `default_fd_true_nmse`: normalized true directional MSE when
  verified as `E[(d_Q-d_star)^2]/E[d_star^2]`.
- `corr`, `default_corr_fd_true`: correlation between finite difference and
  reference directional derivative.
- `active_frac`: fraction of quantized coordinates with nonzero effective
  displacement.
- `alignment`: cosine between `Delta_Q` and `2hu`.
- `norm_ratio`: `||Delta_Q||/||2hu||`.
- `chi`: gain distortion `mean((Delta_Q/(2hu)-1)^2)`.
- `Delta_eff`: effective quantization step used in h-star formulas.
- `G_hat`: directional derivative scale estimate.
- `L_hat`, `L_unit`, `L_raw`, `kappa2`: curvature estimates.

Status fields:

- `complete`: intended run finished and wrote summary.
- `running/incomplete`: outputs exist but no final summary.
- `failed`: run reported failure or NaN.
- `pilot`/`medium`: shorter validation run, not full comparable result.
"""
    (docs / "PARAMETER_DICTIONARY.md").write_text(textwrap.dedent(params).strip() + "\n", encoding="utf-8")

    highlights = r"""
# Result Highlights And Caveats

This file summarizes notable findings visible in the collected outputs.  Use
the CSV indices for exact provenance.

## RoBERTa INT4 activation/visibility

Recent RoBERTa-large/SST-5/INT4 RTNClip final-checkpoint diagnostics showed:

- `h=5e-4`: actual active fraction about 5.7%;
- `h=7e-4`: actual active fraction about 8.1%;
- `h=1e-3`: actual active fraction about 11.7%;
- `h=2e-3`: actual active fraction about 24.1%;
- `h=3e-3`: actual active fraction about 35.3%.

Thus the default `1e-3` sits near the visibility edge for RoBERTa INT4.

## OPT/SST-2 INT4 multi-seed status at package time

The indexed multi-seed lane compares:

- hstar_cont around `5.09e-4`;
- default `1e-3`;
- large `3e-3`.

At the time of packaging, seed 17 hstar completed and showed high early best
accuracy followed by late collapse; seed 42 hstar was still incomplete if no
summary is present. Check `indices/run_index.csv` and the corresponding eval
JSONL files for final status.

## OPT L dimension-correction diagnostic

The raw directional curvature satisfies:

```text
L_raw / L_unit ≈ mean(||u||^2)
```

so raw curvature includes the unnormalized direction-length factor. The checked
clean-L helper divides by `||u||^2`; the remaining issue is sensitivity to
checkpoint, rho, and INT4 visibility, not a simple raw-L double-count bug in the
current clean-L path.

## Paper artifact caveats

Several historical figures used geometry/proxy metrics where true directional
MSE was later required. The final paper artifact packages contain reconciliation
and validation reports; use those final manifests for paper-facing figures.

## Do not overclaim

The collected results support conservative statements:

- default `h=1e-3` is strong when it lies inside a broad/visible window;
- fixed-small h often fails in low-bit regimes;
- analytical/reference h can help in narrow or extreme low-precision cases;
- no single h policy universally beats default across all tasks and modes.
"""
    (docs / "RESULT_HIGHLIGHTS_AND_CAVEATS.md").write_text(textwrap.dedent(highlights).strip() + "\n", encoding="utf-8")


def package_results(output_name: str, max_mb: int) -> Tuple[Path, Path]:
    package_dir = REPO_ROOT / output_name
    zip_path = REPO_ROOT / f"{output_name}.zip"
    if package_dir.exists():
        shutil.rmtree(package_dir)
    if zip_path.exists():
        zip_path.unlink()
    (package_dir / "artifacts").mkdir(parents=True)
    (package_dir / "indices").mkdir(parents=True)

    max_bytes = int(max_mb * 1024 * 1024)
    manifest_rows: List[Dict[str, object]] = []
    excluded_rows: List[Dict[str, object]] = []

    for src in iter_candidate_files():
        if not src.exists() or not src.is_file():
            continue
        rel = src.relative_to(REPO_ROOT)
        include, reason = include_decision(src, max_bytes)
        size = src.stat().st_size
        if include:
            dst = package_dir / "artifacts" / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            manifest_rows.append(
                {
                    "source_path": str(rel),
                    "package_path": str(dst.relative_to(package_dir)),
                    "size_bytes": size,
                    "sha256": sha256_file(src),
                    "artifact_kind": classify_file(src),
                    "status": "included",
                    "reason": reason,
                }
            )
        else:
            excluded_rows.append(
                {
                    "source_path": str(rel),
                    "size_bytes": size,
                    "suffix": src.suffix,
                    "status": "excluded",
                    "reason": reason,
                }
            )

    run_rows = build_run_index()
    family_rows = build_experiment_family_index()
    write_csv(package_dir / "indices" / "artifact_manifest.csv", manifest_rows)
    write_csv(package_dir / "indices" / "excluded_files.csv", excluded_rows)
    write_csv(package_dir / "indices" / "run_index.csv", run_rows)
    write_csv(package_dir / "indices" / "experiment_family_index.csv", family_rows)

    metadata = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "repo_root": str(REPO_ROOT),
        "git_commit": git_commit(),
        "hostname": socket.gethostname(),
        "python": sys.executable,
        "python_version": platform.python_version(),
        "max_file_mb": max_mb,
        "included_artifacts": len(manifest_rows),
        "excluded_files": len(excluded_rows),
        "indexed_runs": len(run_rows),
        "indexed_experiment_roots": len(family_rows),
    }
    write_json(package_dir / "metadata.json", metadata)
    write_docs(package_dir, len(run_rows), len(manifest_rows), len(excluded_rows))

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1) as zf:
        for file in package_dir.rglob("*"):
            if file.is_file():
                zf.write(file, file.relative_to(REPO_ROOT))
    return package_dir, zip_path


def main() -> int:
    output_name = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_OUT
    package_dir, zip_path = package_results(output_name, MAX_FILE_MB)
    print(f"Package directory: {package_dir}")
    print(f"Zip archive: {zip_path}")
    print(f"Zip size bytes: {zip_path.stat().st_size}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
