#!/usr/bin/env python
"""Calibrate direct nMSE bounds against observed d_h - g^T u errors.

This is a diagnostic helper, not a training launcher.  It reads existing probe
records, computes the observed default finite-difference nMSE

    sum((d_h - g^T u)^2) / sum((g^T u)^2),

and compares two Delta choices in the direct bound:

    B(h) = Delta^2/(4h^2)
         + 2 Delta L sqrt(d(d+2))/G
         + 4 h^2 L^2 d(d+2)/G^2.

The first Delta is the configured parameter/grid Delta when available.  The
second is an empirical effective FD Delta inferred from the left tail:

    Delta_eff(h) = 2 h sqrt(nMSE_observed(h)).

The point is to audit whether a proposed Delta convention is close to the real
`d_h - g^T u` error across dense, sparse, prefix, and FP16 full-parameter paths.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


EPS = 1e-30


@dataclass(frozen=True)
class SourceSpec:
    source: str
    path: Path
    fmt: str
    setting: str
    metadata_kind: str = ""


DEFAULT_SOURCES = [
    SourceSpec(
        "prefix_int4_base_sst5",
        Path("outputs/rtnclip_int4_prefix_fd_nmse_bound_20260522_185011/probe_records.csv"),
        "csv",
        "prefix",
        "prefix_components",
    ),
    SourceSpec(
        "fp16_mnli_fullparam",
        Path("outputs/fp16_mnli_roberta_short_h_mse/mse_probe_records.jsonl"),
        "jsonl",
        "fp16_fullparam",
        "",
    ),
    SourceSpec(
        "int4_dense_full_tasks",
        Path("outputs/rtnclip_int4_roberta_full_dataset_hstar_20260521/hstar_probe_records.csv"),
        "csv",
        "dense",
        "dense_hstar_csv",
    ),
    SourceSpec(
        "int4_sparse_p0p1_taskgrad_full_tasks",
        Path("outputs/int4_sparsep0p1_probe_minmse_vs_default_2k_20260522_181148/probes_sparse_p0p1_minmse"),
        "jsonl_dir",
        "sparse_p0p1_taskgrad",
        "sparse_hstar_csv",
    ),
]


def finite_float(value: object) -> Optional[float]:
    try:
        if value is None or value == "":
            return None
        out = float(value)
    except Exception:
        return None
    return out if math.isfinite(out) else None


def corr(xs: Sequence[float], ys: Sequence[float]) -> Optional[float]:
    if len(xs) < 2 or len(xs) != len(ys):
        return None
    mx = sum(xs) / len(xs)
    my = sum(ys) / len(ys)
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    if vx <= 0.0 or vy <= 0.0:
        return None
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / math.sqrt(vx * vy)


def read_csv(path: Path) -> Iterable[Dict[str, object]]:
    with path.open(newline="", encoding="utf-8") as f:
        yield from csv.DictReader(f)


def read_jsonl(path: Path) -> Iterable[Dict[str, object]]:
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def iter_source_rows(spec: SourceSpec) -> Iterable[Dict[str, object]]:
    if spec.fmt == "csv":
        yield from read_csv(spec.path)
    elif spec.fmt == "jsonl":
        yield from read_jsonl(spec.path)
    elif spec.fmt == "jsonl_dir":
        for path in sorted(spec.path.glob("*/probe_records.jsonl")):
            for row in read_jsonl(path):
                row = dict(row)
                row.setdefault("_dataset_from_path", path.parent.name)
                yield row
    else:
        raise ValueError(f"Unsupported source format: {spec.fmt}")


def pick_first(row: Dict[str, object], keys: Sequence[str]) -> Optional[float]:
    for key in keys:
        val = finite_float(row.get(key))
        if val is not None:
            return val
    return None


def normalize_record(spec: SourceSpec, row: Dict[str, object]) -> Optional[Dict[str, object]]:
    h = pick_first(row, ["h", "h_raw"])
    dh = pick_first(row, ["d_h", "d_h_Q", "d_fd"])
    dtrue = pick_first(row, ["d_true", "d_true_default", "gTu"])
    if h is None or dh is None or dtrue is None:
        return None
    dataset = str(row.get("task_name") or row.get("dataset") or row.get("_dataset_from_path") or "")
    if not dataset:
        dataset = "unknown"
    # Prefix probe writes adapter=SST only; make the dataset explicit.
    if spec.setting == "prefix":
        dataset = "sst-5"
    elif spec.source == "fp16_mnli_fullparam":
        dataset = "mnli"
    loss_plus = finite_float(row.get("loss_plus"))
    loss_minus = finite_float(row.get("loss_minus"))
    return {
        "source": spec.source,
        "setting": spec.setting,
        "dataset": dataset.lower(),
        "h": h,
        "d_h": dh,
        "d_true": dtrue,
        "loss_plus": loss_plus,
        "loss_minus": loss_minus,
    }


def group_key(row: Dict[str, object]) -> Tuple[str, str, str]:
    return str(row["source"]), str(row["setting"]), str(row["dataset"])


def grouped(records: Iterable[Dict[str, object]]) -> Dict[Tuple[str, str, str], List[Dict[str, object]]]:
    out: Dict[Tuple[str, str, str], List[Dict[str, object]]] = {}
    for record in records:
        out.setdefault(group_key(record), []).append(record)
    return out


def records_by_h(records: List[Dict[str, object]]) -> Dict[float, List[Dict[str, object]]]:
    out: Dict[float, List[Dict[str, object]]] = {}
    for record in records:
        out.setdefault(float(record["h"]), []).append(record)
    return out


def pooled_stats(rows: List[Dict[str, object]]) -> Dict[str, object]:
    pairs = [(float(r["d_h"]), float(r["d_true"])) for r in rows]
    if not pairs:
        return {}
    err2 = sum((a - b) ** 2 for a, b in pairs)
    true2 = sum(b * b for _, b in pairs)
    dh2 = sum(a * a for a, _ in pairs)
    n = len(pairs)
    zero_dh = sum(1 for a, _ in pairs if a == 0.0)
    exact_loss = sum(
        1
        for r in rows
        if r.get("loss_plus") is not None
        and r.get("loss_minus") is not None
        and float(r["loss_plus"]) == float(r["loss_minus"])
    )
    return {
        "n": n,
        "observed_nmse": err2 / max(true2, EPS),
        "observed_mse": err2 / n,
        "g_rms": math.sqrt(true2 / n),
        "d_h_rms": math.sqrt(dh2 / n),
        "err_rms": math.sqrt(err2 / n),
        "corr": corr([a for a, _ in pairs], [b for _, b in pairs]),
        "zero_dh_frac": zero_dh / n,
        "exact_loss_frac": exact_loss / n,
    }


def read_hstar_csv(path: Path, direction_mode: str) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    if not path.exists():
        return out
    for row in read_csv(path):
        if direction_mode and str(row.get("direction_mode", "")).lower() != direction_mode:
            continue
        dataset = str(row.get("task_name") or row.get("dataset") or "").lower()
        if not dataset:
            continue
        out[dataset] = {
            "delta_param": finite_float(row.get("Delta_value")),
            "g_metadata": finite_float(row.get("G_value")),
            "l_hat": finite_float(row.get("L_hat")),
            "d_trainable": finite_float(row.get("d_trainable")),
            "delta_mode": row.get("Delta_mode", ""),
            "g_mode": row.get("G_mode", ""),
            "l_mode": row.get("L_mode", ""),
        }
    return out


def load_metadata() -> Dict[Tuple[str, str, str], Dict[str, object]]:
    meta: Dict[Tuple[str, str, str], Dict[str, object]] = {}
    prefix_path = Path("outputs/rtnclip_int4_prefix_fd_nmse_bound_20260522_185011/direct_formula_bound_components.json")
    if prefix_path.exists():
        data = json.loads(prefix_path.read_text(encoding="utf-8"))
        meta[("prefix_int4_base_sst5", "prefix", "sst-5")] = {
            "delta_param": finite_float(data.get("Delta", {}).get("Delta_value")),
            "g_metadata": finite_float(data.get("G", {}).get("G_rms")),
            "g_abs_metadata": finite_float(data.get("G", {}).get("G_abs")),
            "l_hat": finite_float(data.get("L", {}).get("selected", {}).get("lambda_q90")),
            "d_trainable": finite_float(data.get("d_trainable")),
            "delta_mode": data.get("Delta", {}).get("Delta_mode", ""),
            "g_mode": "prefix_record_g_rms",
            "l_mode": "prefix_quantized_base_second_diff_lambda_q90",
        }
    dense = read_hstar_csv(
        Path("outputs/int4_dense_hstar_cont_vs_default_2k_20260522_163849/hstar_dense_lowbitG/hstar_full_data_summary.csv"),
        "dense",
    )
    for dataset, row in dense.items():
        meta[("int4_dense_full_tasks", "dense", dataset)] = row
    sparse = read_hstar_csv(
        Path("outputs/int4_full_data_hstar_dense_sparse_20260522_113710/hstar_sparse_p0p1_taskgrad_lowbitG_20260522_162254/hstar_full_data_summary.csv"),
        "sparse",
    )
    for dataset, row in sparse.items():
        meta[("int4_sparse_p0p1_taskgrad_full_tasks", "sparse_p0p1_taskgrad", dataset)] = row
    return meta


def bound_nmse(delta: Optional[float], h: float, g: float, l_hat: Optional[float], d_trainable: Optional[float]) -> Optional[float]:
    if delta is None or delta <= 0.0 or h <= 0.0 or g <= 0.0:
        return None
    small = delta * delta / (4.0 * h * h)
    if l_hat is None or d_trainable is None or l_hat <= 0.0 or d_trainable <= 0.0:
        return small
    dim = float(d_trainable)
    root = math.sqrt(dim * (dim + 2.0))
    cross = 2.0 * delta * l_hat * root / g
    large = 4.0 * h * h * l_hat * l_hat * dim * (dim + 2.0) / (g * g)
    return small + cross + large


def log10_error(pred: Optional[float], obs: Optional[float]) -> Optional[float]:
    if pred is None or obs is None or pred <= 0.0 or obs <= 0.0:
        return None
    return abs(math.log10(pred / obs))


def summarize_group(
    key: Tuple[str, str, str],
    records: List[Dict[str, object]],
    meta: Dict[Tuple[str, str, str], Dict[str, object]],
    left_tail_points: int,
) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    by_h = records_by_h(records)
    h_stats: Dict[float, Dict[str, object]] = {h: pooled_stats(rows) for h, rows in by_h.items()}
    hs = sorted(h for h, stats in h_stats.items() if finite_float(stats.get("observed_nmse")) is not None and h > 0.0)
    delta_eff_by_h = {
        h: 2.0 * h * math.sqrt(float(h_stats[h]["observed_nmse"]))
        for h in hs
        if float(h_stats[h]["observed_nmse"]) > 0.0
    }
    left = [delta_eff_by_h[h] for h in hs[:left_tail_points] if h in delta_eff_by_h and math.isfinite(delta_eff_by_h[h])]
    delta_eff_left = median(left) if left else None
    best_h = min(hs, key=lambda h: float(h_stats[h]["observed_nmse"])) if hs else None
    group_meta = meta.get(key, {})
    # Use record G_rms for normalized d_h - gTu nMSE, because that is the
    # denominator of the observed metric in these probe records.
    all_stats = pooled_stats(records)
    g_rms = finite_float(all_stats.get("g_rms"))
    rows: List[Dict[str, object]] = []
    err_param: List[float] = []
    err_eff: List[float] = []
    for h in hs:
        stats = h_stats[h]
        obs = finite_float(stats.get("observed_nmse"))
        param = finite_float(group_meta.get("delta_param"))
        l_hat = finite_float(group_meta.get("l_hat"))
        d_train = finite_float(group_meta.get("d_trainable"))
        pred_param = bound_nmse(param, h, g_rms or 0.0, l_hat, d_train)
        pred_eff = bound_nmse(delta_eff_left, h, g_rms or 0.0, l_hat, d_train)
        e_param = log10_error(pred_param, obs)
        e_eff = log10_error(pred_eff, obs)
        if e_param is not None:
            err_param.append(e_param)
        if e_eff is not None:
            err_eff.append(e_eff)
        rows.append(
            {
                "source": key[0],
                "setting": key[1],
                "dataset": key[2],
                "h": h,
                "n": stats.get("n"),
                "observed_nmse": obs,
                "corr": stats.get("corr"),
                "zero_dh_frac": stats.get("zero_dh_frac"),
                "exact_loss_frac": stats.get("exact_loss_frac"),
                "g_rms_records": g_rms,
                "g_metadata": group_meta.get("g_metadata"),
                "g_metadata_over_g_rms": (
                    float(group_meta["g_metadata"]) / g_rms
                    if finite_float(group_meta.get("g_metadata")) is not None and g_rms and g_rms > 0.0
                    else None
                ),
                "delta_param": param,
                "delta_param_mode": group_meta.get("delta_mode", ""),
                "delta_eff_from_observed_h": delta_eff_by_h.get(h),
                f"delta_eff_lefttail_median_first{left_tail_points}": delta_eff_left,
                "delta_eff_over_param": (
                    delta_eff_left / param
                    if delta_eff_left is not None and param is not None and param > 0.0
                    else None
                ),
                "l_hat": l_hat,
                "d_trainable": d_train,
                "bound_nmse_param_delta": pred_param,
                "bound_nmse_eff_delta": pred_eff,
                "abs_log10_error_param_delta": e_param,
                "abs_log10_error_eff_delta": e_eff,
                "best_h_by_observed_nmse": best_h,
            }
        )
    summary = {
        "source": key[0],
        "setting": key[1],
        "dataset": key[2],
        "n_h": len(hs),
        "g_rms_records": g_rms,
        "g_metadata": group_meta.get("g_metadata"),
        "g_metadata_over_g_rms": (
            float(group_meta["g_metadata"]) / g_rms
            if finite_float(group_meta.get("g_metadata")) is not None and g_rms and g_rms > 0.0
            else None
        ),
        "delta_param": group_meta.get("delta_param"),
        "delta_param_mode": group_meta.get("delta_mode", ""),
        f"delta_eff_lefttail_median_first{left_tail_points}": delta_eff_left,
        "delta_eff_over_param": (
            delta_eff_left / float(group_meta["delta_param"])
            if delta_eff_left is not None
            and finite_float(group_meta.get("delta_param")) is not None
            and float(group_meta["delta_param"]) > 0.0
            else None
        ),
        "l_hat": group_meta.get("l_hat"),
        "d_trainable": group_meta.get("d_trainable"),
        "best_h_by_observed_nmse": best_h,
        "best_observed_nmse": h_stats[best_h]["observed_nmse"] if best_h is not None else None,
        "median_abs_log10_error_param_delta": median(err_param) if err_param else None,
        "median_abs_log10_error_eff_delta": median(err_eff) if err_eff else None,
    }
    return rows, summary


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    cols: List[str] = []
    for row in rows:
        for key in row:
            if key not in cols:
                cols.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def fmt(value: object) -> str:
    val = finite_float(value)
    if val is None:
        return "NA"
    return f"{val:.6g}"


def write_md(path: Path, group_rows: List[Dict[str, object]], per_h_rows: List[Dict[str, object]], left_tail_points: int) -> None:
    lines = [
        "# FD nMSE Bound Calibration",
        "",
        "Metric: observed nMSE is pooled `(d_h - g^T u)^2 / (g^T u)^2`.",
        f"`effective_fd_delta` is the median of `2 h sqrt(observed_nMSE)` over the first {left_tail_points} h points.",
        "",
        "## Group Summary",
        "",
        "| source | setting | dataset | G_rms | G_meta/G_rms | Delta_param | Delta_eff | eff/param | L_hat | best_h | best_nMSE | med log10 err param | med log10 err eff |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    eff_col = f"delta_eff_lefttail_median_first{left_tail_points}"
    for row in group_rows:
        lines.append(
            f"| {row['source']} | {row['setting']} | {row['dataset']} | {fmt(row.get('g_rms_records'))} | "
            f"{fmt(row.get('g_metadata_over_g_rms'))} | {fmt(row.get('delta_param'))} | {fmt(row.get(eff_col))} | "
            f"{fmt(row.get('delta_eff_over_param'))} | {fmt(row.get('l_hat'))} | {fmt(row.get('best_h_by_observed_nmse'))} | "
            f"{fmt(row.get('best_observed_nmse'))} | {fmt(row.get('median_abs_log10_error_param_delta'))} | "
            f"{fmt(row.get('median_abs_log10_error_eff_delta'))} |"
        )
    lines.extend(
        [
            "",
            "## Per-h Rows",
            "",
            "| source | dataset | h | observed nMSE | corr | zero d_h | Delta_eff(h) | bound param | bound eff |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in per_h_rows:
        lines.append(
            f"| {row['source']} | {row['dataset']} | {fmt(row.get('h'))} | {fmt(row.get('observed_nmse'))} | "
            f"{fmt(row.get('corr'))} | {fmt(row.get('zero_dh_frac'))} | {fmt(row.get('delta_eff_from_observed_h'))} | "
            f"{fmt(row.get('bound_nmse_param_delta'))} | {fmt(row.get('bound_nmse_eff_delta'))} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate FD nMSE direct bounds from existing probe records.")
    parser.add_argument("--output_dir", type=Path, default=Path("outputs/fd_nmse_bound_calibration"))
    parser.add_argument("--left_tail_points", type=int, default=3)
    args = parser.parse_args()

    records: List[Dict[str, object]] = []
    skipped: List[str] = []
    for spec in DEFAULT_SOURCES:
        if not spec.path.exists():
            skipped.append(f"{spec.source}: missing {spec.path}")
            continue
        for raw in iter_source_rows(spec):
            row = normalize_record(spec, raw)
            if row is not None:
                records.append(row)
    meta = load_metadata()
    per_h_rows: List[Dict[str, object]] = []
    group_rows: List[Dict[str, object]] = []
    for key, recs in sorted(grouped(records).items()):
        rows, summary = summarize_group(key, recs, meta, int(args.left_tail_points))
        per_h_rows.extend(rows)
        group_rows.append(summary)

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / "fd_nmse_bound_calibration_per_h.csv", per_h_rows)
    write_csv(out / "fd_nmse_bound_calibration_groups.csv", group_rows)
    write_md(out / "fd_nmse_bound_calibration.md", group_rows, per_h_rows, int(args.left_tail_points))
    (out / "run_summary.json").write_text(
        json.dumps(
            {
                "records_loaded": len(records),
                "groups": len(group_rows),
                "left_tail_points": int(args.left_tail_points),
                "skipped": skipped,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Output: {out}")
    for row in group_rows:
        print(
            f"{row['source']} {row['dataset']}: "
            f"G_rms={fmt(row.get('g_rms_records'))} "
            f"Delta_param={fmt(row.get('delta_param'))} "
            f"Delta_eff={fmt(row.get(f'delta_eff_lefttail_median_first{int(args.left_tail_points)}'))} "
            f"err_param={fmt(row.get('median_abs_log10_error_param_delta'))} "
            f"err_eff={fmt(row.get('median_abs_log10_error_eff_delta'))}"
        )


if __name__ == "__main__":
    main()
