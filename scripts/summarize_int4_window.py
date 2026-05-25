#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def rows_from_csv(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: List[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def h_key(value: Any) -> Optional[float]:
    try:
        return float(value)
    except Exception:
        return None


def pick_row(rows: List[Dict[str, Any]], h: float) -> Dict[str, Any]:
    for row in rows:
        hv = h_key(row.get("h"))
        if hv is not None and abs(hv - h) <= max(1e-18, abs(h) * 1e-10):
            return row
    return {}


def summarize_probe_dir(root: Path, rel: str) -> List[Dict[str, Any]]:
    base = root / rel
    out: List[Dict[str, Any]] = []
    for path in sorted(base.glob("*/probe_results.csv")):
        setting = path.parent.name
        hstar_path = path.parent / "hstar_summary.json"
        hstar = read_json(hstar_path) if hstar_path.exists() else {}
        for row in rows_from_csv(path):
            item = dict(row)
            item["probe_root"] = rel
            item["setting"] = item.get("setting") or setting
            item["h_star"] = hstar.get("h_star")
            item["window_low"] = hstar.get("window_low")
            item["window_high"] = hstar.get("window_high")
            item["membership_1e-5"] = hstar.get("membership_1e-5")
            item["membership_1e-3"] = hstar.get("membership_1e-3")
            item["membership_hstar"] = hstar.get("membership_hstar")
            item["fisher_default_path_status"] = hstar.get("fisher_default_path_status")
            out.append(item)
    return out


def summarize_train(root: Path) -> Dict[str, Any]:
    run_dir = root / "train_sst5_fisher_dense_int4_h1e-3"
    summary_path = run_dir / "run_summary.json"
    if summary_path.exists():
        summary = read_json(summary_path)
    else:
        metrics = rows_from_csv(run_dir / "metrics.csv")
        eval_rows = []
        eval_path = run_dir / "eval_metrics.jsonl"
        if eval_path.exists():
            for line in eval_path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    eval_rows.append(json.loads(line))
        if metrics:
            last = metrics[-1]
            summary = {
                "status": "running",
                "steps_completed": int(float(last.get("step", 0) or 0)),
                "final_train_loss": float(last["train_loss"]) if last.get("train_loss") else None,
                "last_eval_acc": eval_rows[-1].get("eval_acc") if eval_rows else None,
                "last_eval_loss": eval_rows[-1].get("eval_loss") if eval_rows else None,
                "last_eval_step": eval_rows[-1].get("step") if eval_rows else None,
                "best_eval_acc": max((r.get("eval_acc") for r in eval_rows if r.get("eval_acc") is not None), default=None),
                "best_eval_loss": min((r.get("eval_loss") for r in eval_rows if r.get("eval_loss") is not None), default=None),
            }
        else:
            summary = {"status": "not_started"}
    hstar_path = root / "probe_sst5_fisher_dense" / "dense" / "hstar_summary.json"
    hstar = read_json(hstar_path) if hstar_path.exists() else {}
    item = dict(summary)
    item["run_dir"] = str(run_dir)
    item["train_summary_created_by"] = "scripts/summarize_int4_window.py"
    item["fisher_default_path_status"] = hstar.get("fisher_default_path_status", "unknown")
    item["membership_1e-3"] = hstar.get("membership_1e-3", "unknown")
    item["h_star_from_preflight"] = hstar.get("h_star")
    item["window_low_from_preflight"] = hstar.get("window_low")
    item["window_high_from_preflight"] = hstar.get("window_high")
    write_json(run_dir / "train_summary.json", item)
    write_csv(run_dir / "train_summary.csv", [item])
    return item


def write_markdown(root: Path, probe_rows: List[Dict[str, Any]], train: Dict[str, Any]) -> None:
    lines = [
        "# INT4 Window Preflight Summary",
        "",
        f"Output root: `{root}`",
        "",
        "## Probe Key Rows",
        "",
        "| setting | h* | window | h=1e-5 | h=1e-3 | nMSE@1e-5 | nMSE@1e-3 | corr@1e-5 | corr@1e-3 |",
        "| --- | ---: | --- | --- | --- | ---: | ---: | ---: | ---: |",
    ]
    by_setting: Dict[str, List[Dict[str, Any]]] = {}
    for row in probe_rows:
        if row.get("probe_root") == "probes_sst5_all_settings":
            by_setting.setdefault(str(row.get("setting")), []).append(row)
    for setting, rows in sorted(by_setting.items()):
        r_small = pick_row(rows, 1e-5)
        r_default = pick_row(rows, 1e-3)
        first = rows[0]
        window = f"[{first.get('window_low')}, {first.get('window_high')}]"
        lines.append(
            f"| {setting} | {first.get('h_star')} | {window} | {first.get('membership_1e-5')} | "
            f"{first.get('membership_1e-3')} | {r_small.get('fd_true_nmse', '')} | "
            f"{r_default.get('fd_true_nmse', '')} | {r_small.get('corr', '')} | {r_default.get('corr', '')} |"
        )
    lines.extend(
        [
            "",
            "## Training Smoke",
            "",
            f"Run dir: `{train.get('run_dir')}`",
            f"Status: `{train.get('status')}`",
            f"Steps completed: `{train.get('steps_completed')}`",
            f"Best dev/eval acc: `{train.get('best_eval_acc')}`",
            f"Final/last dev acc: `{train.get('last_eval_acc')}`",
            f"Fisher/default-path status: `{train.get('fisher_default_path_status')}`",
            f"h=1e-3 window membership: `{train.get('membership_1e-3')}`",
            "",
        ]
    )
    (root / "summary_int4_window_preflight.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_root", default="outputs/int4_window_preflight")
    args = parser.parse_args()
    root = Path(args.output_root)
    probe_rows = []
    probe_rows.extend(summarize_probe_dir(root, "probe_sst5_fisher_dense"))
    probe_rows.extend(summarize_probe_dir(root, "probes_sst5_all_settings"))
    write_csv(root / "summary_probe_table.csv", probe_rows)
    train = summarize_train(root)
    write_markdown(root, probe_rows, train)
    write_json(root / "summary_int4_window_preflight.json", {"probe_rows": probe_rows, "train": train})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
