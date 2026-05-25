#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import math
import statistics
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from tools import int4_window_preflight_probe as probe
from tools import smoke_rtnclip_roberta_sst5 as smoke


def corr(pairs):
    xs = [a for a, _ in pairs]
    ys = [b for _, b in pairs]
    mx = sum(xs) / len(xs)
    my = sum(ys) / len(ys)
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    if vx <= 0.0 or vy <= 0.0:
        return None
    return sum((x - mx) * (y - my) for x, y in pairs) / math.sqrt(vx * vy)


def nmse(pairs):
    return sum((a - b) ** 2 for a, b in pairs) / max(sum(b * b for _, b in pairs), 1e-12)


def mean(rows, key):
    vals = []
    for row in rows:
        try:
            val = float(row.get(key, ""))
        except Exception:
            continue
        if math.isfinite(val):
            vals.append(val)
    return statistics.mean(vals) if vals else None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", default="outputs/int4_window_preflight/probe_sst5_fisher_dense/dense/probe_records.csv")
    parser.add_argument("--out_csv", default="outputs/int4_window_preflight/probe_sst5_fisher_dense/dense/probe_results_corrected_legacy_vs_visibility.csv")
    parser.add_argument("--out_md", default="outputs/int4_window_preflight/probe_sst5_fisher_dense/dense/probe_results_corrected_legacy_vs_visibility.md")
    parser.add_argument("--seed", type=int, default=16)
    parser.add_argument("--data_seed", type=int, default=16)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--directions", type=int, default=16)
    args = parser.parse_args()

    records = list(csv.DictReader(Path(args.records).open(newline="", encoding="utf-8")))

    class LoadArgs:
        pass

    load_args = LoadArgs()
    load_args.seed = args.seed
    load_args.data_seed = args.data_seed
    load_args.batch_size = args.batch_size

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    model, train_loader, _dev_loader, _data_args, _sampler = probe.load_base(load_args, device)
    params = smoke.named_parameter_map(model)
    master = probe.make_master(params, device)
    q_names = [name for name in smoke.linear_weight_names(model) if name in master and "prefix" not in name]
    states, _ = smoke.refresh_quantizer_states(master, q_names, 4, 128)
    batch = smoke.move_batch(next(iter(train_loader)), device)
    probe.set_probe_grad_flags(params, set(master.keys()))
    probe.compute_quantized_grad(model, params, master, states, batch)

    legacy_by_k = {}
    for k in range(args.directions):
        directions = probe.sample_dense(master, args.seed * 1000003 + k)
        val = torch.zeros((), device=device, dtype=torch.float64)
        for name, param in params.items():
            if name in master and name in directions and param.grad is not None:
                val += (param.grad.detach().float() * directions[name].float()).double().sum()
        legacy_by_k[k] = float(val.detach().cpu())

    rows = []
    for h in sorted({float(r["h"]) for r in records}):
        group = [r for r in records if abs(float(r["h"]) - h) < 1e-18]
        legacy_pairs = []
        eff_pairs = []
        for r in group:
            k = int(r["k_dir"])
            d_h = float(r["d_h"])
            legacy_pairs.append((d_h, legacy_by_k[k]))
            eff_pairs.append((d_h, float(r["d_true"])))
        rows.append(
            {
                "h": h,
                "legacy_fd_true_nmse_dh_vs_gTu": nmse(legacy_pairs),
                "legacy_corr_dh_vs_gTu": corr(legacy_pairs),
                "fd_eff_nmse_dh_vs_grad_deltaq_over_2h": nmse(eff_pairs),
                "fd_eff_corr": corr(eff_pairs),
                "visibility_nmse_deltaq_vs_2hu": mean(group, "delta_visibility_nmse"),
                "active_frac": mean(group, "active_frac"),
                "alignment_deltaq_2hu": mean(group, "alignment"),
                "norm_ratio_deltaq_2hu": mean(group, "norm_ratio"),
            }
        )

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# Corrected INT4 Dense Few-Shot Probe",
        "",
        "| h | legacy nMSE d_h vs gTu | legacy corr | visibility nMSE DeltaQ vs 2hu | active_frac | fd_eff_nMSE |",
        "| ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in rows:
        legacy_corr = r["legacy_corr_dh_vs_gTu"]
        lines.append(
            f"| {r['h']:.6g} | {r['legacy_fd_true_nmse_dh_vs_gTu']:.6g} | "
            f"{'NA' if legacy_corr is None else f'{legacy_corr:.6g}'} | "
            f"{r['visibility_nmse_deltaq_vs_2hu']:.6g} | {r['active_frac']:.6g} | "
            f"{r['fd_eff_nmse_dh_vs_grad_deltaq_over_2h']:.6g} |"
        )
    Path(args.out_md).write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"wrote {out_csv}")
    print(f"wrote {args.out_md}")
    for r in rows:
        print(
            f"h={r['h']:.6g} legacy_nmse={r['legacy_fd_true_nmse_dh_vs_gTu']:.6g} "
            f"legacy_corr={r['legacy_corr_dh_vs_gTu']:.6g} "
            f"visibility_nmse={r['visibility_nmse_deltaq_vs_2hu']:.6g} "
            f"active={r['active_frac']:.6g} "
            f"fd_eff={r['fd_eff_nmse_dh_vs_grad_deltaq_over_2h']:.6g}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
