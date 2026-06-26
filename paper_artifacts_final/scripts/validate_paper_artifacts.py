#!/usr/bin/env python3
"""Validate the final h-window paper artifact package."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def fail(msgs: list[str], msg: str) -> None:
    msgs.append(f"FAIL: {msg}")


def ok(msgs: list[str], msg: str) -> None:
    msgs.append(f"PASS: {msg}")


def exists(root: Path, rel: str) -> bool:
    return (root / rel).exists()


def main() -> int:
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "paper_artifacts_final"
    report: list[str] = ["# Validation Report", ""]
    errors: list[str] = []

    fig_path = root / "FIGURE_DATA_MANIFEST.csv"
    tab_path = root / "TABLE_DATA_MANIFEST.csv"
    if not fig_path.exists():
        fail(errors, "FIGURE_DATA_MANIFEST.csv missing")
    if not tab_path.exists():
        fail(errors, "TABLE_DATA_MANIFEST.csv missing")
    if errors:
        (root / "VALIDATION_REPORT.md").write_text("\n".join(report + errors) + "\n")
        return 1

    figs = pd.read_csv(fig_path)
    tabs = pd.read_csv(tab_path)

    # 1-2. MSE figures must use the canonical target and not proxy metrics.
    for _, r in figs.iterrows():
        text = " ".join(str(r.get(c, "")) for c in ["figure_id", "filename_pdf", "metric_definition"]).lower()
        if "mse" in text and "visibility" not in text:
            if "true_directional_nmse" not in text and "true directional" not in text and "surrogate mse" not in text:
                fail(errors, f"MSE figure {r.get('figure_id')} does not declare true/surrogate target")
            bad = ["a_cross", "a_interval", "sigma_raw2", "lowbit_true_nmse", "delta_visibility"]
            if any(b in text for b in bad):
                fail(errors, f"MSE figure {r.get('figure_id')} mentions proxy metric in its target definition")
    if not errors:
        ok(report, "MSE figure metric declarations are canonical or explicitly analytical surrogate")

    # 3. INT4 true-MSE min h must match source CSV and manifest statement.
    true_csv = root / "data" / "processed" / "sst5_true_directional_mse.csv"
    if true_csv.exists():
        true_df = pd.read_csv(true_csv)
        i4 = true_df[true_df["precision"] == "int4"]
        if i4.empty:
            fail(errors, "INT4 true-MSE processed data missing")
        else:
            best = i4.loc[i4["true_directional_nmse"].astype(float).idxmin()]
            if not math.isclose(float(best["h"]), 0.002, rel_tol=1e-9, abs_tol=1e-12):
                fail(errors, f"INT4 true-MSE minimum expected 2e-3, got {best['h']}")
            else:
                ok(report, "INT4 true-MSE minimum matches canonical CSV at h=2e-3")
    else:
        fail(errors, "sst5_true_directional_mse.csv missing")

    # 4. Figure source files exist.
    for _, r in figs.iterrows():
        for col in ["filename_pdf", "filename_png", "processed_source_file"]:
            raw = str(r.get(col, ""))
            if not raw or raw == "nan":
                fail(errors, f"{r.get('figure_id')} missing {col}")
                continue
            for part in [p.strip() for p in raw.split(";") if p.strip()]:
                if col.startswith("filename") and not exists(root, part):
                    fail(errors, f"Figure file missing: {part}")
                if col == "processed_source_file" and not exists(root, part):
                    fail(errors, f"Processed source missing for {r.get('figure_id')}: {part}")
    ok(report, "Figure files and processed sources checked")

    # 5. INT4 rho_min>1 must not draw theoretical tau=1 interval.
    precision_csv = root / "data" / "processed" / "precision_window_theory_vs_empirical.csv"
    if precision_csv.exists():
        pw = pd.read_csv(precision_csv)
        i4 = pw[pw["precision"] == "int4"]
        if not i4.empty:
            row = i4.iloc[0]
            if float(row["rho_min"]) > 1 and str(row.get("theoretical_window", "")).lower() != "none":
                fail(errors, "INT4 rho_min>1 but theoretical interval is present")
            else:
                ok(report, "INT4 has no tau=1 theoretical interval when rho_min>1")
    else:
        fail(errors, "precision_window_theory_vs_empirical.csv missing")

    # 6. Analytic Panel C uses empirical center, not theoretical h_ref.
    analytic_table = root / "data" / "processed" / "table_analytic_window.csv"
    if analytic_table.exists():
        at = pd.read_csv(analytic_table)
        if any("emp_center" in c for c in at.columns) and not any(c.startswith("h_ref_slope") for c in at.columns):
            ok(report, "Analytic scaling table uses empirical center columns")
        else:
            fail(errors, "Analytic scaling table appears to use non-empirical slope columns")
    else:
        fail(errors, "table_analytic_window.csv missing")

    # 7-9. RoBERTa table policy/run-type checks.
    rob_csv = root / "data" / "processed" / "roberta_int4_multitask_main.csv"
    if rob_csv.exists():
        rob = pd.read_csv(rob_csv)
        if rob["canonical_policy"].isna().any() or (rob["canonical_policy"].astype(str).str.len() == 0).any():
            fail(errors, "RoBERTa main table has empty policy")
        dup = rob.duplicated(["task", "mode", "canonical_policy"]).any()
        if dup:
            fail(errors, "RoBERTa main table has duplicate task/mode/policy")
        if set(rob["run_type"].dropna().astype(str)) != {"full"}:
            fail(errors, "RoBERTa main table mixes full with non-full runs")
        if not (rob["source_path"].astype(str).map(lambda p: (ROOT / p).exists()).all()):
            fail(errors, "Some RoBERTa source_path values do not exist")
        if not errors:
            ok(report, "RoBERTa main table policy/full/source checks passed")
    else:
        fail(errors, "roberta_int4_multitask_main.csv missing")

    # 10. OPT table retains TREC.
    opt_csv = root / "data" / "processed" / "opt_cross_arch_main.csv"
    if opt_csv.exists():
        opt = pd.read_csv(opt_csv)
        if "trec" not in set(opt["task"].astype(str).str.lower()):
            fail(errors, "OPT table does not retain TREC")
        else:
            ok(report, "OPT table retains TREC")
    else:
        fail(errors, "opt_cross_arch_main.csv missing")

    # 11. source_path values exist for processed CSVs that carry them.
    for csv_path in (root / "data" / "processed").glob("*.csv"):
        df = pd.read_csv(csv_path)
        for col in ["source_path", "source_log"]:
            if col in df.columns:
                for val in df[col].dropna().astype(str).unique():
                    if val and not (ROOT / val).exists():
                        fail(errors, f"Missing source path from {csv_path.name}: {val}")
    if not errors:
        ok(report, "All checked source_path/source_log values exist")

    # 12. radius names separate current/legacy/training.
    rad_csv = root / "data" / "processed" / "radius_provenance.csv"
    if rad_csv.exists():
        rad = pd.read_csv(rad_csv)
        kinds = set(rad["radius_kind"].astype(str))
        if not {"h_ref_current", "legacy_hstar", "training_h"}.issubset(kinds):
            fail(errors, f"Radius provenance missing required kinds: {kinds}")
        else:
            ok(report, "Radius provenance separates h_ref_current, legacy_hstar, and training_h")
    else:
        fail(errors, "radius_provenance.csv missing")

    # 13. captions/metric definitions consistency via manifest.
    for _, r in figs.iterrows():
        if not str(r.get("metric_definition", "")).strip():
            fail(errors, f"Figure {r.get('figure_id')} missing metric definition")
    for _, r in tabs.iterrows():
        if not str(r.get("metric_definition", "")).strip():
            fail(errors, f"Table {r.get('table_id')} missing metric definition")

    # 14. TeX referenced files exist.
    for _, r in tabs.iterrows():
        tex = str(r.get("filename_tex", ""))
        if tex and not exists(root, tex):
            fail(errors, f"Table TeX missing: {tex}")
    if not errors:
        ok(report, "Manifest TeX/table files exist")

    if errors:
        report.extend(["", "## Failures", *errors])
    else:
        report.extend(["", "## Result", "All validation checks passed."])

    (root / "VALIDATION_REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print("\n".join(report))
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
