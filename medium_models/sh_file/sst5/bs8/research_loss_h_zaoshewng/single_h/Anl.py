#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
扫描当前目录下的形如：
  sst5_singleh_eps1e-3
的子目录，解析其中的 eps，读取 16-42/metrics_logs/ 目录下的 CSV：
  - 为每个 CSV 画图：x=训练步数, y=loss, 标题=eps
  - 为每个 CSV 取最小 eval_loss
最后按 eps 的指数（log10(eps) 四舍五入到整数）画一张 “指数 vs 最小 eval_loss” 的图。
"""

import os
import re
import math
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional

# ---------- 可按需调整 ----------
ROOT = os.getcwd()    # 扫描当前目录
SUBDIR_PATTERN = r"eps(?P<eps>[-+]?(\d+(\.\d*)?|\.\d+)([eE][-+]?\d+)?)(?:$|[^0-9eE.+-].*)"  # 捕获 eps 后面的数字（含科学计数法），兼容 sst5_singleh_eps1e-3 等命名
LOG_SUBPATH = os.path.join("seed16", "metrics_logs")
OUTPUT_DIR = "plots_eps"
SUMMARY_CSV = os.path.join(OUTPUT_DIR, "summary.csv")
# 识别训练步数字段的候选列名（按优先级）
STEP_COL_CANDIDATES = ["global_step", "step", "steps", "iteration", "iter", "Step", "globalStep"]
LOSS_COL = "train_loss"
EVAL_LOSS_COL = "eval_loss"
# --------------------------------

# 平滑设置：滑动窗口大小（奇数更合适，例如 5/7/11）。设置为 1 表示不平滑。
SMOOTH_WINDOW = 150


def find_step_column(df: pd.DataFrame) -> Optional[str]:
    for col in STEP_COL_CANDIDATES:
        if col in df.columns:
            return col
    # 兜底：如果存在类似 step 的列名
    for col in df.columns:
        if "step" in col.lower():
            return col
    return None


def parse_eps_from_dirname(dirname: str) -> Optional[float]:
    m = re.search(SUBDIR_PATTERN, dirname)
    if not m:
        return None
    try:
        return float(m.group("eps"))
    except Exception:
        return None


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def main():
    ensure_dir(OUTPUT_DIR)

    # 1) 扫描子目录并解析 eps
    subdirs = [d for d in os.listdir(ROOT) if os.path.isdir(os.path.join(ROOT, d))]
    # 仅保留包含 eps 的目录
    eps_dirs = []
    for d in subdirs:
        eps_val = parse_eps_from_dirname(d)
        if eps_val is not None:
            eps_dirs.append((d, eps_val))

    if not eps_dirs:
        print("未找到包含 eps 的子目录。请确认目录命名是否符合约定。")
        return

    # 按 eps 排序（可选）
    eps_dirs.sort(key=lambda x: x[1])

    summary_rows = []

    # 2) 遍历每个目录，读取 metrics_logs 下的所有 CSV
    for dirname, eps_val in eps_dirs:
        full_log_dir = os.path.join(ROOT, dirname, LOG_SUBPATH)
        if not os.path.isdir(full_log_dir):
            print(f"[跳过] {dirname} 未找到日志目录：{LOG_SUBPATH}")
            continue

        csv_files = glob.glob(os.path.join(full_log_dir, "*.csv"))
        if not csv_files:
            print(f"[跳过] {dirname} 的 {LOG_SUBPATH} 下没有 CSV 文件")
            continue

        # 每个 CSV 单独画图
        for csv_path in csv_files:
            try:
                df = pd.read_csv(csv_path)
            except Exception as e:
                print(f"[警告] 读取 CSV 失败：{csv_path}，错误：{e}")
                continue

            # 寻找步数字段
            step_col = find_step_column(df)
            if step_col is None:
                print(f"[跳过] {csv_path} 未找到步数字段（{STEP_COL_CANDIDATES}）")
                continue

            if LOSS_COL not in df.columns:
                print(f"[跳过] {csv_path} 未找到 '{LOSS_COL}' 列")
                continue

            # 3) 画每个 CSV 的 loss 曲线（带滑动窗口平滑）
            # 计算中心滑动平均，窗口为 SMOOTH_WINDOW；若窗口太大，自动退化为可用长度
            win = int(SMOOTH_WINDOW) if 'SMOOTH_WINDOW' in globals() else 11
            win = max(1, win)
            if win > 1:
                # min_periods 取一半窗口，避免开头/结尾为空
                loss_smoothed = (
                    pd.Series(df[LOSS_COL].values)
                      .rolling(window=win, min_periods=max(1, win // 2), center=True)
                      .mean()
                      .to_numpy()
                )
            else:
                loss_smoothed = df[LOSS_COL].values

            fig, ax = plt.subplots(figsize=(7, 4))
            ax.plot(df[step_col].values, loss_smoothed, linewidth=1.6)
            ax.set_xlabel(step_col)
            ax.set_ylabel(LOSS_COL + (f" (MA w={win})" if win > 1 else ""))
            ax.set_title(f"h = {eps_val:g}")
            ax.grid(True, linestyle="--", alpha=0.3)
            # 输出图以 h 的值命名，避免覆盖同一 h 下多个 CSV，附上 CSV 名作为后缀
            base_name = os.path.splitext(os.path.basename(csv_path))[0]
            # 以 h 的值命名，避免覆盖同一 h 下多个 CSV，附上 CSV 名作为后缀
            out_png = os.path.join(OUTPUT_DIR, f"h={eps_val:.6g}__{base_name}.png")
            fig.tight_layout()
            fig.savefig(out_png, dpi=150)
            plt.close(fig)

            # 4) 记录该 CSV 的最小 eval_loss
            if EVAL_LOSS_COL in df.columns:
                try:
                    best_eval = float(np.nanmin(df[EVAL_LOSS_COL].values))
                except Exception:
                    best_eval = None
            else:
                best_eval = None
                print(f"[提示] {csv_path} 无 '{EVAL_LOSS_COL}' 列，跳过该文件在汇总中的 eval_loss。")

            # 记录该 CSV 的最小 train_loss（与 LOSS_COL 一致）
            try:
                best_train = float(np.nanmin(df[LOSS_COL].values))
            except Exception:
                best_train = None

            # 计算 eps 的指数（取 log10 并四舍五入为整数）
            try:
                eps_exp = int(round(math.log10(eps_val)))
            except Exception:
                eps_exp = None

            summary_rows.append({
                "dir": dirname,
                "csv": os.path.basename(csv_path),
                "eps": eps_val,
                "eps_exponent": eps_exp,
                "best_eval_loss": best_eval,
                "best_train_loss": best_train,
            })

    # 5) 写出 summary.csv
    if not summary_rows:
        print("没有可汇总的结果（可能没有 CSV 或缺少必要列）。")
        return

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(SUMMARY_CSV, index=False, encoding="utf-8")
    print(f"[信息] 已写出汇总：{SUMMARY_CSV}")

    # 6) 画 “eps 的指数 vs 最佳 eval_loss” 的图
    #    过滤掉没有 eval_loss 或没有 exponent 的行
    plot_df = summary_df.dropna(subset=["best_eval_loss", "eps_exponent"]).copy()
    if plot_df.empty:
        print("[提示] 没有有效的 eval_loss 数据用于画 ‘eps 指数 vs eval_loss’ 图。")
        return

    # 同一个 eps_exponent 可能有多个 CSV，取各 exponent 下的最优（最小）eval_loss
    agg = plot_df.groupby("eps", as_index=False)["best_eval_loss"].min().sort_values("eps")

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(agg["eps"].values, agg["best_eval_loss"].values, marker="o", linewidth=1.8)
    ax.set_xscale("log")
    ax.set_xlabel("h (10^x)")
    ax.set_ylabel("best eval_loss (min across CSVs)")
    ax.set_title("Best eval_loss vs exponent of h")
    ax.grid(True, linestyle="--", alpha=0.3)
    fig.tight_layout()
    out_png = os.path.join(OUTPUT_DIR, "best_eval_loss_vs_eps_exponent.png")
    fig.savefig(out_png, dpi=160)
    plt.close(fig)
    print(f"[信息] 已输出：{out_png}")

    # 7) 画 “h (eps) vs 最佳 train_loss” 的图（每个 CSV 先取最小 train_loss，再按 eps 聚合最优）
    plot_df2 = summary_df.dropna(subset=["best_train_loss", "eps"]).copy()
    if plot_df2.empty:
        print("[提示] 没有有效的 train_loss 数据用于画 ‘h vs train_loss’ 图。")
    else:
        agg2 = plot_df2.groupby("eps", as_index=False)["best_train_loss"].min().sort_values("eps")
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        ax2.plot(agg2["eps"].values, agg2["best_train_loss"].values, marker="o", linewidth=1.8)
        ax2.set_xscale("log")
        ax2.set_xlabel("h (10^x)")
        ax2.set_ylabel("best train_loss (min across CSVs)")
        ax2.set_title("Best train_loss vs h")
        ax2.grid(True, linestyle="--", alpha=0.3)
        fig2.tight_layout()
        out_png2 = os.path.join(OUTPUT_DIR, "best_train_loss_vs_eps.png")
        fig2.savefig(out_png2, dpi=160)
        plt.close(fig2)
        print(f"[信息] 已输出：{out_png2}")

    print("[完成] 单文件 loss 曲线与汇总关系图已生成到目录：", OUTPUT_DIR)


if __name__ == "__main__":
    main()