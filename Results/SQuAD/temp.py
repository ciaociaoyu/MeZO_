

# -*- coding: utf-8 -*-
"""
直接运行本脚本即可：
- 自动遍历当前目录与 ./result/ 目录内的所有 *.csv 结果文件（例如 metrics_*.csv）
- 解析列：time, step, epoch, phase, metric, value, task, model, eps
- 过滤出 phase == "train" 且 metric == "loss" 的行，按 epoch 作为横坐标、value 作为纵坐标绘制折线图
- 标题为：任务名 + "+" + eps（科学计数法，如 SQuAD+1e-1）
- 结果图片保存到 ./result/plots/ 目录，文件名：{task}+{eps}.png

注意：
1) 本脚本不需要外部参数；
2) 会尽量从 CSV 的列中读取 task/eps；若缺失，将尝试从文件名中推断；
3) 若某些行的 epoch 缺失或为负数，会被忽略；
"""

import os
import csv
import glob
import math
import re
from typing import List, Tuple, Optional

import matplotlib.pyplot as plt


# ---- 工具函数 ---------------------------------------------------------------

def _ensure_dir(path: str) -> None:
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def _sci_eps(eps_val: str) -> str:
    """将字符串形式的 eps 规范为形如 1e-1 的样式。
    接受 '0.1' / '1e-01' / '1E-1' / '1E-01' 等，返回如 '1e-1'。
    如果无法解析，原样返回。
    """
    s = str(eps_val).strip()
    # 先尝试按浮点数解析
    try:
        v = float(s)
    except Exception:
        # 如果像 'eps0.1' 这种，从中提取数值部分
        m = re.search(r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)", s)
        if not m:
            return s
        try:
            v = float(m.group(1))
        except Exception:
            return s
    # 格式化为 1e-1 风格（去掉指数前导 0）
    sci = f"{v:.0e}"  # 1e-01
    # 统一小写 e
    sci = sci.replace("E", "e")
    # 去掉指数里的前导 0
    sci = re.sub(r"e([+-])0+(\d+)$", r"e\1\2", sci)
    # 去掉多余的 + 号（通常 eps 为正）
    sci = sci.replace("e+", "e")
    return sci


def _try_parse_task_eps_from_row(row: dict) -> Tuple[Optional[str], Optional[str]]:
    task = row.get("task") or row.get("Task") or None
    eps = row.get("eps") or row.get("EPS") or None
    if eps is not None:
        eps = _sci_eps(eps)
    return task, eps


def _try_parse_task_eps_from_filename(fname: str) -> Tuple[Optional[str], Optional[str]]:
    """尽力从文件名推断 task/eps，例如 metrics_SQuAD-opt-1.3b-eps0.1-....csv"""
    base = os.path.basename(fname)
    # 1) 任务名：假设在 metrics_ 之后到第一个 '-' 或 '+' 之前的段含有任务名（宽松匹配常见任务）
    task = None
    m = re.search(r"metrics_([^_+\-]+)", base)
    if m:
        task = m.group(1)
    # 2) eps：匹配 eps 后跟数字
    eps = None
    m2 = re.search(r"eps([\d\.eE+-]+)", base)
    if m2:
        eps = _sci_eps(m2.group(1))
    return task, eps


def _collect_csv_files() -> List[str]:
    here = os.path.abspath(os.path.dirname(__file__))
    candidates = []
    # 当前目录的 csv
    candidates += glob.glob(os.path.join(here, "*.csv"))
    # ./result 目录的 csv（优先）
    result_dir = os.path.join(here, "result")
    candidates += glob.glob(os.path.join(result_dir, "*.csv"))
    # 常见命名：metrics_*.csv
    candidates = [f for f in candidates if os.path.isfile(f)]
    # 去重
    seen = set()
    uniq = []
    for f in candidates:
        if f not in seen:
            seen.add(f)
            uniq.append(f)
    return uniq


# ---- 主逻辑：读取并绘图 -----------------------------------------------------

def load_train_loss_series(csv_path: str) -> Tuple[List[float], List[float], Optional[str], Optional[str]]:
    """
    读取一个 metrics CSV，返回 (epochs, losses, task, eps)
    仅保留 phase==train 且 metric==loss 的行；epoch 为浮点、>=0。
    """
    epochs: List[float] = []
    losses: List[float] = []
    task: Optional[str] = None
    eps: Optional[str] = None

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # 尝试从行中拿 task/eps（只要拿到一次即可）
            if task is None or eps is None:
                t, e = _try_parse_task_eps_from_row(row)
                task = task or t
                eps = eps or e

            phase = (row.get("phase") or "").strip().lower()
            metric = (row.get("metric") or "").strip().lower()
            if phase != "train" or metric != "loss":
                continue
            # 解析 epoch 与 loss
            try:
                ep = float(row.get("epoch", "nan"))
                val = float(row.get("value", "nan"))
            except Exception:
                continue
            if math.isnan(ep) or math.isnan(val) or ep < 0:
                continue
            epochs.append(ep)
            losses.append(val)

    # 如果 task/eps 还没有，尝试从文件名推断
    if task is None or eps is None:
        t2, e2 = _try_parse_task_eps_from_filename(csv_path)
        task = task or t2
        eps = eps or e2

    return epochs, losses, task, eps


def plot_series(epochs: List[float], losses: List[float], task: Optional[str], eps: Optional[str], out_dir: str) -> Optional[str]:
    if not epochs or not losses:
        return None
    title_task = task or "Task"
    title_eps = _sci_eps(eps) if eps is not None else "eps"
    title = f"{title_task}+{title_eps}"

    plt.figure()
    plt.plot(epochs, losses)
    plt.xlabel("epoch")
    plt.ylabel("trainloss")
    plt.title(title)

    safe_task = re.sub(r"[^\w\-\.]+", "_", title_task)
    safe_eps = re.sub(r"[^\w\-\.]+", "_", title_eps)
    out_name = f"{safe_task}+{safe_eps}.png"
    _ensure_dir(out_dir)
    out_path = os.path.join(out_dir, out_name)
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    return out_path


def main() -> None:
    csv_files = _collect_csv_files()
    if not csv_files:
        print("未找到任何 CSV 文件。请确认当前目录或 ./result/ 下是否有 metrics_*.csv 文件。")
        return

    here = os.path.abspath(os.path.dirname(__file__))
    plots_dir = os.path.join(here, "result", "plots")
    saved = []

    for fp in csv_files:
        try:
            epochs, losses, task, eps = load_train_loss_series(fp)
            outp = plot_series(epochs, losses, task, eps, plots_dir)
            if outp:
                print(f"已生成图像: {outp}")
                saved.append(outp)
            else:
                print(f"跳过（无有效 train loss 数据）: {fp}")
        except Exception as e:
            print(f"处理文件失败: {fp}\n  错误: {e}")

    if saved:
        print("\n全部完成。输出目录：")
        print(os.path.dirname(saved[0]))


if __name__ == "__main__":
    main()