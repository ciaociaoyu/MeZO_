#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
读取目录下所有 JSON 文件，提取 accuracy，画折线图并保存为 accuracy_plot.png
默认按文件名进行“自然排序”（file2 < file10），也可选择按修改时间排序。
"""

import argparse
import json
import os
import re
import sys
from glob import glob
from typing import List, Tuple

import matplotlib.pyplot as plt


def natural_sort_key(s: str):
    """将字符串拆分为数字与非数字片段，以实现自然排序（file2 < file10）"""
    return [int(t) if t.isdigit() else t.lower() for t in re.findall(r'\d+|\D+', s)]


def extract_eps_value(filename: str) -> str:
    """从文件名中提取 eps 后面的值，直到下一个 '-'"""
    match = re.search(r'eps([^-]+)', filename)
    if match:
        return match.group(1)
    else:
        return filename  # 如果没找到，返回原文件名


def find_json_files(root: str, pattern: str, recursive: bool) -> List[str]:
    """查找 JSON 文件"""
    if recursive:
        # 递归搜索
        return sorted(
            glob(os.path.join(root, "**", pattern), recursive=True),
            key=natural_sort_key
        )
    else:
        # 仅当前目录
        return sorted(
            glob(os.path.join(root, pattern)),
            key=natural_sort_key
        )


def load_accuracy(fp: str):
    """从 JSON 文件中读取 accuracy 字段"""
    try:
        with open(fp, "r", encoding="utf-8") as f:
            data = json.load(f)
        acc = data.get("accuracy", None)
        if acc is None:
            raise KeyError("missing 'accuracy'")
        return float(acc)
    except Exception as e:
        # 返回 None 并在主流程里统一处理
        return None


def collect_data(files: List[str], sort_by: str) -> Tuple[List[str], List[float]]:
    """读取所有文件并按需求排序，返回 (labels, accuracies)"""
    records = []
    for f in files:
        acc = load_accuracy(f)
        if acc is not None:
            # 记录文件名、修改时间和 accuracy
            mtime = os.path.getmtime(f)
            records.append((f, mtime, acc))

    if not records:
        print("未找到有效的 JSON（包含 'accuracy' 字段）文件。", file=sys.stderr)
        sys.exit(1)

    if sort_by == "name":
        records.sort(key=lambda x: natural_sort_key(os.path.basename(x[0])))
    elif sort_by == "mtime":
        records.sort(key=lambda x: x[1])
    else:
        # 默认 name
        records.sort(key=lambda x: natural_sort_key(os.path.basename(x[0])))

    labels = [extract_eps_value(os.path.basename(r[0])) for r in records]
    accuracies = [r[2] for r in records]
    return labels, accuracies


def plot_line(labels: List[str], values: List[float], rotate_xticks: bool, out: str):
    """绘制折线图并保存"""
    plt.figure(figsize=(10, 5))
    plt.plot(range(len(values)), values, marker="o")
    plt.title("Accuracy over JSON files")
    plt.xlabel("File (sorted)")
    plt.ylabel("Accuracy")

    # X 轴使用文件名作为刻度
    plt.xticks(ticks=range(len(labels)), labels=labels, rotation=45 if rotate_xticks else 0, ha="right")
    plt.tight_layout()
    plt.savefig(out, dpi=200)
    print(f"图已保存：{out}")


def main():
    parser = argparse.ArgumentParser(description="读取 JSON 的 accuracy 字段并画折线图")
    parser.add_argument("--dir", default=".", help="搜索的根目录（默认当前目录）")
    parser.add_argument("--pattern", default="*.json", help="文件匹配模式（默认 *.json）")
    parser.add_argument("--no-recursive", action="store_true", help="不递归子目录")
    parser.add_argument("--sort-by", choices=["name", "mtime"], default="name", help="排序方式：按文件名或修改时间")
    parser.add_argument("--rotate-xticks", action="store_true", help="X 轴标签旋转 45°，防止重叠")
    parser.add_argument("--out", default="accuracy_plot.png", help="输出图片文件名")
    args = parser.parse_args()

    files = find_json_files(args.dir, args.pattern, recursive=not args.no_recursive)
    if not files:
        print("没有找到匹配的 JSON 文件。", file=sys.stderr)
        sys.exit(1)

    labels, values = collect_data(files, args.sort_by)
    # 打印一个简要统计
    print(f"有效文件数：{len(values)}")
    print(f"accuracy 最小/最大/平均：{min(values):.6f} / {max(values):.6f} / {sum(values)/len(values):.6f}")

    plot_line(labels, values, rotate_xticks=args.rotate_xticks, out=args.out)


if __name__ == "__main__":
    main()