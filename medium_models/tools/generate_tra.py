"""This script samples K examples randomly without replacement from the original data."""

import argparse
import os
import numpy as np
import pandas as pd
from pandas import DataFrame

def get_label(task, line):
    if task in ["MNLI", "MRPC", "QNLI", "QQP", "RTE", "SNLI", "SST-2", "STS-B", "WNLI", "CoLA"]:
        # GLUE style
        line = line.strip().split('\t')
        if task == 'CoLA':
            return line[1]
        elif task == 'MNLI':
            return line[-1]
        elif task == 'MRPC':
            return line[0]
        elif task == 'QNLI':
            return line[-1]
        elif task == 'QQP':
            return line[-1]
        elif task == 'RTE':
            return line[-1]
        elif task == 'SNLI':
            return line[-1]
        elif task == 'SST-2':
            return line[-1]
        elif task == 'STS-B':
            return 0 if float(line[-1]) < 2.5 else 1
        elif task == 'WNLI':
            return line[-1]
        else:
            raise NotImplementedError
    else:
        return line[0]

def load_datasets(data_dir, tasks):
    datasets = {}
    for task in tasks:
        if task in ["MNLI", "MRPC", "QNLI", "QQP", "RTE", "SNLI", "SST-2", "STS-B", "WNLI", "CoLA"]:
            # GLUE style (tsv)
            dataset = {}
            dirname = os.path.join(data_dir, task)
            if task == "MNLI":
                splits = ["train", "dev_matched", "dev_mismatched"]
            else:
                splits = ["train", "dev"]
            for split in splits:
                filename = os.path.join(dirname, f"{split}.tsv")
                with open(filename, "r") as f:
                    lines = f.readlines()
                dataset[split] = lines
            datasets[task] = dataset
        else:
            # Other datasets (csv)
            dataset = {}
            dirname = os.path.join(data_dir, task)
            splits = ["train", "test"]
            for split in splits:
                filename = os.path.join(dirname, f"{split}.csv")
                dataset[split] = pd.read_csv(filename, header=None)
            datasets[task] = dataset
    return datasets

def split_header(task, lines):
    """
    Returns if the task file has a header or not. Only for GLUE tasks.
    """
    if task in ["CoLA"]:
        return [], lines
    elif task in ["MNLI", "MRPC", "QNLI", "QQP", "RTE", "SNLI", "SST-2", "STS-B", "WNLI"]:
        return lines[0:1], lines[1:]
    else:
        raise ValueError("Unknown GLUE task.")

def main():
    parser = argparse.ArgumentParser()
    # 只使用 SST-2 数据集
    parser.add_argument("--task", type=str, nargs="+",
        default=['SST-2'],
        help="Task names")
    parser.add_argument("--data_dir", type=str, default="data/original", help="Path to original data")
    parser.add_argument("--output_dir", type=str, default="data", help="Output path")

    args = parser.parse_args()
    # 使用传统划分方式，将输出放到 data/traditional/ 下
    args.output_dir = os.path.join(args.output_dir, "traditional")

    # 加载数据集（这里只会加载 SST-2）
    datasets = load_datasets(args.data_dir, args.task)

    for task, dataset in datasets.items():
        print("| Task = %s" % task)
        # 对于 GLUE 格式任务，读取 train 和 dev 文件（包含 header 和所有数据行）
        train_header, train_lines = split_header(task, dataset["train"])
        dev_header, dev_lines = split_header(task, dataset["dev"])

        # 设置输出目录
        task_dir = os.path.join(args.output_dir, task)
        os.makedirs(task_dir, exist_ok=True)

        # 写入完整的训练集
        with open(os.path.join(task_dir, "train.tsv"), "w") as f:
            for line in train_header:
                f.write(line)
            for line in train_lines:
                f.write(line)
        # 将原始 dev 数据集划分为两部分：前一半用于验证，后一半用于测试
        split_index = len(dev_lines) // 4
        dev_lines_eval = dev_lines[:split_index]
        dev_lines_test = dev_lines[split_index:]

        # 写入验证集（dev.tsv）使用划分后的前一半数据
        with open(os.path.join(task_dir, "dev.tsv"), "w") as f:
            for line in dev_header:
                f.write(line)
            for line in dev_lines_eval:
                f.write(line)

        # 写入测试集（test.tsv）使用划分后的后一半数据
        with open(os.path.join(task_dir, "test.tsv"), "w") as f:
            for line in dev_header:
                f.write(line)
            for line in dev_lines_test:
                f.write(line)

if __name__ == "__main__":
    main()
