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
        if task == 'SST-2':
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
    parser.add_argument("--k", type=int, default=16,
        help="Training examples for each class.")
    parser.add_argument("--task", type=str, nargs="+",
        default=['SST-2'],
        help="Task names")
    parser.add_argument("--seed", type=int, nargs="+",
        default=[16],
        help="Random seeds")

    parser.add_argument("--data_dir", type=str, default="data/original", help="Path to original data")
    parser.add_argument("--output_dir", type=str, default="data", help="Output path")
    parser.add_argument("--mode", type=str, default='k-shot', choices=['k-shot', 'k-shot-10x', 'k-shot-1k-test'], help="k-shot or k-shot-10x (10x dev set)")

    args = parser.parse_args()
    args.output_dir = os.path.join(args.output_dir, args.mode)

    k = args.k
    print("K =", k)
    datasets = load_datasets(args.data_dir, args.task)

    for seed in args.seed:
        print("Seed = %d" % (seed))
        for task, dataset in datasets.items():
            # Set random seed
            np.random.seed(seed)

            # Shuffle the training set
            print("| Task = %s" % (task))
            train_header, train_lines = split_header(task, dataset["train"])
            dev_header, dev_lines = split_header(task, dataset["dev"])
            
            np.random.shuffle(train_lines)

            # Set up dir
            task_dir = os.path.join(args.output_dir, task)
            setting_dir = os.path.join(task_dir, f"{k}-{seed}")
            os.makedirs(setting_dir, exist_ok=True)


            # 写入完整的训练集
            with open(os.path.join(setting_dir, "train.tsv"), "w") as f:
                for line in train_header:
                    f.write(line)
                for line in train_lines:
                    f.write(line)
            # 将原始 dev 数据集划分为两部分：前一半用于验证，后一半用于测试
            split_index = len(dev_lines) // 4
            dev_lines_eval = dev_lines[:split_index]
            dev_lines_test = dev_lines[split_index:]

            # 写入验证集（dev.tsv）使用划分后的前一半数据
            with open(os.path.join(setting_dir, "dev.tsv"), "w") as f:
                for line in dev_header:
                    f.write(line)
                for line in dev_lines_eval:
                    f.write(line)

            # 写入测试集（test.tsv）使用划分后的后一半数据
            with open(os.path.join(setting_dir, "test.tsv"), "w") as f:
                for line in dev_header:
                    f.write(line)
                for line in dev_lines_test:
                    f.write(line)


if __name__ == "__main__":
    main()
