"""This script samples K examples randomly without replacement from the original data."""

import argparse
import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from pandas import DataFrame

GLUE_STYLE_TASKS = {
    "MNLI",
    "MRPC",
    "QNLI",
    "QQP",
    "RTE",
    "SNLI",
    "SST-2",
    "STS-B",
    "WNLI",
    "CoLA",
}

DEFAULT_TASKS = [
    "SST-2",
    "sst-5",
    "mr",
    "cr",
    "mpqa",
    "subj",
    "trec",
    "CoLA",
    "MRPC",
    "QQP",
    "STS-B",
    "MNLI",
    "SNLI",
    "QNLI",
    "RTE",
]

TASK_NAME_CANONICAL = {
    "cola": "CoLA",
    "mnli": "MNLI",
    "mnli-mm": "MNLI",
    "mrpc": "MRPC",
    "sst-2": "SST-2",
    "sts-b": "STS-B",
    "qqp": "QQP",
    "qnli": "QNLI",
    "rte": "RTE",
    "wnli": "WNLI",
    "snli": "SNLI",
    "mr": "mr",
    "sst-5": "sst-5",
    "subj": "subj",
    "trec": "trec",
    "cr": "cr",
    "mpqa": "mpqa",
}


def canonicalize_task_name(task: str) -> str:
    task = task.strip()
    if task in GLUE_STYLE_TASKS:
        return task
    return TASK_NAME_CANONICAL.get(task.lower(), task)


def is_glue_style_task(task: str) -> bool:
    return canonicalize_task_name(task) in GLUE_STYLE_TASKS


def get_label(task, line):
    task = canonicalize_task_name(task)
    if task in GLUE_STYLE_TASKS:
        # GLUE style
        line = line.strip().split('\t')
        if task == "CoLA":
            return line[1]
        elif task in {"MNLI", "QNLI", "QQP", "RTE", "SNLI", "SST-2", "WNLI"}:
            return line[-1]
        elif task == "MRPC":
            return line[0]
        elif task == "STS-B":
            return 0 if float(line[-1]) < 2.5 else 1
        else:
            raise NotImplementedError
    return line[0]


def required_original_files(task: str) -> List[str]:
    task = canonicalize_task_name(task)
    if task == "MNLI":
        return ["train.tsv", "dev_matched.tsv", "dev_mismatched.tsv"]
    if task in GLUE_STYLE_TASKS:
        return ["train.tsv", "dev.tsv"]
    return ["train.csv", "test.csv"]


def required_materialized_files(task: str) -> List[str]:
    task = canonicalize_task_name(task)
    if task == "MNLI":
        return [
            "train.tsv",
            "dev_matched.tsv",
            "test_matched.tsv",
            "dev_mismatched.tsv",
            "test_mismatched.tsv",
        ]
    if task in GLUE_STYLE_TASKS:
        return ["train.tsv", "dev.tsv", "test.tsv"]
    return ["train.csv", "dev.csv", "test.csv"]


def is_original_dataset_available(data_dir: str, task: str) -> bool:
    task = canonicalize_task_name(task)
    task_dir = os.path.join(data_dir, task)
    for filename in required_original_files(task):
        if not os.path.isfile(os.path.join(task_dir, filename)):
            return False
    return True


def is_materialized_split_complete(setting_dir: str, task: str) -> bool:
    task = canonicalize_task_name(task)
    for filename in required_materialized_files(task):
        if not os.path.isfile(os.path.join(setting_dir, filename)):
            return False
    return True


def load_datasets(data_dir, tasks):
    datasets = {}
    for task in tasks:
        task = canonicalize_task_name(task)
        if task in GLUE_STYLE_TASKS:
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
    task = canonicalize_task_name(task)
    if task in ["CoLA"]:
        return [], lines
    elif task in ["MNLI", "MRPC", "QNLI", "QQP", "RTE", "SNLI", "SST-2", "STS-B", "WNLI"]:
        return lines[0:1], lines[1:]
    else:
        raise ValueError("Unknown GLUE task.")


def _write_tsv(path: str, header: List[str], rows: List[str]) -> None:
    with open(path, "w") as f:
        for line in header:
            f.write(line)
        for line in rows:
            f.write(line)


def _stratified_split_rows(task: str, rows: List, seed: int, dev_ratio: float = 0.1) -> Tuple[List, List]:
    """
    Deterministic, stratified split on labels.
    - Per label, reserve round(dev_ratio * n_label) for dev.
    - Clamp per-label dev count to [1, n_label - 1] when n_label >= 2.
    - Keep labels with n_label <= 1 entirely in train.
    """
    rng = np.random.RandomState(seed)
    task = canonicalize_task_name(task)

    label_list: Dict[str, List] = {}
    for row in rows:
        label = get_label(task, row)
        label_list.setdefault(label, []).append(row)

    train_rows = []
    dev_rows = []
    for label in sorted(label_list.keys(), key=lambda x: str(x)):
        bucket = list(label_list[label])
        rng.shuffle(bucket)
        if len(bucket) <= 1:
            dev_count = 0
        else:
            dev_count = int(round(len(bucket) * dev_ratio))
            dev_count = max(1, min(dev_count, len(bucket) - 1))
        dev_rows.extend(bucket[:dev_count])
        train_rows.extend(bucket[dev_count:])

    rng.shuffle(train_rows)
    rng.shuffle(dev_rows)
    return train_rows, dev_rows


def materialize_fewshot_split(task: str, dataset: Dict, setting_dir: str, k: int, seed: int, mode: str = "k-shot-1k-test"):
    """
    Keep the historical few-shot behavior unchanged.
    """
    task = canonicalize_task_name(task)
    np.random.seed(seed)

    if task in GLUE_STYLE_TASKS:
        # GLUE style
        train_header, train_lines = split_header(task, dataset["train"])
        np.random.shuffle(train_lines)
    else:
        # Other datasets
        train_lines = dataset["train"].values.tolist()
        np.random.shuffle(train_lines)

    os.makedirs(setting_dir, exist_ok=True)

    # Write test splits
    if task in GLUE_STYLE_TASKS:
        # Use the original development set as the test set
        for split, lines in dataset.items():
            if split.startswith("train"):
                continue
            split = split.replace("dev", "test")

            test_header, test_lines = split_header(task, lines)
            if "1k-test" in mode and len(test_lines) > 1000:
                np.random.seed(42)
                np.random.shuffle(test_lines)
                test_lines = test_lines[:1000]
            _write_tsv(os.path.join(setting_dir, f"{split}.tsv"), test_header, test_lines)
    else:
        # Use the original test sets
        test_dataset = dataset["test"]
        if "1k-test" in mode and len(test_dataset.index) > 1000:
            test_dataset = test_dataset.sample(n=1000, random_state=42)
        test_dataset.to_csv(os.path.join(setting_dir, "test.csv"), header=False, index=False)

    # Get label list for balanced sampling
    label_list = {}
    for line in train_lines:
        label = get_label(task, line)
        if label not in label_list:
            label_list[label] = [line]
        else:
            label_list[label].append(line)

    if task in GLUE_STYLE_TASKS:
        _write_tsv(
            os.path.join(setting_dir, "train.tsv"),
            train_header,
            [line for label in label_list for line in label_list[label][:k]],
        )
        dev_name = "dev_matched.tsv" if task == "MNLI" else "dev.tsv"
        dev_rows = []
        for label in label_list:
            dev_rate = 11 if "10x" in mode else 2
            dev_rows.extend(label_list[label][k : k * dev_rate])
        _write_tsv(os.path.join(setting_dir, dev_name), train_header, dev_rows)
    else:
        new_train = []
        for label in label_list:
            new_train.extend(label_list[label][:k])
        DataFrame(new_train).to_csv(os.path.join(setting_dir, "train.csv"), header=False, index=False)

        new_dev = []
        for label in label_list:
            dev_rate = 11 if "10x" in mode else 2
            new_dev.extend(label_list[label][k : k * dev_rate])
        DataFrame(new_dev).to_csv(os.path.join(setting_dir, "dev.csv"), header=False, index=False)


def materialize_full_split(task: str, dataset: Dict, setting_dir: str, seed: int, dev_ratio: float = 0.1):
    """
    Full-dataset materialization:
    - train: full original train minus deterministic stratified validation holdout
    - dev: deterministic stratified holdout from original train
    - test: labeled original evaluation split (never hidden/unlabeled official test)
    """
    task = canonicalize_task_name(task)
    os.makedirs(setting_dir, exist_ok=True)

    if task in GLUE_STYLE_TASKS:
        train_header, train_lines = split_header(task, dataset["train"])
        new_train, new_dev = _stratified_split_rows(task, train_lines, seed=seed, dev_ratio=dev_ratio)
        _write_tsv(os.path.join(setting_dir, "train.tsv"), train_header, new_train)

        if task == "MNLI":
            # Keep MNLI matched/mismatched filenames compatible with existing processors.
            _write_tsv(os.path.join(setting_dir, "dev_matched.tsv"), train_header, new_dev)
            _write_tsv(os.path.join(setting_dir, "dev_mismatched.tsv"), train_header, new_dev)

            test_header_matched, test_lines_matched = split_header(task, dataset["dev_matched"])
            _write_tsv(os.path.join(setting_dir, "test_matched.tsv"), test_header_matched, test_lines_matched)

            test_header_mismatched, test_lines_mismatched = split_header(task, dataset["dev_mismatched"])
            _write_tsv(os.path.join(setting_dir, "test_mismatched.tsv"), test_header_mismatched, test_lines_mismatched)
        else:
            _write_tsv(os.path.join(setting_dir, "dev.tsv"), train_header, new_dev)
            test_header, test_lines = split_header(task, dataset["dev"])
            _write_tsv(os.path.join(setting_dir, "test.tsv"), test_header, test_lines)
    else:
        train_rows = dataset["train"].values.tolist()
        new_train, new_dev = _stratified_split_rows(task, train_rows, seed=seed, dev_ratio=dev_ratio)
        DataFrame(new_train).to_csv(os.path.join(setting_dir, "train.csv"), header=False, index=False)
        DataFrame(new_dev).to_csv(os.path.join(setting_dir, "dev.csv"), header=False, index=False)
        dataset["test"].to_csv(os.path.join(setting_dir, "test.csv"), header=False, index=False)


def materialize_task_data(
    task: str,
    data_dir: str,
    output_dir: str,
    dataset_mode: str,
    seed: int,
    k: int = 16,
    fewshot_mode: str = "k-shot-1k-test",
    full_dev_ratio: float = 0.1,
    output_task_name: str = None,
    output_setting_name: str = None,
) -> str:
    task = canonicalize_task_name(task)
    dataset_mode = dataset_mode.lower()
    if dataset_mode not in {"fewshot", "full"}:
        raise ValueError(f"Unsupported dataset_mode: {dataset_mode}")

    datasets = load_datasets(data_dir, [task])
    dataset = datasets[task]

    task_dir_name = output_task_name if output_task_name is not None else task
    task_dir = os.path.join(output_dir, task_dir_name)

    if output_setting_name is not None:
        setting_name = output_setting_name
    elif dataset_mode == "fewshot":
        setting_name = f"{k}-{seed}"
    else:
        setting_name = f"full-{seed}"

    setting_dir = os.path.join(task_dir, setting_name)
    if dataset_mode == "fewshot":
        materialize_fewshot_split(task, dataset, setting_dir, k=k, seed=seed, mode=fewshot_mode)
    else:
        materialize_full_split(task, dataset, setting_dir, seed=seed, dev_ratio=full_dev_ratio)

    return setting_dir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=16, help="Training examples for each class.")
    parser.add_argument("--task", type=str, nargs="+", default=DEFAULT_TASKS, help="Task names")
    parser.add_argument("--seed", type=int, nargs="+", default=[100, 13, 21, 42, 87], help="Random seeds")
    parser.add_argument("--data_dir", type=str, default="data/original", help="Path to original data")
    parser.add_argument("--output_dir", type=str, default="data", help="Output path")
    parser.add_argument(
        "--mode",
        type=str,
        default="k-shot",
        choices=["k-shot", "k-shot-10x", "k-shot-1k-test"],
        help="k-shot or k-shot-10x (10x dev set)",
    )
    parser.add_argument(
        "--dataset_mode",
        type=str,
        default="fewshot",
        choices=["fewshot", "full"],
        help="fewshot keeps historical behavior; full materializes full-dataset train/dev/test.",
    )
    parser.add_argument(
        "--full_dev_ratio",
        type=float,
        default=0.1,
        help="Validation ratio for full mode (deterministic stratified split from train).",
    )

    args = parser.parse_args()
    args.output_dir = os.path.join(args.output_dir, args.mode)

    print("K =", args.k)
    print("Dataset mode =", args.dataset_mode)
    datasets = load_datasets(args.data_dir, args.task)

    for seed in args.seed:
        print("Seed = %d" % (seed))
        for task, dataset in datasets.items():
            task = canonicalize_task_name(task)
            print("| Task = %s" % (task))
            task_dir = os.path.join(args.output_dir, task)
            if args.dataset_mode == "fewshot":
                setting_dir = os.path.join(task_dir, f"{args.k}-{seed}")
                materialize_fewshot_split(task, dataset, setting_dir, k=args.k, seed=seed, mode=args.mode)
            else:
                setting_dir = os.path.join(task_dir, f"full-{seed}")
                materialize_full_split(task, dataset, setting_dir, seed=seed, dev_ratio=args.full_dev_ratio)


if __name__ == "__main__":
    main()
