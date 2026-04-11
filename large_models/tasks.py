from templates import *
from utils import temp_seed
import json
import os
from datasets import load_dataset
from dataclasses import dataclass
from typing import Dict, List, Union
import string
import random
import datasets
import sys
import numpy as np
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


TASK_NAME_ALIASES = {
    "sst2": "SST2",
    "sst-2": "SST2",
    "sst5": "SST5",
    "sst-5": "SST5",
    "boolq": "BoolQ",
    "snli": "SNLI",
    "mnli": "MNLI",
    "rte": "RTE",
    "squad": "SQuAD",
    "drop": "DROP",
    "cb": "CB",
    "copa": "Copa",
    "multirc": "MultiRC",
    "record": "ReCoRD",
    "wic": "WIC",
    "wsc": "WSC",
}

TASK_CLASS_NAMES = {
    "SST2": "SST2Dataset",
    "SST-2": "SST2Dataset",
    "SST5": "SST5Dataset",
    "SST-5": "SST5Dataset",
    "BoolQ": "BoolQDataset",
    "SNLI": "SNLIDataset",
    "MNLI": "MNLIDataset",
    "RTE": "RTEDataset",
    "SQuAD": "SQuADDataset",
    "DROP": "DROPDataset",
    "CB": "CBDataset",
    "Copa": "CopaDataset",
    "MultiRC": "MultiRCDataset",
    "ReCoRD": "ReCoRDDataset",
    "WIC": "WICDataset",
    "WSC": "WSCDataset",
}


def canonicalize_task_name(task_name: str) -> str:
    parts = task_name.split("__", 1)
    primary = parts[0].strip()
    canonical_primary = TASK_NAME_ALIASES.get(primary.lower(), primary)
    if len(parts) == 2:
        return f"{canonical_primary}__{parts[1]}"
    return canonical_primary


def get_task(task_name):
    normalized_task_name = canonicalize_task_name(task_name)
    aa = normalized_task_name.split("__")
    if len(aa) == 2:
        task_group, subtask = aa
    else:
        task_group = aa[0]
        subtask = None
    class_name = TASK_CLASS_NAMES.get(task_group, f"{task_group}Dataset")
    class_ = getattr(sys.modules[__name__], class_name)
    instance = class_(subtask)
    return instance


@dataclass
class Sample:
    id: int = None
    data: dict = None
    correct_candidate: Union[str, List[str]] = None
    candidates: List[str] = None


class Dataset:
    mixed_set = False
    train_sep = "\n\n"
    generation = False # whether this is a generation task

    def __init__(self, subtask=None, **kwargs) -> None:
        self.subtask = subtask
    
    def get_task_name(self):
        return self.subtask
        
    def load_dataset():
        raise NotImplementedError
    
    def get_template(self, template_version=0):
       templates = {0: Template}
       return templates[template_version]
   
    def build_sample(self, example):
        return 

    @staticmethod
    def resolve_dataset_mode(dataset_mode="auto", num_train=None, num_k=16):
        dataset_mode = (dataset_mode or "auto").lower()
        if dataset_mode not in {"auto", "fewshot", "full"}:
            raise ValueError(f"Unsupported dataset_mode={dataset_mode}. Expected one of ['auto', 'fewshot', 'full']")
        if dataset_mode != "auto":
            return dataset_mode
        if isinstance(num_train, int):
            if num_train < 0:
                return "full"
            if num_train > 0:
                return "fewshot"
        return "fewshot" if num_k is not None and int(num_k) > 0 else "full"

    def _fewshot_bucket_key(self, sample):
        if self.generation or isinstance(sample.correct_candidate, list):
            return None
        return sample.correct_candidate

    def sample_fewshot_subset(self, data_split="train", seed=0, num_k=16):
        samples = self.samples[data_split]
        if num_k is None or (isinstance(num_k, int) and num_k <= 0):
            return list(samples)

        buckets: Dict[object, List[Sample]] = {}
        for sample in samples:
            bucket_key = self._fewshot_bucket_key(sample)
            if bucket_key is None:
                return self.sample_subset(data_split=data_split, seed=seed, num=num_k)
            buckets.setdefault(bucket_key, []).append(sample)

        if len(buckets) <= 1:
            return self.sample_subset(data_split=data_split, seed=seed, num=num_k)

        with temp_seed(seed):
            selected = []
            for key in sorted(buckets.keys(), key=lambda x: str(x)):
                bucket = list(buckets[key])
                order = np.random.permutation(len(bucket)).tolist()
                take = min(int(num_k), len(bucket))
                selected.extend(bucket[i] for i in order[:take])

            selected_order = np.random.permutation(len(selected)).tolist()
            return [selected[i] for i in selected_order]

    def sample_train_sets(
        self,
        num_train=32,
        num_dev=None,
        num_eval=None,
        num_train_sets=None,
        seed=None,
        dataset_mode="auto",
        num_k=16,
    ):
        resolved_dataset_mode = self.resolve_dataset_mode(dataset_mode=dataset_mode, num_train=num_train, num_k=num_k)
        if seed is not None:
            seeds = [seed]
        elif num_train_sets is not None:
            seeds = list(range(num_train_sets))
        else:
            # one train/demo set per evaluation sample
            # (len of valid to evaluate on). If num_eval is None or <=0, use all valid.
            total_valid = len(self.samples["valid"])
            eff_num_eval = total_valid if (num_eval is None or (isinstance(num_eval, int) and num_eval <= 0)) else min(num_eval, total_valid)
            with temp_seed(0):
                seeds = np.random.randint(0, 10000, eff_num_eval)

        # Normalize "use all" semantics for train/dev/eval sizes
        total_train = len(self.samples["train"])
        total_valid = len(self.samples["valid"])

        if (num_train is None) or (isinstance(num_train, int) and num_train <= 0):
            eff_num_train = total_train
        else:
            eff_num_train = min(num_train, total_train)

        if (num_eval is None) or (isinstance(num_eval, int) and num_eval <= 0):
            eff_num_eval = total_valid
        else:
            eff_num_eval = min(num_eval, total_valid)

        if (num_dev is not None) and (isinstance(num_dev, int) and num_dev > 0):
            eff_num_dev = min(num_dev, total_train)
        else:
            eff_num_dev = None

        effective_num_k = num_k
        if (effective_num_k is None or (isinstance(effective_num_k, int) and effective_num_k <= 0)) and (
            isinstance(num_train, int) and num_train > 0
        ):
            effective_num_k = num_train

        train_samples = []
        for i, set_seed in enumerate(seeds):
            if self.mixed_set:
                raise NotImplementedError
            else:
                if resolved_dataset_mode == "fewshot":
                    sampled = self.sample_fewshot_subset(data_split="train", seed=set_seed, num_k=effective_num_k)
                    train_samples.append(sampled)
                    logger.info(
                        "Sample few-shot train set %d/%d (num_k=%s, seed=%s)",
                        len(sampled),
                        len(self.samples["train"]),
                        effective_num_k,
                        set_seed,
                    )
                elif eff_num_dev is not None:
                    # dev set is included at the end of train set
                    num_take = min(eff_num_train + eff_num_dev, total_train)
                    train_samples.append(self.sample_subset(data_split="train", seed=set_seed, num=num_take))
                    if eff_num_train + eff_num_dev > total_train:
                        logger.warn("num_train + num_dev > available training examples")
                else:
                    num_take = min(eff_num_train, total_train)
                    train_samples.append(self.sample_subset(data_split="train", seed=set_seed, num=num_take))

                if eff_num_dev is not None:
                    logger.info(f"Sample train set {len(train_samples[-1])}/{len(self.samples['train'])}")
                    logger.info(f"... including dev set {eff_num_dev} samples")
        return train_samples

    def sample_subset(self, data_split="train", seed=0, num=100, exclude=None):
        with temp_seed(seed):
            samples = self.samples[data_split]
            lens = len(samples)
            max_take = lens if exclude is None else lens - 1
            if num is None or (isinstance(num, int) and num <= 0):
                take = max_take
            else:
                take = min(num, max_take)
            index = np.random.permutation(lens).tolist()
            if exclude is not None and 0 <= exclude < lens:
                # Remove the excluded index if present
                if exclude in index:
                    index.remove(exclude)
            index = index[:take]
            return [samples[i] for i in index]
    
    @property
    def valid_samples(self):
        return self.samples["valid"]

    def get_eval_splits(self):
        return {"valid": self.samples["valid"]}


class SST2Dataset(Dataset):
    train_sep = "\n\n"
    def __init__(self, subtask=None, **kwargs) -> None:
        self.load_dataset(subtask, **kwargs)
        
    def load_dataset(self, path, **kwargs):
        d = load_dataset('glue', 'sst2')
        train_d = d["train"]
        validation_d = d["validation"]
        
        train_samples = [self.build_sample(example) for example in train_d]
        valid_samples = [self.build_sample(example) for example in validation_d]
        
        self.samples = {"train": train_samples, "valid": valid_samples}
    
    # for generative tasks, candidates are []
    def build_sample(self, example):
        label = int(example["label"])
        return Sample(id=example["idx"], data=example, correct_candidate=label, candidates=[0, 1])
        
    def get_template(self, template_version=0):
        return {0: SST2Template}[template_version]()


class SST5Dataset(Dataset):
    train_sep = "\n\n"

    def __init__(self, subtask=None, **kwargs) -> None:
        self.load_dataset(subtask, **kwargs)

    def load_dataset(self, path, **kwargs):
        d = load_dataset("SetFit/sst5")
        train_d = d["train"]
        valid_split = "validation" if "validation" in d else "test"
        valid_d = d[valid_split]

        train_samples = [self.build_sample(example, idx) for idx, example in enumerate(train_d)]
        valid_samples = [self.build_sample(example, idx) for idx, example in enumerate(valid_d)]
        self.samples = {"train": train_samples, "valid": valid_samples}

    def build_sample(self, example, idx):
        label = int(example["label"])
        return Sample(id=idx, data=example, correct_candidate=label, candidates=[0, 1, 2, 3, 4])

    def get_template(self, template_version=0):
        return {0: SST5Template}[template_version]()


class CopaDataset(Dataset):
    train_sep = "\n\n"
    mixed_set = False

    def __init__(self, subtask=None, **kwargs) -> None:
        self.load_dataset(subtask, **kwargs)
        
    def load_dataset(self, path, **kwargs):
        train_examples = load_dataset('super_glue', "copa")["train"]
        valid_examples = load_dataset('super_glue', "copa")["validation"]
    
        train_samples = [self.build_sample(example) for example in train_examples]
        valid_samples = [self.build_sample(example) for example in valid_examples]
        self.samples = {"train": train_samples, "valid": valid_samples}
    
    # for generative tasks, candidates are []
    def build_sample(self, example):
        sample = \
            Sample(
                id=example["idx"],
                data=example,
                candidates=[example["choice1"], example["choice2"]],
                correct_candidate=example[f"choice{example['label'] + 1}"],
            )
        
        return sample
        
    def get_template(self, template_version=0):
        return {0: CopaTemplate}[template_version]()


class BoolQDataset(Dataset):
    def __init__(self, subtask=None, **kwargs) -> None:
        self.load_dataset(subtask, **kwargs)
    
    def load_dataset(self, path, **kwargs):
        d = load_dataset("google/boolq")
        train_set = d["train"]
        valid_set = d["validation"]

        train_samples = [self.build_sample(example) for example in train_set]
        valid_samples = [self.build_sample(example) for example in valid_set]
        self.samples = {"train": train_samples, "valid": valid_samples}
    
    def build_sample(self, example):
        sample = \
            Sample(
                data=example,
                candidates=["Yes", "No"],
                correct_candidate="Yes" if example["answer"] else "No",
            )
        
        return sample
    
    def get_template(self, template_version=2):
        return {0: BoolQTemplate, 1: BoolQTemplateV2, 2: BoolQTemplateV3}[template_version]()


class MultiRCDataset(Dataset):
    
    def __init__(self, subtask=None, **kwargs) -> None:
        self.load_dataset(subtask, **kwargs)
    
    def load_dataset(self, path, **kwargs):
        d = load_dataset("super_glue", "multirc")
        train_set = d["train"]
        valid_set = d["validation"]

        train_samples = [self.build_sample(example) for example in train_set]
        valid_samples = [self.build_sample(example) for example in valid_set]
        self.samples = {"train": train_samples, "valid": valid_samples}
    
    def build_sample(self, example):
        sample = \
            Sample(
                data=example,
                candidates=[0, 1],
                correct_candidate=example['label']
            )
        
        return sample
    
    def get_template(self, template_version=0):
        return {0: MultiRCTemplate}[template_version]()


class CBDataset(Dataset):
    
    def __init__(self, subtask=None, **kwargs) -> None:
        self.load_dataset(subtask, **kwargs)
    
    def load_dataset(self, path, **kwargs):
        d = load_dataset("super_glue", "cb")
        train_set = d["train"]
        valid_set = d["validation"]

        train_samples = [self.build_sample(example) for example in train_set]
        valid_samples = [self.build_sample(example) for example in valid_set]
        self.samples = {"train": train_samples, "valid": valid_samples}
    
    def build_sample(self, example):
        sample = \
            Sample(
                data=example,
                candidates=[0, 1, 2],
                correct_candidate=example['label']
            )
        
        return sample
    
    def get_template(self, template_version=0):
        return {0: CBTemplate}[template_version]()


class WICDataset(Dataset):
    
    def __init__(self, subtask=None, **kwargs) -> None:
        self.load_dataset(subtask, **kwargs)
    
    def load_dataset(self, path, **kwargs):
        d = load_dataset("super_glue", "wic")
        train_set = d["train"]
        valid_set = d["validation"]

        train_samples = [self.build_sample(example) for example in train_set]
        valid_samples = [self.build_sample(example) for example in valid_set]
        self.samples = {"train": train_samples, "valid": valid_samples}
    
    def build_sample(self, example):
        sample = \
            Sample(
                data=example,
                candidates=[0, 1],
                correct_candidate=example['label']
            )
        
        return sample
    
    def get_template(self, template_version=0):
        return {0: WICTemplate}[template_version]()


class WSCDataset(Dataset):
    
    def __init__(self, subtask=None, **kwargs) -> None:
        self.load_dataset(subtask, **kwargs)
    
    def load_dataset(self, path, **kwargs):
        d = load_dataset("super_glue", "wsc.fixed")
        train_set = d["train"]
        valid_set = d["validation"]

        train_samples = [self.build_sample(example) for example in train_set]
        valid_samples = [self.build_sample(example) for example in valid_set]
        self.samples = {"train": train_samples, "valid": valid_samples}
    
    def build_sample(self, example):
        sample = \
            Sample(
                data=example,
                candidates=[0, 1],
                correct_candidate=example['label']
            )
        
        return sample
    
    def get_template(self, template_version=0):
        return {0: WSCTemplate}[template_version]()


class ReCoRDDataset(Dataset):
    
    def __init__(self, subtask=None, **kwargs) -> None:
        self.load_dataset(subtask, **kwargs)
    
    def load_dataset(self, path, **kwargs):
        d = load_dataset("super_glue", "record")
        train_set = d["train"]
        valid_set = d["validation"]

        train_samples = [self.build_sample(example) for example in train_set]
        valid_samples = [self.build_sample(example) for example in valid_set]
        self.samples = {"train": train_samples, "valid": valid_samples}
    
    def build_sample(self, example):
        sample = \
            Sample(
                data=example,
                candidates=example['entities'],
                correct_candidate=example['answers']
            )
        
        return sample
    
    def get_template(self, template_version=0):
        return {0: ReCoRDTemplateGPT3}[template_version]()


class RTEDataset(Dataset):
    
    def __init__(self, subtask=None, **kwargs) -> None:
        self.load_dataset(subtask, **kwargs)
    
    def load_dataset(self, path, **kwargs):
        d = load_dataset("super_glue", "rte")
        train_set = d["train"]
        valid_set = d["validation"]

        train_samples = [self.build_sample(example) for example in train_set]
        valid_samples = [self.build_sample(example) for example in valid_set]
        self.samples = {"train": train_samples, "valid": valid_samples}
    
    def build_sample(self, example):
        sample = \
            Sample(
                data=example,
                candidates=[0, 1],
                correct_candidate=example['label']
            )
        
        return sample
    
    def get_template(self, template_version=0):
        return {0: RTETemplate}[template_version]()


class SNLIDataset(Dataset):

    def __init__(self, subtask=None, **kwargs) -> None:
        self.load_dataset(subtask, **kwargs)

    def load_dataset(self, path, **kwargs):
        d = load_dataset("stanfordnlp/snli")
        label_names = d["train"].features["label"].names
        self.label_to_id = {name: idx for idx, name in enumerate(label_names)}

        train_samples = [
            self.build_sample(example, idx)
            for idx, example in enumerate(d["train"])
            if int(example["label"]) != -1
        ]
        valid_samples = [
            self.build_sample(example, idx)
            for idx, example in enumerate(d["validation"])
            if int(example["label"]) != -1
        ]
        self.samples = {"train": train_samples, "valid": valid_samples}

    def build_sample(self, example, idx):
        return Sample(
            id=idx,
            data=example,
            candidates=[0, 1, 2],
            correct_candidate=int(example["label"]),
        )

    def get_template(self, template_version=0):
        return {0: MNLITemplate}[template_version](label_to_id=self.label_to_id)


class MNLIDataset(Dataset):

    def __init__(self, subtask=None, **kwargs) -> None:
        self.load_dataset(subtask, **kwargs)

    def load_dataset(self, path, **kwargs):
        d = load_dataset("glue", "mnli")
        train_set = d["train"]
        valid_matched_set = d["validation_matched"]
        valid_mismatched_set = d["validation_mismatched"]

        label_names = train_set.features["label"].names
        self.label_to_id = {name: idx for idx, name in enumerate(label_names)}

        train_samples = [self.build_sample(example) for example in train_set]
        valid_matched_samples = [self.build_sample(example) for example in valid_matched_set]
        valid_mismatched_samples = [self.build_sample(example) for example in valid_mismatched_set]

        # Keep "valid" as matched split for backward compatibility with the current pipeline.
        self.samples = {
            "train": train_samples,
            "valid": valid_matched_samples,
            "valid_mismatched": valid_mismatched_samples,
        }

    def build_sample(self, example):
        return Sample(
            id=example["idx"],
            data=example,
            candidates=[0, 1, 2],
            correct_candidate=int(example["label"]),
        )

    def get_eval_splits(self):
        return {
            "valid": self.samples["valid"],
            "valid_mismatched": self.samples["valid_mismatched"],
        }

    def get_template(self, template_version=0):
        return {0: MNLITemplate}[template_version](label_to_id=self.label_to_id)


class SQuADDataset(Dataset):
    metric_name = "f1"
    generation = True

    def __init__(self, subtask=None, **kwargs) -> None:
        self.load_dataset()
        
    def load_dataset(self):
        dataset = load_dataset("squad")
        train_examples = dataset["train"]
        valid_examples = dataset["validation"]

        train_samples = [self.build_sample(example, idx) for idx, example in enumerate(train_examples)]
        valid_samples = [self.build_sample(example, idx) for idx, example in enumerate(valid_examples)]
        self.samples = {"train": train_samples, "valid": valid_samples}
    
    # for generative tasks, candidates are []
    def build_sample(self, example, idx):
        answers = example['answers']['text']
        assert len(answers) > 0
        return Sample(
            id=idx,
            data={
                "title": example['title'],
                "context": example['context'],
                "question": example['question'],
                "answers": answers
            },
            candidates=None,
            correct_candidate=answers
        )
        
    def get_template(self, template_version=0):
        return {0: SQuADv2Template}[template_version]()


class DROPDataset(Dataset):
    metric_name = "f1"
    generation = True

    def __init__(self, subtask=None, **kwargs) -> None:
        self.load_dataset()
        
    def load_dataset(self):
        dataset = load_dataset("drop")
        train_examples = dataset["train"]
        valid_examples = dataset["validation"]

        train_samples = [self.build_sample(example, idx) for idx, example in enumerate(train_examples)]
        valid_samples = [self.build_sample(example, idx) for idx, example in enumerate(valid_examples)]
        self.samples = {"train": train_samples, "valid": valid_samples}
    
    # for generative tasks, candidates are []
    def build_sample(self, example, idx):
        answers = example['answers_spans']['spans']
        assert len(answers) > 0
        return Sample(
            id=idx,
            data={
                "context": example['passage'],
                "question": example['question'],
                "answers": answers
            },
            candidates=None,
            correct_candidate=answers
        )
        
    def get_template(self, template_version=0):
        return {0: DROPTemplate}[template_version]()
