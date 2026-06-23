"""Task adapter helpers for OPT MeZO option-loss experiments."""

from __future__ import annotations

import sys
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
LARGE_MODELS_DIR = REPO_ROOT / "large_models"
if str(LARGE_MODELS_DIR) not in sys.path:
    sys.path.insert(0, str(LARGE_MODELS_DIR))

from tasks import Sample, get_task as get_large_task  # noqa: E402
from utils import temp_seed  # noqa: E402


TASK_TO_LARGE = {
    "sst-2": "SST2",
    "sst2": "SST2",
    "sst-5": "SST5",
    "sst5": "SST5",
    "rte": "RTE",
    "mnli": "MNLI",
    "trec": "TREC",
}


class TRECTemplate:
    DISPLAY_WORDS = {
        "description": "description",
        "entity": "entity",
        "abbreviation": "abbreviation",
        "human": "human",
        "number": "number",
        "location": "location",
    }
    DISPLAY_PERSON = {**DISPLAY_WORDS, "human": "person"}
    DISPLAY_CODES = {
        "description": "DESC",
        "entity": "ENTY",
        "abbreviation": "ABBR",
        "human": "HUM",
        "number": "NUM",
        "location": "LOC",
    }

    def __init__(self) -> None:
        style = str(os.environ.get("TREC_VERBALIZER_STYLE", "words")).strip().lower()
        if style == "coarse_codes":
            self.display = self.DISPLAY_CODES
        elif style == "person":
            self.display = self.DISPLAY_PERSON
        else:
            self.display = self.DISPLAY_WORDS

    def encode(self, sample: Sample) -> str:
        text = str(sample.data["text"]).strip()
        return f"Question: {text}\nAnswer type:"

    def verbalize(self, sample: Sample, candidate: str) -> str:
        return f"{self.encode(sample)} {self.display.get(str(candidate), str(candidate))}"

    def encode_sfc(self, sample: Sample) -> str:
        return "Answer type:"

    def verbalize_sfc(self, sample: Sample, candidate: str) -> str:
        return f"Answer type: {self.display.get(str(candidate), str(candidate))}"


class TRECOptionTask:
    train_sep = "\n\n"
    generation = False
    mixed_set = False

    def __init__(self) -> None:
        # SetFit/TREC-QC coarse label order: DESC, ENTY, ABBR, HUM, NUM, LOC.
        self.candidates = ["description", "entity", "abbreviation", "human", "number", "location"]
        self.samples = self._load_samples()

    def get_template(self, template_version: int = 0) -> TRECTemplate:
        return TRECTemplate()

    def get_eval_splits(self) -> Dict[str, List[Sample]]:
        return {"valid": self.samples["valid"]}

    def _load_samples(self) -> Dict[str, List[Sample]]:
        from datasets import load_dataset

        try:
            ds_all = load_dataset("SetFit/TREC-QC")
        except Exception:
            try:
                ds_all = load_dataset("CogComp/trec")
            except Exception:
                ds_all = load_dataset("trec")
        train_split = "train" if "train" in ds_all else next(iter(ds_all.keys()))
        valid_split = "validation" if "validation" in ds_all else ("test" if "test" in ds_all else train_split)

        def row_label(ex: Dict[str, Any]) -> int:
            for key in ("label_coarse", "coarse_label", "label-coarse"):
                if key in ex:
                    return int(ex[key])
            if "label_coarse_text" in ex:
                text = str(ex["label_coarse_text"]).lower()
                for idx, cand in enumerate(self.candidates):
                    if cand in text:
                        return idx
            if "label_coarse_original" in ex:
                coarse_map = {"DESC": 0, "ENTY": 1, "ABBR": 2, "HUM": 3, "NUM": 4, "LOC": 5}
                text = str(ex["label_coarse_original"]).upper()
                if text in coarse_map:
                    return coarse_map[text]
            if "label" in ex and 0 <= int(ex["label"]) < len(self.candidates):
                return int(ex["label"])
            if "label_text" in ex:
                text = str(ex["label_text"]).lower()
                for idx, cand in enumerate(self.candidates):
                    if cand in text:
                        return idx
            raise KeyError(f"Cannot infer TREC label keys from {sorted(ex.keys())}")

        def row_text(ex: Dict[str, Any]) -> str:
            for key in ("text", "question", "sentence"):
                if key in ex:
                    return str(ex[key])
            raise KeyError(f"Cannot infer TREC text keys from {sorted(ex.keys())}")

        def convert(split_name: str) -> List[Sample]:
            rows = []
            for idx, ex in enumerate(ds_all[split_name]):
                label = row_label(ex)
                rows.append(
                    Sample(
                        id=idx,
                        data={"text": row_text(ex), "label": label},
                        correct_candidate=self.candidates[label],
                        candidates=list(self.candidates),
                    )
                )
            return rows

        return {"train": convert(train_split), "valid": convert(valid_split)}

    def sample_subset(self, data_split: str = "train", seed: int = 0, num: Optional[int] = 100, exclude: Optional[int] = None):
        samples = self.samples[data_split]
        max_take = len(samples) if exclude is None else len(samples) - 1
        take = max_take if num is None or int(num) <= 0 else min(int(num), max_take)
        with temp_seed(seed):
            order = np.random.permutation(len(samples)).tolist()
        if exclude is not None and 0 <= int(exclude) < len(samples) and int(exclude) in order:
            order.remove(int(exclude))
        return [samples[i] for i in order[:take]]

    def sample_fewshot_subset(self, data_split: str = "train", seed: int = 0, num_k: int = 16):
        buckets: Dict[str, List[Sample]] = {}
        for sample in self.samples[data_split]:
            buckets.setdefault(str(sample.correct_candidate), []).append(sample)
        with temp_seed(seed):
            selected: List[Sample] = []
            for key in sorted(buckets):
                bucket = buckets[key]
                order = np.random.permutation(len(bucket)).tolist()
                selected.extend(bucket[i] for i in order[: min(int(num_k), len(bucket))])
            order = np.random.permutation(len(selected)).tolist()
        return [selected[i] for i in order]

    def sample_train_sets(
        self,
        num_train: int = -1,
        num_dev: Optional[int] = None,
        num_eval: Optional[int] = None,
        num_train_sets: Optional[int] = 1,
        seed: Optional[int] = 0,
        dataset_mode: str = "full",
        num_k: int = 16,
    ):
        seeds = [0 if seed is None else int(seed)] if num_train_sets is None else list(range(int(num_train_sets)))
        if seed is not None:
            seeds = [int(seed)]
        out = []
        for set_seed in seeds:
            if str(dataset_mode).lower() == "fewshot":
                out.append(self.sample_fewshot_subset("train", seed=set_seed, num_k=int(num_k)))
            else:
                out.append(self.sample_subset("train", seed=set_seed, num=num_train))
        return out


def get_option_task(task_name: str):
    normalized = str(task_name).strip().lower()
    if normalized in {"trec", "trec-qc", "trec_qc"}:
        return TRECOptionTask()
    return get_large_task(TASK_TO_LARGE.get(normalized, task_name))
