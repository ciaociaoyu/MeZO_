import logging
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from tools.generate_k_shot_data import (
    TASK_NAME_CANONICAL,
    canonicalize_task_name,
    is_materialized_split_complete,
    is_original_dataset_available,
    materialize_task_data,
)

DEFAULT_DATA_ROOT = "data/k-shot-1k-test"
DEFAULT_ORIGINAL_DATA_ROOT = "data/original"
DATASET_MODE_CHOICES = {"auto", "fewshot", "full"}

TASK_DIR_NAME_PREFERENCES = {
    "cola": ["CoLA", "cola"],
    "mnli": ["MNLI", "mnli"],
    "mnli-mm": ["MNLI", "mnli"],
    "mrpc": ["MRPC", "mrpc"],
    "sst-2": ["SST-2", "sst-2"],
    "sts-b": ["STS-B", "sts-b"],
    "qqp": ["QQP", "qqp"],
    "qnli": ["QNLI", "qnli"],
    "rte": ["RTE", "rte"],
    "wnli": ["WNLI", "wnli"],
    "snli": ["SNLI", "snli"],
    "mr": ["mr", "MR"],
    "sst-5": ["sst-5", "SST-5"],
    "subj": ["subj", "SUBJ"],
    "trec": ["trec", "TREC"],
    "cr": ["cr", "CR"],
    "mpqa": ["mpqa", "MPQA"],
}


@dataclass
class DataResolutionResult:
    requested_dataset_mode: str
    resolved_dataset_mode: str
    resolved_data_dir: str
    data_root: str
    original_data_root: str
    task_dir_name: str
    data_seed: int
    downloaded_original_data: bool
    generated_data_split: bool
    used_legacy_full_alias: bool


def _as_abs_path(path_value: str, project_root: Path) -> Path:
    path = Path(path_value).expanduser()
    if not path.is_absolute():
        path = project_root / path
    return path.resolve()


def _is_full_dir_name(name: str) -> bool:
    if not name.startswith("full-"):
        return False
    seed = name.split("-", 1)[1]
    return seed.isdigit()


def _is_legacy_full_dir_name(name: str) -> bool:
    return re.fullmatch(r"-16-\d+", name) is not None


def _is_k_seed_dir_name(name: str) -> bool:
    return re.fullmatch(r"-?\d+-\d+", name) is not None


def _looks_like_setting_dir(name: str) -> bool:
    return _is_k_seed_dir_name(name) or _is_full_dir_name(name) or _is_legacy_full_dir_name(name)


def _infer_data_root(data_dir: Path, explicit_data_root: Optional[str], project_root: Path) -> Path:
    if explicit_data_root:
        return _as_abs_path(explicit_data_root, project_root)

    if _looks_like_setting_dir(data_dir.name) and data_dir.parent.name:
        return data_dir.parent.parent

    return _as_abs_path(DEFAULT_DATA_ROOT, project_root)


def _task_aliases(task_name: str) -> List[str]:
    task_name = task_name.lower()
    aliases = TASK_DIR_NAME_PREFERENCES.get(task_name)
    if aliases:
        return aliases

    canonical = TASK_NAME_CANONICAL.get(task_name, canonicalize_task_name(task_name))
    return [canonical]


def _resolve_task_dir_name(task_name: str, data_root: Path, explicit_data_dir: Path) -> str:
    if _looks_like_setting_dir(explicit_data_dir.name):
        explicit_task = explicit_data_dir.parent.name
        if explicit_task:
            return explicit_task

    for alias in _task_aliases(task_name):
        if (data_root / alias).exists():
            return alias

    return _task_aliases(task_name)[0]


def _determine_dataset_mode(requested_mode: str, data_dir_name: str, num_k: int) -> str:
    requested_mode = (requested_mode or "auto").lower()
    if requested_mode not in DATASET_MODE_CHOICES:
        raise ValueError(
            f"Unsupported dataset_mode={requested_mode}. "
            f"Expected one of {sorted(DATASET_MODE_CHOICES)}"
        )

    if requested_mode != "auto":
        return requested_mode

    if _is_full_dir_name(data_dir_name) or _is_legacy_full_dir_name(data_dir_name):
        return "full"
    if num_k < 0:
        return "full"
    return "fewshot"


def _is_complete(path: Path, task_dir_name: str) -> bool:
    return path.is_dir() and is_materialized_split_complete(str(path), task_dir_name)


def _ensure_original_data(
    task_dir_name: str,
    original_data_root: Path,
    project_root: Path,
    logger: logging.Logger,
) -> bool:
    canonical_task = canonicalize_task_name(task_dir_name)
    if is_original_dataset_available(str(original_data_root), task_dir_name):
        return False
    if canonical_task != task_dir_name and is_original_dataset_available(str(original_data_root), canonical_task):
        return False

    data_dir = project_root / "data"
    download_script = data_dir / "download_dataset.sh"
    if not download_script.exists():
        raise FileNotFoundError(f"Cannot find download script: {download_script}")

    logger.info(
        "[data] Missing original dataset for task=%s under %s. Auto-downloading...",
        task_dir_name,
        original_data_root,
    )
    subprocess.run(["bash", str(download_script)], cwd=str(data_dir), check=True)

    available_after_download = is_original_dataset_available(str(original_data_root), task_dir_name) or (
        canonical_task != task_dir_name and is_original_dataset_available(str(original_data_root), canonical_task)
    )
    if not available_after_download:
        raise FileNotFoundError(
            "Original dataset is still incomplete after auto-download. "
            f"task={task_dir_name}, canonical_task={canonical_task}, original_root={original_data_root}"
        )

    return True


def resolve_and_prepare_data(
    data_args,
    training_args,
    logger: Optional[logging.Logger] = None,
) -> DataResolutionResult:
    logger = logger or logging.getLogger(__name__)
    project_root = Path(__file__).resolve().parent.parent

    explicit_data_dir = _as_abs_path(getattr(data_args, "data_dir", DEFAULT_DATA_ROOT), project_root)
    data_root = _infer_data_root(explicit_data_dir, getattr(data_args, "data_root", None), project_root)
    original_data_root = _as_abs_path(DEFAULT_ORIGINAL_DATA_ROOT, project_root)

    num_k = int(getattr(data_args, "num_k", 16))
    train_seed = int(getattr(training_args, "seed"))
    data_seed = getattr(training_args, "data_seed", None)
    if data_seed is None:
        data_seed = train_seed
    data_seed = int(data_seed)

    requested_dataset_mode = (getattr(data_args, "dataset_mode", "auto") or "auto").lower()
    resolved_dataset_mode = _determine_dataset_mode(
        requested_mode=requested_dataset_mode,
        data_dir_name=explicit_data_dir.name,
        num_k=num_k,
    )
    task_dir_name = _resolve_task_dir_name(data_args.task_name, data_root, explicit_data_dir)

    task_root = data_root / task_dir_name
    preferred_full_dir = task_root / f"full-{data_seed}"
    legacy_full_dir = task_root / f"-16-{data_seed}"

    preferred_fewshot_dir = task_root / f"{num_k}-{data_seed}"
    train_seed_fewshot_dir = task_root / f"{num_k}-{train_seed}"

    downloaded_original_data = False
    generated_data_split = False
    used_legacy_full_alias = False
    resolved_data_dir: Optional[Path] = None

    if resolved_dataset_mode == "fewshot":
        if num_k <= 0:
            raise ValueError(
                f"fewshot mode requires num_k > 0, but got num_k={num_k}. "
                "Use --dataset_mode full for full-dataset training."
            )

        fewshot_candidates: List[Path] = []
        if not (_is_full_dir_name(explicit_data_dir.name) or _is_legacy_full_dir_name(explicit_data_dir.name)):
            fewshot_candidates.append(explicit_data_dir)
        for candidate in [preferred_fewshot_dir, train_seed_fewshot_dir]:
            if candidate not in fewshot_candidates:
                fewshot_candidates.append(candidate)

        for candidate in fewshot_candidates:
            if _is_complete(candidate, task_dir_name):
                resolved_data_dir = candidate
                break

        if resolved_data_dir is None:
            downloaded_original_data = _ensure_original_data(
                task_dir_name=task_dir_name,
                original_data_root=original_data_root,
                project_root=project_root,
                logger=logger,
            )
            fewshot_mode = data_root.name if data_root.name else "k-shot-1k-test"
            materialize_task_data(
                task=task_dir_name,
                data_dir=str(original_data_root),
                output_dir=str(data_root),
                dataset_mode="fewshot",
                seed=data_seed,
                k=num_k,
                fewshot_mode=fewshot_mode,
                output_task_name=task_dir_name,
                output_setting_name=f"{num_k}-{data_seed}",
            )
            generated_data_split = True
            resolved_data_dir = preferred_fewshot_dir

    else:
        full_candidates = [preferred_full_dir, legacy_full_dir]
        if (
            _is_full_dir_name(explicit_data_dir.name) or _is_legacy_full_dir_name(explicit_data_dir.name)
        ) and explicit_data_dir not in full_candidates:
            full_candidates.append(explicit_data_dir)

        for candidate in full_candidates:
            if _is_complete(candidate, task_dir_name):
                resolved_data_dir = candidate
                used_legacy_full_alias = _is_legacy_full_dir_name(candidate.name)
                break

        if resolved_data_dir is None:
            downloaded_original_data = _ensure_original_data(
                task_dir_name=task_dir_name,
                original_data_root=original_data_root,
                project_root=project_root,
                logger=logger,
            )
            materialize_task_data(
                task=task_dir_name,
                data_dir=str(original_data_root),
                output_dir=str(data_root),
                dataset_mode="full",
                seed=data_seed,
                full_dev_ratio=float(getattr(data_args, "full_dev_ratio", 0.1)),
                output_task_name=task_dir_name,
                output_setting_name=f"full-{data_seed}",
            )
            generated_data_split = True
            resolved_data_dir = preferred_full_dir

    if resolved_data_dir is None:
        raise RuntimeError("Failed to resolve data directory.")

    logger.info(
        "[data] dataset_mode requested=%s resolved=%s",
        requested_dataset_mode,
        resolved_dataset_mode,
    )
    logger.info("[data] task_dir=%s data_seed=%s", task_dir_name, data_seed)
    if downloaded_original_data:
        logger.info("[data] auto_download=1 original_data_root=%s", original_data_root)
    else:
        logger.info("[data] auto_download=0 original_data_root=%s", original_data_root)
    if generated_data_split:
        logger.info("[data] auto_generate=1 target=%s", resolved_data_dir)
    else:
        logger.info("[data] auto_generate=0 target=%s", resolved_data_dir)
    if used_legacy_full_alias:
        logger.info("[data] legacy_full_alias=1 path=%s", resolved_data_dir)
    logger.info("[data] resolved_data_dir=%s", resolved_data_dir)

    return DataResolutionResult(
        requested_dataset_mode=requested_dataset_mode,
        resolved_dataset_mode=resolved_dataset_mode,
        resolved_data_dir=str(resolved_data_dir),
        data_root=str(data_root),
        original_data_root=str(original_data_root),
        task_dir_name=task_dir_name,
        data_seed=data_seed,
        downloaded_original_data=downloaded_original_data,
        generated_data_split=generated_data_split,
        used_legacy_full_alias=used_legacy_full_alias,
    )
