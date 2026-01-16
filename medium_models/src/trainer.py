########## The following part is copied from Transformers' trainer (3.4.0) and later ported to be compatible with v4.4.2 and to support initialization from linear head probing. ##########

# coding=utf-8
# 最后修改时间：2026-01-15
# 修改摘要：
# - Adaptive-h：rolling probe buffer、多batch(nb)与多方向(nd)估计，log-EMA 平滑更新
# - 训练 eps 更新：加入 h_trunc_alpha（alpha^{-1/6}）放大因子；estimate_nu3 的 h_start 使用 min(eps_train, eps_f^{1/5}) 折中初始化
# - estimate_nu3：Δ(h)=(18a) + proximity=(18b) 双测试选 h；无可接受 h 则返回 NaN 丢弃方向
# - 数值稳健性：Δ3=0 时在满足(18a)(18b)前提下重试放大 h；仍失败则 NaN
# - initialize_c：修复 scale_norm_by_num_params 归一化 bug；retrieve_c 增加启发式 fallback
# Copyright 2020-present the HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
The Trainer class, to easily train a 🤗 Transformers from scratch or finetune it on a new task.
"""

import json
import random
import collections
import inspect
import math
import os
import re
import shutil
import warnings
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import csv
from packaging import version
from torch import nn
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import RandomSampler, SequentialSampler
from torch.optim.lr_scheduler import LambdaLR
import math
import time

import transformers
from transformers.file_utils import is_datasets_available, is_in_notebook
# 兼容性处理：新版本 Transformers 可能已移除 utils.is_torch_tpu_available
try:
    from transformers.utils import is_torch_tpu_available  # 旧版本存在
except Exception:
    def is_torch_tpu_available() -> bool:  # 回退：默认不使用 TPU
        return False
    # 若顶层 transformers 模块也缺少该符号，则注入一个同名函数，
    # 以兼容代码中 later 的 `transformers.is_torch_tpu_available()` 调用
    if not hasattr(transformers, "is_torch_tpu_available"):
        transformers.is_torch_tpu_available = is_torch_tpu_available  # type: ignore
from transformers.integrations import (
    is_comet_available,
    is_optuna_available,
    is_ray_available,
    is_tensorboard_available,
    is_wandb_available,
)
# 兼容性处理：新版本 Transformers 可能移除了 transformers.optimization.AdamW
from transformers.optimization import get_linear_schedule_with_warmup, get_scheduler
try:
    from transformers.optimization import AdamW as HF_AdamW  # 旧版本存在
    AdamW = HF_AdamW
except Exception:
    from torch.optim import AdamW  # 新版本请直接使用 PyTorch 自带的 AdamW

from transformers.trainer_callback import (
    DefaultFlowCallback,
    ProgressCallback,
)
from transformers.trainer_utils import (
    default_compute_objective,
)
from transformers.training_args import TrainingArguments
from transformers.utils import logging
from transformers.trainer_utils import TrainOutput

from tqdm import tqdm, trange
from torch.optim import SGD
import torch.nn.functional as F

from src.linearhead_trainer import LinearHeadTrainer
from transformers.trainer_callback import TrainerState

import copy

_use_native_amp = False
_use_apex = False

DEFAULT_CALLBACKS = [DefaultFlowCallback]
DEFAULT_PROGRESS_CALLBACK = ProgressCallback

if is_in_notebook():
    from transformers.utils.notebook import NotebookProgressCallback

    DEFAULT_PROGRESS_CALLBACK = NotebookProgressCallback

# Check if Pytorch version >= 1.6 to switch between Native AMP and Apex
if version.parse(torch.__version__) < version.parse("1.6"):
    from transformers.file_utils import is_apex_available

    if is_apex_available():
        from apex import amp
    _use_apex = True
else:
    _use_native_amp = True
    from torch.cuda.amp import autocast

if version.parse(torch.__version__) < version.parse("1.2"):
    _use_ddp_no_sync = False
else:
    _use_ddp_no_sync = True

if is_datasets_available():
    import datasets

if is_torch_tpu_available():
    import torch_xla.core.xla_model as xm
    import torch_xla.debug.metrics as met
    import torch_xla.distributed.parallel_loader as pl

if is_tensorboard_available():
    from transformers.integrations import TensorBoardCallback

    DEFAULT_CALLBACKS.append(TensorBoardCallback)


if is_wandb_available():
    from transformers.integrations import WandbCallback

    DEFAULT_CALLBACKS.append(WandbCallback)

if is_comet_available():
    from transformers.integrations import CometCallback

    DEFAULT_CALLBACKS.append(CometCallback)

if is_optuna_available():
    import optuna

if is_ray_available():
    from ray import tune

logger = logging.get_logger(__name__)
logger.setLevel(logging.INFO)


from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

def _debug_get_train_dataloader_unused(self):
    """
    UNUSED: this is a module-level function and does not override Trainer.get_train_dataloader.
    Override HF default to ensure dataloader_shuffle/data_seed take effect,
    and decouple data-order RNG from MeZO perturbation RNG.
    """
    if self.train_dataset is None:
        raise ValueError("Trainer: training requires a train_dataset.")

    # Resolve shuffle flag (robust to string 'True'/'False')
    shuffle_flag = getattr(self.args, "dataloader_shuffle", True)
    if isinstance(shuffle_flag, str):
        shuffle_flag = shuffle_flag.lower() in ("1", "true", "yes", "y")
    shuffle_flag = bool(shuffle_flag)

    # data_seed fallback
    data_seed = getattr(self.args, "data_seed", None)
    if data_seed is None:
        data_seed = getattr(self.args, "seed", 42)

    # Sampler
    if self.args.local_rank != -1:
        sampler = DistributedSampler(self.train_dataset, shuffle=shuffle_flag, seed=int(data_seed))
    else:
        if shuffle_flag:
            g = torch.Generator()
            g.manual_seed(int(data_seed))
            sampler = RandomSampler(self.train_dataset, generator=g)
        else:
            sampler = SequentialSampler(self.train_dataset)

    return DataLoader(
        self.train_dataset,
        sampler=sampler,
        batch_size=self.args.train_batch_size,
        collate_fn=self.data_collator,
        drop_last=self.args.dataloader_drop_last,
        num_workers=self.args.dataloader_num_workers,
        pin_memory=self.args.dataloader_pin_memory,
    )

########## The above part is copied from Transformers' trainer (3.4.0) ##########

def default_dev_objective(metrics):
    """
    Objective used for picking the best model on development sets
    """
    if "eval_mnli/acc" in metrics:
        return metrics["eval_mnli/acc"]
    elif "eval_mnli-mm/acc" in metrics:
        return metrics["eval_mnli-mm/acc"]
    elif "eval_f1" in metrics:
        return metrics["eval_f1"]
    elif "eval_mcc" in metrics:
        return metrics["eval_mcc"]
    elif "eval_pearson" in metrics:
        return metrics["eval_pearson"]
    elif "eval_acc" in metrics:
        return metrics["eval_acc"]

    raise Exception("No metric founded for {}".format(metrics))

class Trainer(LinearHeadTrainer):

    # ================= CSV 训练日志（每步）=================
    def _setup_metrics_csv(self):
        """创建日志文件夹与 CSV 文件；根据是否使用自适应 h / c 缩放 / 分层 h 命名文件。"""
        base_dir = getattr(self.args, "output_dir", "./outputs") or "./outputs"
        log_dir = os.path.join(base_dir, "metrics_logs")
        os.makedirs(log_dir, exist_ok=True)
        # 根据开关构造文件名
        use_ah = int(getattr(self.args, "use_adaptive_h", False))
        use_cs = int(getattr(self.args, "use_c_scale", False))
        use_lh = int(getattr(self.args, "use_layerwise_h", False))
        filename = f"metrics_adaptiveH-{use_ah}_cscale-{use_cs}_layerwiseH-{use_lh}.csv"
        self._metrics_csv_path = os.path.join(log_dir, filename)
        # 若文件不存在则写入表头
        if not os.path.exists(self._metrics_csv_path):
            with open(self._metrics_csv_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["epoch", "global_step", "train_loss", "train_acc", "eval_ran", "eval_loss", "eval_acc"])

    def _compute_train_acc(self, model, inputs) -> Optional[float]:
        """计算当前 batch 的训练准确率（不影响梯度）。若任务非分类或缺少 labels，返回 None。"""
        try:
            if "labels" not in inputs:
                return None
            model.eval()
            _in = self._prepare_inputs(inputs)
            with torch.no_grad():
                out = model(**_in)
                # 兼容 (loss, logits) 或 只返回 logits 的情况
                if isinstance(out, (tuple, list)):
                    logits = out[1] if len(out) > 1 else out[0]
                elif hasattr(out, "logits"):
                    logits = out.logits
                else:
                    return None
                preds = torch.argmax(logits, dim=-1)
                labels = _in["labels"]
                acc = (preds == labels).float().mean().item()
                return float(acc)
        except Exception:
            return None

    def _extract_eval_acc(self, metrics: Dict[str, float]) -> Optional[float]:
        """从 evaluate() 的 metrics 字典里尽量找出准确率字段。常见键：eval_accuracy / eval_acc / eval_mnli/acc 等。"""
        if not isinstance(metrics, dict):
            return None
        # 优先常见命名
        for k in ["eval_accuracy", "eval_acc", "accuracy", "acc"]:
            if k in metrics and isinstance(metrics[k], (int, float)):
                return float(metrics[k])
        # 次优：包含 acc 的键（如 eval_mnli/acc）
        for k, v in metrics.items():
            if isinstance(k, str) and "acc" in k and isinstance(v, (int, float)):
                return float(v)
        return None
    def get_train_dataloader(self):
        """Override to ensure dataloader_shuffle/data_seed actually take effect.

        NOTE: We implement this here (inside the class) because module-level functions do not override Trainer methods.
        """
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        # Resolve shuffle flag (robust to string 'True'/'False')
        shuffle_flag = getattr(self.args, "dataloader_shuffle", True)
        shuffle_flag_raw = shuffle_flag
        if isinstance(shuffle_flag, str):
            shuffle_flag = shuffle_flag.lower() in ("1", "true", "yes", "y")
        shuffle_flag = bool(shuffle_flag)

        # data_seed fallback
        data_seed = getattr(self.args, "data_seed", None)
        if data_seed is None:
            data_seed = getattr(self.args, "seed", 42)

        logger.info(
            f"[dataloader][override] Trainer.get_train_dataloader active. "
            f"shuffle_raw={shuffle_flag_raw} type={type(shuffle_flag_raw).__name__} -> shuffle={shuffle_flag}; "
            f"data_seed={data_seed} seed={getattr(self.args,'seed','?')} local_rank={getattr(self.args,'local_rank','?')}"
        )

        # Choose sampler
        if getattr(self.args, "local_rank", -1) != -1:
            sampler = DistributedSampler(self.train_dataset, shuffle=shuffle_flag, seed=int(data_seed))
        else:
            if shuffle_flag:
                g = torch.Generator()
                g.manual_seed(int(data_seed))
                sampler = RandomSampler(self.train_dataset, generator=g)
            else:
                sampler = SequentialSampler(self.train_dataset)

        logger.info(f"[dataloader][override] selected sampler={type(sampler).__name__}")

        return DataLoader(
            self.train_dataset,
            sampler=sampler,
            batch_size=self.args.train_batch_size,
            collate_fn=self.data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )
    # =====================================================

    # ================= Adaptive-h helpers (rolling probe buffer) =================
    def _get_init_h(self) -> float:
        h0 = getattr(self.args, "init_h", None)
        if h0 is None:
            h0 = getattr(self.args, "initial_h", None)
        if h0 is None:
            h0 = getattr(self.args, "zero_order_eps", 1e-4)
        return float(h0)

    def _smooth_h_log_ema(self, prev_h: float, new_h: float, beta: float, h_min: float, h_max: float) -> float:
        if (not math.isfinite(prev_h)) or prev_h <= 0:
            prev_h = h_min
        if (not math.isfinite(new_h)) or new_h <= 0:
            return float(min(h_max, max(h_min, prev_h)))
        beta = float(min(1.0, max(0.0, beta)))
        h = math.exp((1 - beta) * math.log(prev_h) + beta * math.log(new_h))
        return float(min(h_max, max(h_min, h)))

    def _update_h_probe_buffer(self, inputs):
        """Maintain a rolling buffer of recent batches (CPU tensors) for stable h estimation."""
        buf_size = int(getattr(self.args, "adaptive_h_probe_buffer_size", 64))
        if buf_size <= 0:
            return

        buf = getattr(self, "_h_probe_buffer", None)
        if not isinstance(buf, list):
            buf = []

        # Store a lightweight CPU copy to avoid holding GPU memory.
        stored = inputs
        try:
            if isinstance(inputs, dict):
                stored = {}
                for k, v in inputs.items():
                    if isinstance(v, torch.Tensor):
                        stored[k] = v.detach().cpu()
                    else:
                        stored[k] = v
        except Exception:
            stored = inputs

        buf.append(stored)
        if len(buf) > buf_size:
            buf = buf[-buf_size:]
        self._h_probe_buffer = buf

    def _get_h_estimation_inputs(self, train_dataloader, base_inputs=None, num_batches: int = 1):
        """Return batches for h estimation: current batch + random sample from rolling probe buffer."""
        num_batches = max(1, int(num_batches))
        batches = []

        # Always include current batch first (shallow copy to avoid in-place mutation)
        if base_inputs is not None:
            batches.append(dict(base_inputs) if isinstance(base_inputs, dict) else base_inputs)

        buf = getattr(self, "_h_probe_buffer", None)
        need = num_batches - len(batches)

        # Prefer sampling from buffer excluding the most recent element (which is often base_inputs)
        if need > 0 and isinstance(buf, list) and len(buf) > 0:
            pool = buf
            if base_inputs is not None and len(buf) > 1:
                pool = buf[:-1]
            if len(pool) > 0:
                replace = len(pool) < need
                idxs = np.random.choice(len(pool), size=need, replace=replace)
                for idx in idxs:
                    item = pool[int(idx)]
                    batches.append(dict(item) if isinstance(item, dict) else item)

        # Fallback: independent iterator over train_dataloader if buffer is empty
        it = getattr(self, "_h_est_iter", None)
        if it is None:
            it = iter(train_dataloader)
            self._h_est_iter = it
        while len(batches) < num_batches:
            try:
                b = next(it)
            except StopIteration:
                it = iter(train_dataloader)
                self._h_est_iter = it
                b = next(it)
            batches.append(dict(b) if isinstance(b, dict) else b)

        return batches[:num_batches]

    def estimate_adaptive_h_multi(
        self,
        model,
        loss_fn,
        inputs_list,
        layer_name=None,
        num_directions=1,
        reduce="mean",
        h_min=1e-5,
        h_max=0.5,
    ):
        """Estimate h using multi-batch x multi-direction, then aggregate by mean/median."""
        gamma = 3 ** (1 / 3)
        h_vals, eps_vals, nu3_vals = [], [], []

        for inputs in inputs_list:
            for _ in range(max(1, int(num_directions))):
                inp1 = dict(inputs) if isinstance(inputs, dict) else inputs
                eps_i = float(self.estimate_noise(model, loss_fn, inp1, layer_name=layer_name))
                if (not math.isfinite(eps_i)) or eps_i <= 0.0:
                    continue

                inp2 = dict(inputs) if isinstance(inputs, dict) else inputs
                nu3_i = float(self.estimate_nu3(model, loss_fn, inp2, layer_name=layer_name, eps_f_override=eps_i))
                # Drop invalid directions (do NOT clamp), otherwise h_raw can be spuriously inflated.
                if (not math.isfinite(nu3_i)) or (nu3_i <= 0.0):
                    continue

                h_i = (eps_i / nu3_i) ** (1 / 3) * gamma
                if (not math.isfinite(h_i)) or h_i <= 0.0:
                    continue

                h_i = float(min(h_max, max(h_min, h_i)))
                h_vals.append(h_i)
                eps_vals.append(eps_i)
                nu3_vals.append(float(nu3_i))

        if len(h_vals) == 0:
            return float("nan"), float("nan"), float("nan")

        arr = np.asarray(h_vals, dtype=np.float64)
        reduce = str(reduce or "mean").lower()
        h_est = float(np.median(arr)) if reduce == "median" else float(np.mean(arr))
        eps_est = float(np.mean(eps_vals)) if len(eps_vals) > 0 else float("nan")
        nu3_est = float(np.mean(nu3_vals)) if len(nu3_vals) > 0 else float("nan")
        return h_est, eps_est, nu3_est

    # =============================
    # h-probes (Probe 1/2/3)
    # Enable via environment variables (no need to touch run.py args parser):
    #   H_PROBE=1
    #   H_PROBE_MODE=123   (any subset of 1/2/3)
    #   H_PROBE_EVERY=0    (0 => only at global_step==0; else run every N steps)
    #   H_PROBE_MIN=1e-6, H_PROBE_MAX=1e-2, H_PROBE_NUM=9
    #   H_PROBE_NDIR=8, H_PROBE_NB=2
    # Output: <output_dir>/hprobe.jsonl
    # =============================

    def _hprobe_enabled(self) -> bool:
        v = os.environ.get("H_PROBE", "0").lower()
        return v in ("1", "true", "yes", "y", "on")

    def _hprobe_cfg(self) -> Dict[str, Any]:
        mode = os.environ.get("H_PROBE_MODE", "123")
        every = int(os.environ.get("H_PROBE_EVERY", "0"))
        num_h = int(os.environ.get("H_PROBE_NUM", "9"))
        h_min = float(os.environ.get("H_PROBE_MIN", "1e-6"))
        h_max = float(os.environ.get("H_PROBE_MAX", "1e-2"))
        ndir = int(os.environ.get("H_PROBE_NDIR", "8"))
        nbatch = int(os.environ.get("H_PROBE_NB", "1"))
        alpha = float(os.environ.get("H_PROBE_ALPHA", "2.0"))
        out_name = os.environ.get("H_PROBE_OUT", "hprobe.jsonl")
        return {
            "mode": mode,
            "every": every,
            "num_h": max(num_h, 2),
            "h_min": h_min,
            "h_max": h_max,
            "ndir": max(ndir, 1),
            "nbatch": max(nbatch, 1),
            "alpha": alpha,
            "out_name": out_name,
        }

    def _hprobe_h_list(self, cfg: Dict[str, Any]) -> List[float]:
        h_min, h_max, num_h = float(cfg["h_min"]), float(cfg["h_max"]), int(cfg["num_h"])
        if (not math.isfinite(h_min)) or (not math.isfinite(h_max)) or h_min <= 0 or h_max <= 0:
            h_min, h_max = 1e-6, 1e-2
        if h_min > h_max:
            h_min, h_max = h_max, h_min
        hs = np.logspace(np.log10(h_min), np.log10(h_max), num=num_h, base=10.0)
        return [float(x) for x in hs]

    def _hprobe_use_wd(self, name: str) -> bool:
        n = name.lower()
        return ("bias" not in n) and ("layer_norm" not in n) and ("layernorm" not in n)

    def _hprobe_eval_at(self, model: nn.Module, inputs: Dict[str, Any], h: float, seed: int, mult: float) -> float:
        """Evaluate f(theta + mult*h*z) and restore params, using the same z via seed."""
        # Ensure list exists
        if (not hasattr(self, "named_parameters_to_optim")) or (self.named_parameters_to_optim is None) or (len(self.named_parameters_to_optim) == 0):
            self.named_parameters_to_optim = [(n, p) for n, p in model.named_parameters() if p.requires_grad]

        torch.manual_seed(seed)
        with torch.no_grad():
            for _, p in self.named_parameters_to_optim:
                z = torch.normal(mean=0, std=1, size=p.data.size(), device=p.data.device, dtype=p.data.dtype)
                p.data.add_(mult * float(h) * z)

        val = float(self.zo_forward(model, inputs).item())

        # Restore
        torch.manual_seed(seed)
        with torch.no_grad():
            for _, p in self.named_parameters_to_optim:
                z = torch.normal(mean=0, std=1, size=p.data.size(), device=p.data.device, dtype=p.data.dtype)
                p.data.add_(-mult * float(h) * z)

        return val

    def _hprobe_proj_grad_at(self, model: nn.Module, inputs: Dict[str, Any], h: float, seed: int) -> Tuple[float, float, float, float]:
        fp = self._hprobe_eval_at(model, inputs, h=h, seed=seed, mult=+1.0)
        fm = self._hprobe_eval_at(model, inputs, h=h, seed=seed, mult=-1.0)
        delta = float(fp - fm)
        ghat = float(delta / (2.0 * float(h)))
        return fp, fm, delta, ghat

    def _hprobe_get_lr(self) -> float:
        # Best-effort: prefer scheduler if present, else args.learning_rate
        try:
            if hasattr(self, "lr_scheduler") and self.lr_scheduler is not None:
                lrs = self.lr_scheduler.get_last_lr()
                if isinstance(lrs, (list, tuple)) and len(lrs) > 0:
                    return float(lrs[0])
        except Exception:
            pass
        return float(getattr(self.args, "learning_rate", 1e-4))

    def _hprobe_apply_update(self, model: nn.Module, seed: int, projected_grad: float, lr: float, weight_decay: float):
        if (not hasattr(self, "named_parameters_to_optim")) or (self.named_parameters_to_optim is None) or (len(self.named_parameters_to_optim) == 0):
            self.named_parameters_to_optim = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
        torch.manual_seed(seed)
        with torch.no_grad():
            for name, p in self.named_parameters_to_optim:
                z = torch.normal(mean=0, std=1, size=p.data.size(), device=p.data.device, dtype=p.data.dtype)
                if self._hprobe_use_wd(name):
                    p.data = p.data - lr * (projected_grad * z + weight_decay * p.data)
                else:
                    p.data = p.data - lr * (projected_grad * z)

    def _hprobe_undo_update(self, model: nn.Module, seed: int, projected_grad: float, lr: float, weight_decay: float):
        if (not hasattr(self, "named_parameters_to_optim")) or (self.named_parameters_to_optim is None) or (len(self.named_parameters_to_optim) == 0):
            self.named_parameters_to_optim = [(n, p) for n, p in model.named_parameters() if p.requires_grad]

        # Invert: p_new = (1-lr*wd) p_old - lr*g*z  =>  p_old = (p_new + lr*g*z)/(1-lr*wd)
        denom = float(1.0 - lr * weight_decay)
        if abs(denom) < 1e-12:
            denom = 1.0

        torch.manual_seed(seed)
        with torch.no_grad():
            for name, p in self.named_parameters_to_optim:
                z = torch.normal(mean=0, std=1, size=p.data.size(), device=p.data.device, dtype=p.data.dtype)
                if self._hprobe_use_wd(name):
                    p.data = (p.data + lr * (projected_grad * z)) / denom
                else:
                    p.data = p.data + lr * (projected_grad * z)

    def _hprobe_pick_batches(self, current_inputs: Dict[str, Any], nbatch: int):
        """Prefer adaptive-h rolling buffer (CPU copies). Fallback to current batch."""
        buf = getattr(self, "_h_probe_buffer", None)
        batches = []
        if isinstance(buf, list) and len(buf) > 0:
            replace = len(buf) < nbatch
            idxs = np.random.choice(len(buf), size=nbatch, replace=replace)
            for idx in idxs:
                item = buf[int(idx)]
                batches.append(dict(item) if isinstance(item, dict) else item)
            eval_batch = batches[-1] if len(batches) > 1 else batches[0]
        else:
            batches = [current_inputs]
            eval_batch = current_inputs
        return batches, eval_batch

    def _hprobe_write_jsonl(self, rows: List[Dict[str, Any]], out_path: str):
        # Only write on main process if distributed
        try:
            if hasattr(self.args, "local_rank") and getattr(self.args, "local_rank", -1) not in (-1, 0):
                return
        except Exception:
            pass

        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "a", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    def _hprobe_run_once(self, model: nn.Module, current_inputs: Dict[str, Any]):
        cfg = self._hprobe_cfg()
        mode = str(cfg["mode"])
        hs = self._hprobe_h_list(cfg)

        # Save RNG states so probe does not affect training randomness
        np_state = np.random.get_state()
        py_state = random.getstate()
        torch_state = torch.random.get_rng_state()
        cuda_states = None
        try:
            if torch.cuda.is_available():
                cuda_states = torch.cuda.get_rng_state_all()
        except Exception:
            cuda_states = None

        try:
            # Ensure parameter list exists
            if (not hasattr(self, "named_parameters_to_optim")) or (self.named_parameters_to_optim is None) or (len(self.named_parameters_to_optim) == 0):
                self.named_parameters_to_optim = [(n, p) for n, p in model.named_parameters() if p.requires_grad]

            nbatch = int(cfg["nbatch"])
            batches, eval_batch = self._hprobe_pick_batches(current_inputs, nbatch=nbatch)

            # base losses
            base_losses = [float(self.zo_forward(model, b).item()) for b in batches]
            base_eval = float(self.zo_forward(model, eval_batch).item())

            lr = float(self._hprobe_get_lr())
            wd = float(getattr(self.args, "weight_decay", 0.0))
            gs = int(getattr(self.state, "global_step", 0))

            rows = []
            for h in hs:
                g_list, r_list, absdf_list = [], [], []
                dL_same_list, dL_new_list = [], []

                for bi, batch in enumerate(batches):
                    base_b = base_losses[bi]
                    for _ in range(int(cfg["ndir"])):
                        seed = int(np.random.randint(0, 1_000_000_000))

                        _, _, delta, ghat = self._hprobe_proj_grad_at(model, batch, h=float(h), seed=seed)

                        if "2" in mode:
                            absdf_list.append(abs(delta))

                        if "1" in mode:
                            _, _, _, ghat2 = self._hprobe_proj_grad_at(model, batch, h=float(cfg["alpha"]) * float(h), seed=seed)
                            R = abs(ghat2 - ghat) / max(1.0, abs(ghat))
                            r_list.append(float(R))
                            g_list.append(float(ghat))

                        if "3" in mode:
                            self._hprobe_apply_update(model, seed=seed, projected_grad=ghat, lr=lr, weight_decay=wd)
                            new_same = float(self.zo_forward(model, batch).item())
                            new_eval = float(self.zo_forward(model, eval_batch).item())
                            self._hprobe_undo_update(model, seed=seed, projected_grad=ghat, lr=lr, weight_decay=wd)

                            dL_same_list.append(new_same - base_b)
                            dL_new_list.append(new_eval - base_eval)

                def _mean_std(x):
                    if not x:
                        return None, None
                    arr = np.asarray(x, dtype=np.float64)
                    return float(arr.mean()), float(arr.std())

                row = {
                    "global_step": gs,
                    "h": float(h),
                    "lr": lr,
                    "wd": wd,
                    "mode": mode,
                    "ndir": int(cfg["ndir"]),
                    "nbatch": len(batches),
                }

                if "1" in mode:
                    mR, sR = _mean_std(r_list)
                    mg, sg = _mean_std(g_list)
                    row.update({"R_mean": mR, "R_std": sR, "ghat_mean": mg, "ghat_std": sg})

                if "2" in mode:
                    mdf, sdf = _mean_std(absdf_list)
                    row.update({"abs_delta_f_mean": mdf, "abs_delta_f_std": sdf})

                if "3" in mode:
                    md1, sd1 = _mean_std(dL_same_list)
                    md2, sd2 = _mean_std(dL_new_list)
                    row.update({"dL_same_mean": md1, "dL_same_std": sd1, "dL_new_mean": md2, "dL_new_std": sd2})

                rows.append(row)

            out_path = os.path.join(getattr(self.args, "output_dir", "./outputs") or "./outputs", cfg["out_name"])
            self._hprobe_write_jsonl(rows, out_path)
            try:
                logger.info(f"[hprobe] wrote {len(rows)} rows to {out_path} (step={gs}, mode={mode})")
            except Exception:
                pass

        finally:
            # Restore RNG states
            np.random.set_state(np_state)
            random.setstate(py_state)
            torch.random.set_rng_state(torch_state)
            try:
                if cuda_states is not None and torch.cuda.is_available():
                    torch.cuda.set_rng_state_all(cuda_states)
            except Exception:
                pass

    def _hprobe_maybe_run(self, model: nn.Module, current_inputs: Dict[str, Any]):
        if not self._hprobe_enabled():
            return

        cfg = self._hprobe_cfg()
        every = int(cfg["every"])
        gs = int(getattr(self.state, "global_step", 0))

        should = (gs == 0) if every <= 0 else ((gs % every) == 0)
        if not should:
            return

        if getattr(self, "_hprobe_last_step", None) == gs:
            return
        self._hprobe_last_step = gs

        # Only meaningful for ZO mode
        if not getattr(self.args, "zero_order_optim", False):
            return

        self._hprobe_run_once(model, current_inputs)

    # =============================================================================
    """
    Adding some functions based on Transformers' Trainer class.
    """

    # === Begin Adaptive h (Berahas et al.) ===
    def estimate_nu3(self, model, loss_fn, inputs, tau1=10.0, tau2=0.1, layer_name: Optional[str]=None, eps_f_override: Optional[float]=None):
        """
        估计第三阶导数尺度 ν3（More & Wild / Berahas 风格的“尺度检测 + 接受测试”）。

        核心思路：
        1) 先估计该 batch/方向下的噪声幅度 ε_f（可用 eps_f_override 传入）。
        2) 对扰动尺度 h 做“可接受性测试”（论文式 18a/18b）：
           - (18a) 信号测试：Δ(h)/ε_f >= tau1
                 其中 Δ(h) = |f(-h) - 2 f(0) + f(+h)|
           - (18b) 局部性测试：prox(±h) <= tau2
                 prox(+)=|f(+h)-f(0)|/max(|f(0)|,|f(+h)|)，同理 prox(-)
        3) h 的搜索策略：
           - 从 h_start（接近当前训练 eps）开始先测一次：
             * 若 prox 失败：说明 h 太大（非局部）→ 向下扫描 h/growth^i
             * 若 snr 失败但 prox 通过：说明 h 太小（噪声主导）→ 向上扫描 h*growth^i
           - 找到第一个同时满足 (18a)(18b) 的 h 就接受；若找不到，返回 NaN（让上层丢弃该方向）。
        4) 在接受的 h 上，用三阶中心差分估计：
           Δ3(h) = |-f(2h) + 2f(h) - 2f(-h) + f(-2h)|
           ν3_hat = Δ3(h) / (2 h^3)
           若出现 Δ3=0（数值地板）或 ν3 无效，则在“仍满足 (18a)(18b)”前提下尝试放大 h 重试；
           重试仍失败则返回 NaN（丢弃该方向）。

        返回：float ν3（>0）或 NaN（表示该方向/该 batch 下 ν3 不可靠）。
        """
        # Estimate ε_f for this context (allow override for multi-batch / multi-direction averaging)
        if eps_f_override is not None:
            eps_f = float(eps_f_override)
        else:
            try:
                eps_f = float(getattr(self, "epsilon_f", None))
                if not math.isfinite(eps_f) or eps_f <= 0:
                    raise ValueError("invalid epsilon_f")
            except Exception:
                eps_f = float(self.estimate_noise(model, self.compute_loss, inputs, layer_name=layer_name))
                logger.info(f"[estimate_nu3] on-the-fly epsilon_f = {eps_f:.3e}")
        names, params = [], []
        for name, param in model.named_parameters():
            if self.should_optim(name, param):
                if layer_name is not None and (layer_name not in name):
                    continue
                names.append(name)
                params.append(param)
        param_shapes = [p.data.shape for p in params]
        param_numels = [p.data.numel() for p in params]
        total_numel = sum(param_numels)
        device = params[0].data.device if params else torch.device("cpu")
        dtype = params[0].data.dtype if params else torch.float32
        originals = [p.data.detach().clone().to(dtype=torch.float64) for p in params]
        # Random Gaussian direction z (float64), consistent with training perturbations (no normalization)
        z_flat = torch.randn(total_numel, dtype=torch.float64, device=device)
        z_splits = torch.split(z_flat, param_numels)
        z_list = [z.view(shape) for z, shape in zip(z_splits, param_shapes)]

        def set_params(alpha: float):
            for p, orig, z in zip(params, originals, z_list):
                p.data.copy_((orig + alpha * z).to(dtype=dtype))
        # f0 at base
        with torch.no_grad():
            for p, orig in zip(params, originals):
                p.data.copy_(orig.to(dtype=dtype))
            f0 = float(self.zo_forward(model, inputs))
        def eval_at(alpha: float) -> float:
            try:
                with torch.no_grad():
                    set_params(alpha)
                    val = float(self.zo_forward(model, inputs))
            finally:
                for p, orig in zip(params, originals):
                    p.data.copy_(orig.to(dtype=dtype))
            return val

        # --- Δ(h)-based scale detection (paper's second-order difference) ---
        def dh_tests_on(h_local: float):
            """Return (snr_ok, prox_ok, Delta(h), snr_val, (prox_plus, prox_minus), aux_Dh)."""
            f1  = eval_at( 1.0 * h_local)
            fm1 = eval_at(-1.0 * h_local)

            # Paper (18a): second-order difference
            # Δ(h) = | f(t0-h) - 2 f(t0) + f(t0+h) |
            delta2 = abs(fm1 - 2.0 * f0 + f1)

            # Optional auxiliary (not used for acceptance): D(h)=|f(+h)-f0|+|f(-h)-f0|
            aux_dh = abs(f1 - f0) + abs(fm1 - f0)

            # SNR test on Δ(h): Δ(h)/eps_f >= tau1
            snr_val = delta2 / max(eps_f, 1e-30)
            snr_ok = snr_val >= tau1

            # Proximity: relative change at ±h should be small enough (locality)
            prox_plus = abs(f1 - f0) / max(abs(f0), abs(f1), 1e-30)
            prox_minus = abs(fm1 - f0) / max(abs(f0), abs(fm1), 1e-30)
            prox_ok = (prox_plus <= tau2) and (prox_minus <= tau2)

            return snr_ok, prox_ok, delta2, snr_val, (prox_plus, prox_minus), aux_dh

        def nu3_hat_at(h_local: float):
            """Compute ν3_hat at a chosen h using 3rd-order central finite difference."""
            f2  = eval_at( 2.0 * h_local)
            f1  = eval_at( 1.0 * h_local)
            fm1 = eval_at(-1.0 * h_local)
            fm2 = eval_at(-2.0 * h_local)
            delta3 = abs(-f2 + 2.0 * f1 - 2.0 * fm1 + fm2)
            nu3_hat = delta3 / (2.0 * (h_local ** 3 + 1e-30))
            return float(nu3_hat), float(delta3)

        # === Choose h via Δ(h)-based scale detection, then estimate ν3 at that h ===
        # In ZO training, the finite-difference radius (eps) is usually small (e.g., 1e-4~1e-2).
        # We therefore start near the current training eps and scan *downward* to enforce locality.
        tiny = 1e-30
        h_min, h_max = 1e-8, 5e-2

        # Start from current training eps (adaptive_h if enabled, else zero_order_eps)
        eps_train = float(self.adaptive_h) if getattr(self.args, "use_adaptive_h", False) else float(getattr(self.args, "zero_order_eps", 1e-3))
        # Hybrid warm start: theory scale for v3 (eps_f^{1/5}) but never larger than current training eps
        h_theory = float(max(eps_f, tiny) ** 0.2)
        h_start = float(min(eps_train, h_theory))
        h_start = float(min(h_max, max(h_min, h_start)))

        # shrink factor (>1). We try h_start, h_start/growth, h_start/growth^2, ...
        growth = float(getattr(self.args, "dh_h_growth", 2.0))
        max_trials = int(getattr(self.args, "dh_max_trials", 10))

        tried = []  # list of dicts for logging/selection
        chosen_h = None

        # First evaluate at h_start to decide search direction
        snr0, prox0, delta20, snr_val0, (prox_p0, prox_m0), aux_dh0 = dh_tests_on(h_start)
        tried.append({
            "h": h_start,
            "delta2": delta20,
            "aux_dh": aux_dh0,
            "snr": snr_val0,
            "snr_ok": snr0,
            "prox_ok": prox0,
            "prox_p": prox_p0,
            "prox_m": prox_m0,
        })
        try:
            logger.info(
                f"[estimate_nu3][dh-trial-0] layer={layer_name or 'ALL'} h={h_start:.6e}, "
                f"Delta(h)={delta20:.6e}, Delta/eps={snr_val0:.6e}, D(h)={aux_dh0:.6e} (>= {tau1}? {snr0}), "
                f"prox+={prox_p0:.6e}, prox-={prox_m0:.6e} (<= {tau2}? {prox0})"
            )
        except Exception:
            pass

        if snr0 and prox0:
            chosen_h = h_start
        else:
            # If prox fails -> h too large (non-local): scan downward
            # If snr fails but prox OK -> h too small (noise-dominated): scan upward
            if (not prox0):
                mode = "down"
            elif (not snr0) and prox0:
                mode = "up"
            else:
                # both fail: prefer restoring locality first
                mode = "down"

            for i in range(1, max_trials):
                if mode == "down":
                    h_i = h_start / (growth ** i)
                else:
                    h_i = h_start * (growth ** i)
                h_i = float(min(h_max, max(h_min, h_i)))

                snr_ok, prox_ok, delta2, snr_val, (prox_p, prox_m), aux_dh = dh_tests_on(h_i)
                tried.append({
                    "h": h_i,
                    "delta2": delta2,
                    "aux_dh": aux_dh,
                    "snr": snr_val,
                    "snr_ok": snr_ok,
                    "prox_ok": prox_ok,
                    "prox_p": prox_p,
                    "prox_m": prox_m,
                })
                try:
                    logger.info(
                        f"[estimate_nu3][dh-trial-{i}] layer={layer_name or 'ALL'} h={h_i:.6e}, "
                        f"Delta(h)={delta2:.6e}, Delta/eps={snr_val:.6e}, D(h)={aux_dh:.6e} (>= {tau1}? {snr_ok}), "
                        f"prox+={prox_p:.6e}, prox-={prox_m:.6e} (<= {tau2}? {prox_ok})"
                    )
                except Exception:
                    pass

                if snr_ok and prox_ok:
                    chosen_h = h_i
                    break

            if chosen_h is None and mode == "down":
                # If we tried to restore locality but never got prox_ok, the step is likely still too large.
                # Conversely, if we got prox_ok but SNR failed at the smallest h, h may be too small.
                # (This information is used by the fallback policy below.)
                pass

        # If nothing passes both tests, do NOT accept nu3 (paper-consistent).
        if chosen_h is None:
            if len(tried) > 0:
                def _prox_score(x):
                    return float(max(x.get("prox_p", 1e9), x.get("prox_m", 1e9)))
                best_local = min(tried, key=_prox_score)
                logger.warning(
                    f"[estimate_nu3][dh-fail] layer={layer_name or 'ALL'} no h satisfied both tests; "
                    f"best_local_h={best_local['h']:.6e} (prox_max={max(best_local.get('prox_p', 0.0), best_local.get('prox_m', 0.0)):.3e}, "
                    f"Delta/eps={best_local.get('snr', float('nan')):.3e}). Return NaN."
                )
            else:
                logger.warning(f"[estimate_nu3][dh-fail] layer={layer_name or 'ALL'} tried=0. Return NaN.")
            return float("nan")

        # Now compute ν3 at the chosen_h. If Δ3 collapses to 0 (finite-precision floor),
        # retry with slightly larger h while still satisfying BOTH SNR and prox; otherwise return NaN to drop this direction.
        nu3_accept, delta3_accept = nu3_hat_at(chosen_h)

        if (not math.isfinite(nu3_accept)) or (nu3_accept <= 0.0) or (delta3_accept == 0.0):
            max_retry = int(getattr(self.args, "nu3_retry", 3))
            found = False
            for k in range(1, max_retry + 1):
                h_try = float(chosen_h * (2.0 ** k))
                if h_try > h_max:
                    break
                # Keep locality: require BOTH snr_ok and prox_ok at h_try
                try:
                    snr_ok_t, prox_ok_t, _delta2_t, _snr_val_t, (prox_p_t, prox_m_t), _aux = dh_tests_on(h_try)
                except Exception:
                    snr_ok_t, prox_ok_t = False, False
                if not (snr_ok_t and prox_ok_t):
                    continue

                nu3_t, d3_t = nu3_hat_at(h_try)
                if math.isfinite(nu3_t) and (nu3_t > 0.0) and (d3_t != 0.0):
                    chosen_h = h_try
                    nu3_accept, delta3_accept = nu3_t, d3_t
                    found = True
                    try:
                        logger.info(
                            f"[estimate_nu3][retry] layer={layer_name or 'ALL'} use larger local h={h_try:.6e} "
                            f"to avoid Δ3=0; nu3={nu3_accept:.6e}, Δ3={delta3_accept:.6e}"
                        )
                    except Exception:
                        pass
                    break

            if not found:
                logger.warning(
                    f"[estimate_nu3] nu3/Δ3 invalid at chosen_h and retries; return NaN to drop direction. "
                    f"(layer={layer_name or 'ALL'}, chosen_h={chosen_h:.6e}, Δ3={delta3_accept:.3e})"
                )
                return float("nan")

        try:
            logger.info(
                f"[estimate_nu3][final] layer={layer_name or 'ALL'} chosen_h={chosen_h:.6e}, "
                f"nu3={nu3_accept:.6e}, Δ3={delta3_accept:.6e}"
            )
        except Exception:
            pass

        # Optional sanity test log (does not affect the returned value)
        h_test = float(getattr(self.args, "dh_test_h", 1e-2))
        try:
            snr_ok_t, prox_ok_t, delta2_t, snr_val_t, (prox_p_t, prox_m_t), aux_dh_t = dh_tests_on(h_test)
            nu3_t, delta3_t = nu3_hat_at(h_test)
            logger.info(
                f"[estimate_nu3][**TEST**] layer={layer_name or 'ALL'} h_test={h_test:.6e}, "
                f"Delta(h)={delta2_t:.6e}, Delta/eps={snr_val_t:.6e}, D(h)={aux_dh_t:.6e}, snr_ok={snr_ok_t}, prox_ok={prox_ok_t}, "
                f"prox+={prox_p_t:.6e}, prox-={prox_m_t:.6e}, nu3_test={nu3_t:.6e}, Δ3_test={delta3_t:.6e}"
            )
        except Exception:
            pass
        return float(nu3_accept)

    def estimate_noise(self, model, loss_fn, inputs, q=8, delta=1e-6, layer_name: Optional[str]=None):
        # guard: q must be >= 3 because we use j=3 in the difference table
        q = int(q)
        if q < 3:
            q = 3
        delta = float(delta)
        if (not math.isfinite(delta)) or (delta <= 0.0):
            delta = 1e-6
        # === Float64 precision for more stable epsilon_f / nu3 estimation ===
        # Collect all parameters to optimize
        # 若指定 layer_name，则仅在该层参数子空间内估计 ECnoise
        names, params = [], []
        for name, param in model.named_parameters():
            if not self.should_optim(name, param):
                continue
            if layer_name is not None:
                # 仅选取属于该层的参数；不要依赖 self.cs / retrieve_c
                if layer_name not in name:
                    continue
            names.append(name)
            params.append(param)
        param_shapes = [p.data.shape for p in params]
        param_numels = [p.data.numel() for p in params]
        total_numel = sum(param_numels)
        device = params[0].data.device if params else torch.device("cpu")
        dtype = params[0].data.dtype if params else torch.float32
        originals = [p.data.detach().clone().to(dtype=torch.float64) for p in params]
        # Generate a global random Gaussian direction z (float64), consistent with training perturbations (no normalization)
        z_flat = torch.randn(total_numel, dtype=torch.float64, device=device)
        try:
            z_norm = float(torch.norm(z_flat).item())
            logger.info(f"[estimate_noise] ||z||={z_norm:.3e}, delta={delta:.1e}, q={q}")
        except Exception:
            pass
        z_splits = torch.split(z_flat, param_numels)
        z_list = [z.view(shape) for z, shape in zip(z_splits, param_shapes)]

        # Helper to set params to originals + alpha * z
        def set_params(alpha):
            for p, orig, z in zip(params, originals, z_list):
                p.data.copy_((orig + alpha * z).to(dtype=dtype))
        f_vals = []
        try:
            for i in range(q + 1):
                set_params(i * delta)
                with torch.no_grad():
                    f_vals.append(float(self.zo_forward(model, inputs)))
        finally:
            # Restore original parameters
            for p, orig in zip(params, originals):
                p.data.copy_(orig.to(dtype=dtype))
        T = [[0] * (q + 1) for _ in range(q + 1)]
        for i in range(q + 1):
            T[i][0] = f_vals[i]
        for j in range(1, q + 1):
            for i in range(q + 1 - j):
                T[i][j] = T[i+1][j-1] - T[i][j-1]
        j = 3
        gamma = (math.factorial(j)**2) / math.factorial(2*j)
        s_j_sq = gamma / (q + 1 - j) * sum(T[i][j]**2 for i in range(q + 1 - j))
        epsilon_f = math.sqrt(s_j_sq)
        logger.info(f"Estimated epsilon_f: {epsilon_f}")
        return float(epsilon_f)
    # === End Adaptive h ===

    def create_optimizer_and_scheduler(self, num_training_steps: int):
        """
        Based on Transformers' default one, we add fixing layer option where the bottom n layers' parameters
        are fixed and only the top layers are further fine-tuned.
        """
        if self.args.hf_inference_model:
            return

        if self.optimizer is None:
            params = {}
            for n, p in self.model.named_parameters():
                if self.args.fix_layers > 0:
                    if 'encoder.layer' in n:
                        try:
                            layer_num = int(n[n.find('encoder.layer') + 14:].split('.')[0])
                        except:
                            print(n)
                            raise Exception("")
                        if layer_num >= self.args.fix_layers:
                            print('yes', n)
                            params[n] = p
                        else:
                            print('no ', n)
                    elif 'embeddings' in n:
                        print('no ', n)
                    else:
                        print('yes', n)
                        params[n] = p
                else:
                    params[n] = p
            no_decay = ["bias", "LayerNorm.weight"]
            optimizer_grouped_parameters = [
                {
                    "params": [p for n, p in params.items() if not any(nd in n for nd in no_decay)],
                    "weight_decay": self.args.weight_decay,
                },
                {
                    "params": [p for n, p in params.items() if any(nd in n for nd in no_decay)],
                    "weight_decay": 0.0,
                },
            ]
            if self.args.optimizer == 'adam':
                self.optimizer = AdamW(
                    optimizer_grouped_parameters,
                    lr=self.args.learning_rate,
                    betas=(self.args.adam_beta1, self.args.adam_beta2),
                    eps=self.args.adam_epsilon,
                )
            elif self.args.optimizer == 'sgd':
                self.optimizer = SGD(
                    optimizer_grouped_parameters,
                    lr=self.args.learning_rate
                )
            else:
                raise NotImplementedError
        if self.lr_scheduler is None:
            self.lr_scheduler = get_scheduler(
                self.args.lr_scheduler_type,
                optimizer=self.optimizer,
                num_warmup_steps=self.args.get_warmup_steps(num_training_steps),
                num_training_steps=num_training_steps,
            )

    def should_optim(self, name, param):
        return (not self.args.layer_wise_optim or f".{self.state.global_step % self.model.config.num_hidden_layers}." in name) and param.requires_grad

    def zo_forward(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
        model.eval()
        inputs = self._prepare_inputs(inputs)
        if self.args.optimize_acc:
            loss, logits = model(**inputs)
            preds = F.softmax(logits, dim=-1)
            acc = torch.sum(torch.argmax(preds, 1) == inputs['labels']) / len(preds)
            loss = -acc
        else:
            with self.compute_loss_context_manager():
                loss = self.compute_loss(model, inputs)
            if self.args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
        self.state.zo_forward_step += 1
        return loss.detach()

    def efficient_perturb_parameters(self, model: nn.Module, random_seed: int, scaling_factor=1):
        torch.manual_seed(random_seed)
        # 需要 name 以支持分层 h
        for name, param in self.named_parameters_to_optim:
            z = torch.normal(mean=0, std=1, size=param.data.size(), device=param.data.device, dtype=param.data.dtype)
            # === Begin Adaptive h (Berahas et al.) ===
            # 若启用按层 h（use_layerwise_h=True），则针对该参数所在层选用分层步长；否则使用全局 adaptive_h
            if getattr(self.args, "use_adaptive_h", False):
                if getattr(self.args, "use_layerwise_h", False):
                    cname = self.retrieve_c(name)
                    if hasattr(self, "layerwise_h") and isinstance(self.layerwise_h, dict) and (cname in self.layerwise_h):
                        _h = self.layerwise_h[cname]
                        eps = float(_h.item()) if isinstance(_h, torch.Tensor) else float(_h)
                    else:
                        eps = float(self.adaptive_h)
                else:
                    eps = float(self.adaptive_h)
            else:
                eps = float(self.args.zero_order_eps)
            param.data = param.data + scaling_factor * z * eps
            # === End Adaptive h ===
        return model

    def norm_perturb_parameters(self, model: nn.Module, random_vector=None, scaling_factor=1):
        if random_vector is None:
            random_vector = {}

        for name, param in self.named_parameters_to_optim:
            if name in random_vector:
                z = random_vector[name]
            else:
                z = torch.normal(mean=0, std=1, size=param.data.size(), device=param.data.device, dtype=param.data.dtype)
                random_vector[name] = z

            cname = self.retrieve_c(name)
            # === C-缩放开关：是否用每层的 c 值来缩放扰动（等价于缩放 h）===
            # 说明：若 use_c_scale=False，则完全忽略 cs（与新方法一致，不做分层缩放）。
            if getattr(self.args, "use_c_scale", False) and cname in self.cs:
                # 防止除 0：若该层 c==0，退化为不缩放
                if isinstance(self.cs[cname], torch.Tensor):
                    c_val = self.cs[cname].item() if self.cs[cname].numel()==1 else float(self.cs[cname].mean())
                else:
                    c_val = float(self.cs[cname])
                if c_val != 0.0 and math.isfinite(c_val):
                    z = z / c_val

            # === Begin Adaptive h (Berahas et al.) ===
            if getattr(self.args, "use_adaptive_h", False):
                if getattr(self.args, "use_layerwise_h", False):
                    cname = self.retrieve_c(name)
                    if hasattr(self, "layerwise_h") and isinstance(self.layerwise_h, dict) and (cname in self.layerwise_h):
                        _h = self.layerwise_h[cname]
                        eps = float(_h.item()) if isinstance(_h, torch.Tensor) else float(_h)
                    else:
                        eps = float(self.adaptive_h)
                else:
                    eps = float(self.adaptive_h)
            else:
                eps = float(self.args.zero_order_eps)
            param.data = param.data + scaling_factor * z * eps
            # === End Adaptive h ===

        return model, random_vector

    def perturb_parameters(self, model: nn.Module, random_vector=None, scaling_factor=1):
        if random_vector is None:
            random_vector = {}

        for name, param in self.named_parameters_to_optim:
            if name in random_vector:
                z = random_vector[name]
            else:
                z = torch.normal(mean=0, std=1, size=param.data.size(), device=param.data.device, dtype=param.data.dtype)
                random_vector[name] = z
            # === Begin Adaptive h (Berahas et al.) ===
            eps = float(self.adaptive_h) if getattr(self.args, "use_adaptive_h", False) else float(self.args.zero_order_eps)
            param.data = param.data + scaling_factor * z * eps
            # === End Adaptive h ===

        return model, random_vector

    def perturb_single_layer(self, model, layer_name, random_vector=None, scaling_factor=1):
        if random_vector is None:
            random_vector = {}

        for name, param in self.named_parameters_to_optim:
            cname = self.retrieve_c(name)
            if cname == layer_name:
                if name in random_vector:
                    z = random_vector[name]
                else:
                    z = torch.normal(mean=0, std=1, size=param.data.size(), device=param.data.device, dtype=param.data.dtype)
                    random_vector[name] = z
                # === Begin Adaptive h (Berahas et al.) ===
                # 若启用按层 h（use_layerwise_h=True），则针对该参数所在层选用分层步长；否则使用全局 adaptive_h
                if getattr(self.args, "use_adaptive_h", False):
                    if getattr(self.args, "use_layerwise_h", False):
                        if hasattr(self, "layerwise_h") and isinstance(self.layerwise_h, dict) and (cname in self.layerwise_h):
                            _h = self.layerwise_h[cname]
                            eps = float(_h.item()) if isinstance(_h, torch.Tensor) else float(_h)
                        else:
                            eps = float(self.adaptive_h)
                    else:
                        eps = float(self.adaptive_h)
                else:
                    eps = float(self.args.zero_order_eps)
                param.data = param.data + scaling_factor * z * eps
                # === End Adaptive h ===

        return model, random_vector
# 计算c的地方，这三种方法都是分层计算

    def initialize_c(self, model, inputs):
        # 说明：当 use_c_scale=False 时，cs 仍会被计算（如配置所需/调试用），但在扰动与梯度构造时将被忽略。
        self.named_parameters_to_optim = []
        for name, param in model.named_parameters():
            if self.should_optim(name, param):
                self.named_parameters_to_optim.append((name, param))

        self.cs = {'embed': 0.0, 'lm_head': 0.0}
        # OPT: embed_tokens; embed_positions
        # RoBERTa: embeddings
        self.num_params = copy.deepcopy(self.cs)
        self.num_model_layers = model.config.num_hidden_layers
        layer_name = "layers" if model.config.model_type == "opt" else "layer"
        for i in range(self.num_model_layers):
            self.cs[f'{layer_name}.{i}.'] = 0.0
            self.num_params[f'{layer_name}.{i}.'] = 0

        # === C-缩放总开关：use_c_scale ===
        # 若关闭该开关（默认 False），则本方法走“快速路径”：
        #   1) 不再进行任何基于 ZO / 参数范数 / 反传的逐层 c 估计（这通常较耗时且会做多次 forward/backward）；
        #   2) 直接将每个层位的 c 设为 1.0，相当于“恒等缩放”（后续扰动/梯度阶段也会因开关关闭而完全忽略 cs），
        #      这样可以显著节省初始化/重计算的时间；
        #   3) 仍然保留 layer_names 列表，确保按层 ZO 的流程可以正常迭代各层。
        if not getattr(self.args, "use_c_scale", False):
            for k in self.cs.keys():
                self.cs[k] = 1.0  # 恒等缩放（不会被使用，但避免后续意外除 0）
                self.num_params[k] = 0
            self.layer_names = list(self.cs.keys())
            model.zero_grad()
            return

        # ZO estimation of c's
        if self.args.zo_variant != 'param_norm' and self.args.use_zo_grad_est:
            print('使用ZO estimation of c')
            for layer in self.cs.keys():
                with torch.no_grad():
                    model, z = self.perturb_single_layer(model, layer_name=layer)
                    loss1 = self.zo_forward(model, inputs)
                    model, z = self.perturb_single_layer(model, layer_name=layer, random_vector=z, scaling_factor=-2)
                    loss2 = self.zo_forward(model, inputs)

                eps = self.adaptive_h if getattr(self.args, "use_adaptive_h", False) else self.args.zero_order_eps
                projected_grad = (loss1 - loss2) / (2 * eps)
                self.cs[layer] = torch.abs(projected_grad)

                model, z = self.perturb_single_layer(model, layer_name=layer, random_vector=z)

        # no need to run backprop if we are using parameter norm variant, can just measure them
        elif self.args.zo_variant == 'param_norm':
            for name, param in self.named_parameters_to_optim:
                print(name)
                ckey = self.retrieve_c(name)
                if ckey in self.cs:
                    self.cs[ckey] += torch.sum(param.data ** 2)
                    self.num_params[ckey] += param.data.numel()

            # take sqrt to get norm
            for ckey in self.cs:
                self.cs[ckey] = torch.sqrt(self.cs[ckey])
                if self.args.scale_norm_by_num_params:
                    n = float(self.num_params.get(ckey, 0))
                    denom = math.sqrt(n) if n > 0 else 1.0
                    self.cs[ckey] = self.cs[ckey] / denom

            for ckey in self.cs:
                if self.cs[ckey] != 0:
                    self.cs[ckey] = self.cs[ckey].detach().item()

        # backpropagation estimation fo ZO c's
        #   this is mostly for debugging purposes to disentangle the variance from using ZO to estimate c
        #   from the effectiveness of the preconditioners
        else:
            model.eval()
            inputs = self._prepare_inputs(inputs)
            with self.compute_loss_context_manager():
                loss = self.compute_loss(model, inputs)
            if self.args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            loss.backward()
            for name, param in self.named_parameters_to_optim:
                if param.grad is None:
                    print(name)
                else:
                    ckey = self.retrieve_c(name)
                    if ckey in self.cs:
                        self.cs[ckey] += torch.sum(param.grad ** 2)
                        self.num_params[ckey] += param.grad.numel()

            # take sqrt to get norm
            for ckey in self.cs:
                self.cs[ckey] = torch.sqrt(self.cs[ckey])
                if self.args.scale_norm_by_num_params:
                    n = float(self.num_params.get(ckey, 0))
                    denom = math.sqrt(n) if n > 0 else 1.0
                    self.cs[ckey] = self.cs[ckey] / denom

            for ckey in self.cs:
                if self.cs[ckey] != 0:
                    self.cs[ckey] = self.cs[ckey].detach().item()

        self.layer_names = list(self.cs.keys())
        model.zero_grad()

    def retrieve_c(self, param_name: str) -> str:
        """
        将参数名映射到“层键”（用于分层 c / 分层 h）。
        兼容性：在某些配置下（如 zo_variant=None 或 use_c_scale=False），initialize_c()
        可能尚未被调用，此时 self.cs 尚不存在。为避免 AttributeError，
        这里按以下优先级匹配：
          1) 若存在 self.cs：按 self.cs 的键做子串匹配；
          2) 否则，若存在 self.layer_names：按 layer_names 做子串匹配；
          3) 否则，做最小启发式匹配（embed / lm_head / layer(s).i.）；
          匹配失败则返回空串 ""（表示不归属于任何已知层）。
        """
        # 1) 优先使用 self.cs 的键（若已初始化）
        cs = getattr(self, "cs", None)
        if isinstance(cs, dict) and cs:
            for c_name in cs.keys():
                if c_name and c_name in param_name:
                    return c_name

        # 2) 其次使用已构造的 layer_names（train() 中在 use_layerwise_h=True 时会构造）
        layer_names = getattr(self, "layer_names", None)
        if isinstance(layer_names, (list, tuple)):
            for key in layer_names:
                if key and key in param_name:
                    return key

        # 3) Minimal heuristic fallback
        pn = param_name

        # embeddings
        if ("embeddings" in pn) or ("embed_tokens" in pn) or ("embed_positions" in pn):
            return "embed"

        # head
        if ("lm_head" in pn) or ("classifier" in pn) or ("score" in pn):
            return "lm_head"

        # RoBERTa/BERT: encoder.layer.N.
        m = re.search(r"encoder\.layer\.(\d+)\.", pn)
        if m:
            return f"layer.{m.group(1)}."

        # OPT: layers.N.
        m = re.search(r"(?:^|\.)layers\.(\d+)\.", pn)
        if m:
            return f"layers.{m.group(1)}."

        # generic layer.N.
        m = re.search(r"(?:^|\.)layer\.(\d+)\.", pn)
        if m:
            return f"layer.{m.group(1)}."

        return ""

    def get_num_samples(self):
        if self.args.zero_order_sample_scheduler is None:
            noise_sample_time = 1
        elif self.args.zero_order_sample_scheduler == "linear":
            noise_sample_time = max(1, int(self.state.global_step / self.args.max_steps * self.args.zero_order_sample))
        elif self.args.zero_order_sample_scheduler == "constant":
            noise_sample_time = int(self.args.zero_order_sample)
        else:
            raise NotImplementedError
        # print("Sample %d zs" % (noise_sample_time))

        return noise_sample_time
# 训练的函数
    def train(self, model_path=None, dev_objective=None):
        """
        Main training entry point.

        The training logic is directly borrowed from transformers.Trainer (version 3.0.2).
        Add early stopping.
        """
        if self.args.from_linearhead and model_path is None:
            super().train(model_path, dev_objective) # Train output layer using LinearHeadTrainer

        self.best_dir = None
        self.objective = -float("inf")
        self.dev_objective = dev_objective if dev_objective is not None else default_dev_objective

        # === Begin Adaptive h update freq ===
        # You can also make this self.args.update_noise_every if you want it configurable
        # 更新H的间隔
        update_noise_every = getattr(self.args, "update_noise_every", 1000)
        # === End Adaptive h update freq ===

        # Data loading.
        try:
            logger.info(f"[dataloader][debug] self class={self.__class__.__name__}, get_train_dataloader={self.get_train_dataloader}")
        except Exception:
            pass
        train_dataloader = self.get_train_dataloader()
        # --- Inspect sampler type (RandomSampler / SequentialSampler / DistributedSampler) ---
        try:
            sampler = getattr(train_dataloader, "sampler", None)
            batch_sampler = getattr(train_dataloader, "batch_sampler", None)
            logger.info(f"[dataloader] sampler={type(sampler).__name__}, batch_sampler={type(batch_sampler).__name__}")
            if isinstance(sampler, RandomSampler):
                logger.info("[dataloader] training uses RandomSampler (shuffle).")
            elif isinstance(sampler, SequentialSampler):
                logger.info("[dataloader] training uses SequentialSampler (no shuffle).")
            elif isinstance(sampler, DistributedSampler):
                logger.info("[dataloader] training uses DistributedSampler (sharded).")

            logger.info(f"[dataloader] args.dataloader_shuffle={getattr(self.args, 'dataloader_shuffle', 'MISSING')}")
            logger.info(
                f"[dataloader] args.data_seed={getattr(self.args, 'data_seed', 'MISSING')} args.seed={getattr(self.args, 'seed', 'MISSING')}")
        except Exception as e:
            logger.warning(f"[dataloader] cannot inspect sampler: {e}")
        num_update_steps_per_epoch = len(train_dataloader) // self.args.gradient_accumulation_steps
        if num_update_steps_per_epoch == 0:
            num_update_steps_per_epoch = 1
        if self.args.max_steps > 0:
            t_total = self.args.max_steps
            num_train_epochs = self.args.max_steps // num_update_steps_per_epoch + int(
                self.args.max_steps % num_update_steps_per_epoch > 0
            )
        else:
            t_total = int(len(train_dataloader) // self.args.gradient_accumulation_steps * self.args.num_train_epochs)
            num_train_epochs = self.args.num_train_epochs

        self.create_optimizer_and_scheduler(num_training_steps=t_total)
        optimizer = self.optimizer
        scheduler = self.lr_scheduler

        # Check if saved optimizer or scheduler states exist
        if (
            model_path is not None
            and os.path.isfile(os.path.join(model_path, "optimizer.pt"))
            and os.path.isfile(os.path.join(model_path, "scheduler.pt"))
        ):
            # Load in optimizer and scheduler states
            optimizer.load_state_dict(
                torch.load(os.path.join(model_path, "optimizer.pt"), map_location=self.args.device)
            )
            scheduler.load_state_dict(torch.load(os.path.join(model_path, "scheduler.pt")))

        model = self.model

        if self.args.fp16 and _use_apex:
            if not transformers.is_apex_available():
                raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
            model, optimizer = amp.initialize(model, optimizer, opt_level=self.args.fp16_opt_level)

        # Multi-gpu training (should be after apex fp16 initialization)
        if self.args.n_gpu > 1:
            model = torch.nn.DataParallel(model)

        # Distributed training (should be after apex fp16 initialization)
        if self.args.local_rank != -1:
            model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[self.args.local_rank],
                output_device=self.args.local_rank,
                find_unused_parameters=True,
            )

        # Train
        if transformers.is_torch_tpu_available():
            total_train_batch_size = self.args.train_batch_size * xm.xrt_world_size()
        else:
            total_train_batch_size = (
                self.args.train_batch_size
                * self.args.gradient_accumulation_steps
                * (torch.distributed.get_world_size() if self.args.local_rank != -1 else 1)
            )
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", self.num_examples(train_dataloader))
        logger.info("  Num Epochs = %d", num_train_epochs)
        logger.info("  Instantaneous batch size per device = %d", self.args.per_device_train_batch_size)
        logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d", total_train_batch_size)
        logger.info("  Gradient Accumulation steps = %d", self.args.gradient_accumulation_steps)
        logger.info("  Total optimization steps = %d", t_total)

        self.state = TrainerState()
        # 仅打印一次分层 h 的日志
        self._logged_layerwise_h = False
        # 初始化 CSV 日志文件
        self._setup_metrics_csv()
        _csv_pending = None  # 暂存本 step 的训练度量，待是否有 eval 再一起写入
        self.state.global_step = 0
        start_time = time.time()
        self.state.zo_forward_step = 0
        # === Begin Adaptive h (Berahas et al.) ===
        if getattr(self.args, "use_adaptive_h", False):
            beta = float(getattr(self.args, "adaptive_h_ema_beta", 0.1))
            nb = int(getattr(self.args, "adaptive_h_estimate_num_batches", 4))
            nd = int(getattr(self.args, "adaptive_h_estimate_num_directions", 3))
            reduce = getattr(self.args, "adaptive_h_estimate_reduce", "mean")
            h_min = float(getattr(self.args, "adaptive_h_min", 1e-5))
            h_max = float(getattr(self.args, "adaptive_h_max", 0.5))

            h0 = self._get_init_h()
            h0 = float(min(h_max, max(h_min, h0)))
            self.adaptive_h = float(h0)
            previous_adaptive_h = float(h0)
            logger.info(f"[adaptive h][init] h0={h0:.3e}, beta={beta}")
        else:
            previous_adaptive_h = getattr(self, "adaptive_h", 1e-4)
        # === End Adaptive h ===
        self.epoch = 0
        epochs_trained = 0
        steps_trained_in_current_epoch = 0

        if self.args.gradient_checkpointing:
            model.gradient_checkpointing_enable()

        # Check if continuing training from a checkpoint
        if model_path is not None:
            # set global_step to global_step of last saved checkpoint from model path
            try:
                self.state.global_step = int(model_path.split("-")[-1].split("/")[0])
                epochs_trained = self.state.global_step // (len(train_dataloader) // self.args.gradient_accumulation_steps)
                steps_trained_in_current_epoch = self.state.global_step % (
                    len(train_dataloader) // self.args.gradient_accumulation_steps
                )

                logger.info("  Continuing training from checkpoint, will skip to saved global_step")
                logger.info("  Continuing training from epoch %d", epochs_trained)
                logger.info("  Continuing training from global step %d", self.state.global_step)
                logger.info("  Will skip the first %d steps in the first epoch", steps_trained_in_current_epoch)
            except ValueError:
                self.state.global_step = 0
                logger.info("  Starting fine-tuning.")

        tr_loss = torch.tensor(0.0).to(self.args.device)
        logging_loss_scalar = 0.0
        model.zero_grad()
        metrics = None
        for epoch in range(epochs_trained, int(num_train_epochs)):
            if isinstance(train_dataloader, DataLoader) and isinstance(train_dataloader.sampler, DistributedSampler):
                train_dataloader.sampler.set_epoch(epoch)

            if transformers.is_torch_tpu_available():
                parallel_loader = pl.ParallelLoader(train_dataloader, [self.args.device]).per_device_loader(
                    self.args.device
                )
                epoch_iterator = tqdm(parallel_loader, desc="Iteration", disable=not self.is_local_process_zero())
            else:
                epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=True)

            # Reset the past mems state at the beginning of each epoch if necessary.
            if self.args.past_index >= 0:
                self._past = None

            for step, inputs in enumerate(epoch_iterator):
                if self.args.sync_embedding_layers:
                    assert model.module.model_type == 'opt', 'did not implement embedding layer synchronization for non-OPT models'
                    model.module.model.decoder.embed_tokens.weight = model.module.lm_head.weight

                # estimate c's (param or grad norm) on epoch 0
                if epoch == 0 and step == 0 and self.args.zo_variant is not None:
                    self.initialize_c(model, inputs)
                elif step == 0 and self.args.zo_variant is not None and self.args.recompute_norms:
                    self.initialize_c(model, inputs)

                # Skip past any already trained steps if resuming training
                if steps_trained_in_current_epoch > 0:
                    steps_trained_in_current_epoch -= 1
                    continue
                # --- Rolling probe buffer for adaptive h estimation (and for h-probes when enabled) ---
                if getattr(self.args, "use_adaptive_h", False) or self._hprobe_enabled():
                    self._update_h_probe_buffer(inputs)
                # --- h-probes (Probe 1/2/3): stability / delta-loss floor / one-step gain ---
                self._hprobe_maybe_run(model, inputs)

                if self.args.zero_order_optim:
                    # Get parameters that should be optimized (for layer-wise optimization and prefix-tuning)
                    self.named_parameters_to_optim = []
                    for name, param in model.named_parameters():
                        if self.should_optim(name, param):
                            self.named_parameters_to_optim.append((name, param))

                    if self.args.zo_by_layer:
                        assert not self.args.efficient_zero_order, 'did not implement preconditioned ZO for efficient ZO yet'
                        assert self.args.zero_order_use_trainer_optim, 'preconditioned ZO requires using the trainer optimizer'
                        num_zs = self.get_num_samples()
                        layers = [np.random.choice(self.layer_names)] if self.args.pc_rnd_layer else self.layer_names

                        # for each layer: perturb only that layer and store the gradient estimates in the grad buffer
                        for layer in self.layer_names:
                            for _ in range(num_zs):
                                # === C-缩放开关：是否用每层的 c 值来缩放扰动（等价于缩放 h）===
                                if getattr(self.args, "use_c_scale", False):
                                    c_i = self.cs[layer]
                                    # 将可能的张量/标量统一成 float，且避免除 0
                                    if isinstance(c_i, torch.Tensor):
                                        c_i_val = c_i.item() if c_i.numel()==1 else float(c_i.mean())
                                    else:
                                        c_i_val = float(c_i)
                                    c_i_val = 1.0 if (c_i_val == 0.0 or not math.isfinite(c_i_val)) else c_i_val
                                else:
                                    # 关闭 C-缩放：按新方法，不做分层缩放
                                    c_i_val = 1.0
                                model, random_vector = self.perturb_single_layer(model, layer, scaling_factor=1.0/c_i_val)
                                loss1 = self.zo_forward(model, inputs)
                                model, random_vector = self.perturb_single_layer(model, layer, random_vector=random_vector, scaling_factor=-2.0/c_i_val)
                                loss2 = self.zo_forward(model, inputs)
                                model, random_vector = self.perturb_single_layer(model, layer, random_vector=random_vector, scaling_factor=1.0/c_i_val)

                                # Debugging: check for NaN in losses
                                if torch.isnan(loss1).item() or torch.isnan(loss2).item():
                                    logger.warning("NaN encountered in loss during ZO forward step.")

                                # === Begin Adaptive h (Berahas et al.) ===
                                eps = self.adaptive_h if getattr(self.args, "use_adaptive_h", False) else self.args.zero_order_eps
                                projected_grad = (loss1 - loss2) / (2 * eps)
                                # Debugging: check for NaN or Inf in projected_grad
                                if torch.isnan(projected_grad).item() or torch.isinf(projected_grad).item():
                                    logger.warning(f"projected_grad became invalid. loss1: {loss1.item()}, loss2: {loss2.item()}, eps: {eps}")
                                # === End Adaptive h ===
                                # scale grad according to number of zs sampled
                                if not self.args.scale_lr_with_samples:
                                    projected_grad = projected_grad / float(num_zs)

                                # 在写入 grad 前，用 z_tilde 乘回 c（若启用）
                                for name, param in self.named_parameters_to_optim:
                                    if self.retrieve_c(name) == layer:
                                        z_tilde = random_vector[name] * (c_i_val if getattr(self.args, "use_c_scale", False) else 1.0)
                                        if param.grad is None:
                                            param.grad = projected_grad * z_tilde
                                        else:
                                            param.grad += projected_grad * z_tilde

                                # note that  | E_z [ <z, grad of one layer > ] | is equal to norm of grad for that layer for gaussian z
                                # leverages this fact to update the grad norms
                                if self.args.zo_variant == 'grad_norm' and self.args.norm_running_update:
                                    self.cs[layer] = torch.abs(projected_grad)
                    else:
                        # get number of zs to sample
                        num_zs = self.get_num_samples()
                        if num_zs > 1:
                            assert self.args.zero_order_use_trainer_optim, 'cannot sample multiple zs without storing intermediate gradient. use trainer.'

                        for _ in range(num_zs):
                            # prepare for sampling new zs
                            random_vector = None
                            if self.args.efficient_zero_order:
                                random_seed = np.random.randint(1000000000)

                            with torch.no_grad():
                                # first function evaluation
                                if self.args.efficient_zero_order:
                                    model = self.efficient_perturb_parameters(model, random_seed)
                                elif self.args.zo_variant is not None:
                                    model, random_vector = self.norm_perturb_parameters(model)
                                else:
                                    model, random_vector = self.perturb_parameters(model)
                                loss1 = self.zo_forward(model, inputs)

                                # second function evaluation
                                if self.args.efficient_zero_order:
                                    model = self.efficient_perturb_parameters(model, random_seed, scaling_factor=-2)
                                elif self.args.zo_variant is not None:
                                    model, random_vector = self.norm_perturb_parameters(model, random_vector, scaling_factor=-2)
                                else:
                                    model, random_vector = self.perturb_parameters(model, random_vector, scaling_factor=-2)
                                loss2 = self.zo_forward(model, inputs)

                            # Debugging: check for NaN in losses
                            if torch.isnan(loss1).item() or torch.isnan(loss2).item():
                                logger.warning("NaN encountered in loss during ZO forward step.")

                            # === Begin Adaptive h (Berahas et al.) ===
                            eps = self.adaptive_h if getattr(self.args, "use_adaptive_h", False) else self.args.zero_order_eps
                            # === Original Code ===
                            projected_grad = (loss1 - loss2) / (2 * eps)
                            # === Original Code ===

                            # Debugging: check for NaN or Inf in projected_grad
                            if torch.isnan(projected_grad).item() or torch.isinf(projected_grad).item():
                                logger.warning(f"projected_grad became invalid. loss1: {loss1.item()}, loss2: {loss2.item()}, eps: {eps}")
                            # === End Adaptive h ===

                            # scale grad according to accumulation
                            if self.args.gradient_accumulation_steps > 1:
                                assert self.args.zero_order_use_trainer_optim, 'grad accumulation not implemented for non-trainer ZO yet'
                                projected_grad = projected_grad / self.args.gradient_accumulation_steps

                            # scale grad according to number of zs sampled
                            if not self.args.scale_lr_with_samples:
                                projected_grad = projected_grad / float(num_zs)

                            # store gradient in parameter buffer if using trainer
                            # o/w, the loop will exit after one round and the update will be applied directly (see below)
                            if self.args.zero_order_use_trainer_optim:
                                if self.args.efficient_zero_order:
                                    # print(random_seed)
                                    torch.manual_seed(random_seed)

                                for name, param in self.named_parameters_to_optim:
                                    # recover noise used in perturbations
                                    if self.args.efficient_zero_order:
                                        z = torch.normal(mean=0, std=1, size=param.data.size(), device=param.data.device, dtype=param.data.dtype)
                                    else:
                                        z = random_vector[name]

                                    # === C-缩放开关：仅当 use_c_scale=True 时才按层放大 z ===
                                    # 关闭开关即不使用 C 缩放（与新方法一致）
                                    if getattr(self.args, "use_c_scale", False) and self.args.zo_variant is not None and not self.args.change_grad_estimate:
                                        cname = self.retrieve_c(name)
                                        if cname in self.cs:
                                            c_val = self.cs[cname]
                                            if isinstance(c_val, torch.Tensor):
                                                c_val = c_val.item() if c_val.numel()==1 else float(c_val.mean())
                                            else:
                                                c_val = float(c_val)
                                            if math.isfinite(c_val) and c_val != 0.0:
                                                z = z * c_val

                                    if param.grad is None:
                                        param.grad = projected_grad * z
                                    else:
                                        param.grad += projected_grad * z

                            # reset model back to its parameters at start of step
                            if self.args.efficient_zero_order:
                                model = self.efficient_perturb_parameters(model, random_seed)
                            elif self.args.zo_variant is not None:
                                model, random_vector = self.norm_perturb_parameters(model, random_vector)
                            else:
                                model, random_vector = self.perturb_parameters(model, random_vector)

                    # apply gradient updates
                    # if using trainer, follow trainer logic to clip grad and check if parameters should be updated
                    if self.args.zero_order_use_trainer_optim:
                        if (step + 1) % self.args.gradient_accumulation_steps == 0 or (
                            # last step in epoch but step is always smaller than gradient_accumulation_steps
                            len(epoch_iterator) <= self.args.gradient_accumulation_steps
                            and (step + 1) == len(epoch_iterator)
                        ):
                            # Gradient norm clipping
                            if self.args.zero_order_clip_grad:
                                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), self.args.max_grad_norm)

                            # Update the parameters and step scheduler
                            optimizer.step()
                            scheduler.step()

                            # logging
                            if (self.args.logging_steps > 0 and self.state.global_step % self.args.logging_steps == 0) or (
                                self.state.global_step == 1 and self.args.logging_first_step
                            ):
                                logs = {}
                                logs["loss"] = loss1.item()
                                if not self.args.zero_order_clip_grad:
                                    norm = 0.0
                                    for _, p in model.named_parameters():
                                        if p.grad is not None:
                                            norm += torch.sum(p.grad ** 2)
                                    norm = torch.sqrt(norm)
                                logs["grad_norm"] = norm.item()
                                logs["learning_rate"] = (
                                    scheduler.get_last_lr()[0]
                                    if version.parse(torch.__version__) >= version.parse("1.4")
                                    else scheduler.get_lr()[0]
                                )
                                logs["num_zs"] = num_zs
                                logs["global_step"] = self.state.global_step
                                logs["zo_forward_step"] = self.state.zo_forward_step
                                logs["max_steps"] = self.args.max_steps
                                logs["max_zo_forward_steps"] = self.args.max_zo_forward_steps
                                logs["time"] = int(time.time() - start_time)
                                # Log current eps value as float
                                logs["eps"] = eps if isinstance(eps, float) else eps.item()
                                self.log(logs)
                                logger.info(str(logs))
                                # === CSV：记录本 step 的训练度量，评估结果稍后补充 ===
                                train_acc_csv = self._compute_train_acc(model, inputs)
                                _csv_pending = {
                                    "epoch": float(self.epoch),
                                    "global_step": int(self.state.global_step),
                                    "train_loss": float(logs.get("loss", float("nan"))),
                                    "train_acc": (None if train_acc_csv is None else float(train_acc_csv)),
                                }

                            model.zero_grad()
                            self.state.global_step += 1
                            self.epoch = epoch + (step + 1) / len(epoch_iterator)
                    # if not using the trainer, the updates are resampled and directly applied to the parameters
                    else:
                        # Efficient mode
                        # WARNING: no gradient accumulation when not storing the grad
                        assert self.args.gradient_accumulation_steps == 1, 'gradient accumulation is not supported for zero-order optimization'
                        assert self.args.zero_order_sample_scheduler is None
                        assert not self.args.zero_order_clip_grad, 'gradient clipping not implemented yet for non-trainer ZO'

                        if self.args.efficient_zero_order:
                            torch.manual_seed(random_seed)
                        for name, param in self.named_parameters_to_optim:
                            if self.args.efficient_zero_order:
                                z = torch.normal(mean=0, std=1, size=param.data.size(), device=param.data.device, dtype=param.data.dtype)
                            else:
                                z = random_vector[name]
                            param.data = param.data - self.args.learning_rate * (projected_grad * z + self.args.weight_decay * param.data)

                        if (self.args.logging_steps > 0 and self.state.global_step % self.args.logging_steps == 0) or (
                                self.state.global_step == 1 and self.args.logging_first_step
                            ):
                                logs = {}
                                logs["loss"] = loss1.item()
                                logs["learning_rate"] = self.args.learning_rate
                                logs["global_step"] = self.state.global_step
                                logs["zo_forward_step"] = self.state.zo_forward_step
                                logs["max_steps"] = self.args.max_steps
                                logs["max_zo_forward_steps"] = self.args.max_zo_forward_steps
                                logs["time"] = int(time.time() - start_time)
                                # Log current eps value as float
                                logs["eps"] = eps if isinstance(eps, float) else eps.item()
                                self.log(logs)
                                logger.info(str(logs))
                                # === CSV：记录本 step 的训练度量，评估结果稍后补充 ===
                                train_acc_csv = self._compute_train_acc(model, inputs)
                                _csv_pending = {
                                    "epoch": float(self.epoch),
                                    "global_step": int(self.state.global_step),
                                    "train_loss": float(logs.get("loss", float("nan"))),
                                    "train_acc": (None if train_acc_csv is None else float(train_acc_csv)),
                                }


                        self.state.global_step += 1
                        self.epoch = epoch + (step + 1) / len(epoch_iterator)

                    # Debug information
                    # print("%.5f, %.5f" % (loss1.item(), loss2.item()))
                    # print("Loss: %.10f, projected_grad: %.5f" % (loss1, projected_grad))

                # standard, non-ZO optimization
                else:
                    tr_loss += self.training_step(model, inputs)

                    if (step + 1) % self.args.gradient_accumulation_steps == 0 or (
                        # last step in epoch but step is always smaller than gradient_accumulation_steps
                        len(epoch_iterator) <= self.args.gradient_accumulation_steps
                        and (step + 1) == len(epoch_iterator)
                    ):
                        if self.args.fp16 and _use_native_amp:
                            self.scaler.unscale_(optimizer)
                            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), self.args.max_grad_norm)
                        elif self.args.fp16:
                            norm = torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), self.args.max_grad_norm)
                        else:
                            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), self.args.max_grad_norm)

                        if self.args.optimizer_variant == 'signgd':
                            for n,p in model.named_parameters():
                                if p.grad is not None:
                                    p.grad = torch.sign(p.grad)

                        if transformers.is_torch_tpu_available():
                            xm.optimizer_step(optimizer)
                        elif self.args.fp16 and _use_native_amp:
                            self.scaler.step(optimizer)
                            self.scaler.update()
                        else:
                            optimizer.step()

                        scheduler.step()
                        model.zero_grad()
                        self.state.global_step += 1
                        self.epoch = epoch + (step + 1) / len(epoch_iterator)

                        if (self.args.logging_steps > 0 and self.state.global_step % self.args.logging_steps == 0) or (
                            self.state.global_step == 1 and self.args.logging_first_step
                        ):
                            logs = {}
                            tr_loss_scalar = tr_loss.item()
                            logs["loss"] = (tr_loss_scalar - logging_loss_scalar) / self.args.logging_steps
                            logs["norm"] = norm.item()
                            # backward compatibility for pytorch schedulers
                            logs["learning_rate"] = (
                                scheduler.get_last_lr()[0]
                                if version.parse(torch.__version__) >= version.parse("1.4")
                                else scheduler.get_lr()[0]
                            )
                            logging_loss_scalar = tr_loss_scalar

                            self.log(logs)
                            logger.info(str(logs))
                            # === CSV：记录本 step 的训练度量，评估结果稍后补充 ===
                            train_acc_csv = self._compute_train_acc(model, inputs)
                            _csv_pending = {
                                "epoch": float(self.epoch),
                                "global_step": int(self.state.global_step),
                                "train_loss": float(logs.get("loss", float("nan"))),
                                "train_acc": (None if train_acc_csv is None else float(train_acc_csv)),
                            }

                # === Begin Adaptive h: update h every update_noise_every steps ===
                if (
                    getattr(self.args, "use_adaptive_h", False)
                    and self.state.global_step > 0
                    and (self.state.global_step % update_noise_every == 0)
                ):
                    beta = float(getattr(self.args, "adaptive_h_ema_beta", 0.1))
                    nb = int(getattr(self.args, "adaptive_h_estimate_num_batches", 4))
                    buf_size = int(getattr(self.args, "adaptive_h_probe_buffer_size", 64))
                    nd = int(getattr(self.args, "adaptive_h_estimate_num_directions", 3))
                    reduce = getattr(self.args, "adaptive_h_estimate_reduce", "mean")
                    h_min = float(getattr(self.args, "adaptive_h_min", 1e-5))
                    h_max = float(getattr(self.args, "adaptive_h_max", 0.5))

                    inputs_list = self._get_h_estimation_inputs(train_dataloader, inputs, nb)
                    h_raw, eps_est, nu3_est = self.estimate_adaptive_h_multi(
                        model, self.compute_loss, inputs_list,
                        layer_name=None, num_directions=nd,
                        reduce=reduce, h_min=h_min, h_max=h_max
                    )
                    # If all directions were dropped (or estimate invalid), freeze h (do not update),
                    # but do NOT skip the rest of the training loop.
                    if (not math.isfinite(h_raw)) or (h_raw <= 0.0):
                        logger.warning(
                            f"[adaptive h][skip] step={self.state.global_step} h_raw invalid -> keep h_ema={float(self.adaptive_h):.3e} "
                            f"(nb={nb}, nd={nd}, reduce={reduce})"
                        )
                    else:
                        # Apply alpha (<1) to downweight truncation error: eps* scales by alpha^{-1/6}
                        alpha = float(getattr(self.args, "h_trunc_alpha", 1.0))
                        if math.isfinite(alpha) and alpha > 0.0:
                            h_raw = float(h_raw) * (alpha ** (-1.0 / 6.0))

                        h_sm = self._smooth_h_log_ema(previous_adaptive_h, h_raw, beta, h_min, h_max)
                        self.adaptive_h = float(h_sm)
                        previous_adaptive_h = float(h_sm)
                        self.epsilon_f = eps_est
                        self.nu3 = nu3_est
                        logger.info(
                            f"[adaptive h][update] step={self.state.global_step} "
                            f"h_raw={h_raw:.3e} -> h_ema={h_sm:.3e} "
                            f"(eps≈{eps_est:.2e}, nu3≈{nu3_est:.2e}, nb={nb}, buf={buf_size}, nd={nd}, reduce={reduce})"
                        )
                # === End Adaptive h update ===

                if self.args.max_steps > 0 and self.state.global_step > self.args.max_steps or (self.args.max_zo_forward_steps > 0 and self.state.zo_forward_step > self.args.max_zo_forward_steps):
                    epoch_iterator.close()
                    break

                if self.args.evaluate_during_training and self.state.global_step % self.args.eval_steps == 0:
                    output = self.evaluate()
                    metrics = output.metrics
                    objective = self.dev_objective(metrics)
                    # === CSV：本步触发了评估，把评估度量与训练度量一并写入 ===
                    try:
                        eval_loss = float(metrics.get("eval_loss", float("nan"))) if isinstance(metrics, dict) else float("nan")
                        eval_acc = self._extract_eval_acc(metrics)
                        with open(self._metrics_csv_path, "a", newline="") as f:
                            writer = csv.writer(f)
                            row = [
                                _csv_pending.get("epoch") if _csv_pending else float(self.epoch),
                                _csv_pending.get("global_step") if _csv_pending else int(self.state.global_step),
                                _csv_pending.get("train_loss") if _csv_pending else float("nan"),
                                _csv_pending.get("train_acc") if _csv_pending else None,
                                "YES",
                                eval_loss,
                                (None if eval_acc is None else float(eval_acc)),
                            ]
                            writer.writerow(row)
                    except Exception as e:
                        logger.warning(f"[CSV] failed to write eval row: {e}")
                    if objective > self.objective:
                        logger.info("Best dev result: {}".format(objective))
                        self.objective = objective
                        # self.save_model(self.args.output_dir)

                        # Now we save this to (CPU) memory instead of disk <-- much faster
                        self.best_model_ckpt = {k: v.detach().cpu() for k, v in model.state_dict().items()}
                else:
                    # === CSV：本步未触发评估，立即写入一行并标注未评估 ===
                    try:
                        with open(self._metrics_csv_path, "a", newline="") as f:
                            writer = csv.writer(f)
                            row = [
                                _csv_pending.get("epoch") if _csv_pending else float(self.epoch),
                                _csv_pending.get("global_step") if _csv_pending else int(self.state.global_step),
                                _csv_pending.get("train_loss") if _csv_pending else float("nan"),
                                _csv_pending.get("train_acc") if _csv_pending else None,
                                "NO",
                                None,
                                None,
                            ]
                            writer.writerow(row)
                    except Exception as e:
                        logger.warning(f"[CSV] failed to write non-eval row: {e}")

            if self.args.max_steps > 0 and self.state.global_step > self.args.max_steps or (self.args.max_zo_forward_steps > 0 and self.state.zo_forward_step > self.args.max_zo_forward_steps):
                # train_iterator.close()
                break
            if self.args.tpu_metrics_debug or self.args.debug:
                # tpu-comment: Logging debug metrics for PyTorch/XLA (compile, execute times, ops, etc.)
                xm.master_print(met.metrics_report())

        if self.args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of training
            delattr(self, "_past")

        logger.info("\n\nTraining completed. Do not forget to share your model on huggingface.co/models =)\n\n")
        return TrainOutput(self.state.global_step, tr_loss / self.state.global_step, metrics), self.objective


    """
    Difference compared to original implementation: return output instead of output.metrics (so there is also the logits)
    """
    def evaluate(self, eval_dataset: Optional[Dataset] = None) -> Dict[str, float]:
        """
        Run evaluation and returns metrics.

        The calling script will be responsible for providing a method to compute metrics, as they are
        task-dependent (pass it to the init :obj:`compute_metrics` argument).

        You can also subclass and override this method to inject custom behavior.

        Args:
            eval_dataset (:obj:`Dataset`, `optional`):
                Pass a dataset if you wish to override :obj:`self.eval_dataset`. If it is an :obj:`datasets.Dataset`,
                columns not accepted by the ``model.forward()`` method are automatically removed. It must implement
                the :obj:`__len__` method.

        Returns:
            A dictionary containing the evaluation loss and the potential metrics computed from the predictions.
        """
        if eval_dataset is not None and not isinstance(eval_dataset, collections.abc.Sized):
            raise ValueError("eval_dataset must implement __len__")

        eval_dataloader = self.get_eval_dataloader(eval_dataset)

        output = self.prediction_loop(eval_dataloader, description="Evaluation")

        self.log(output.metrics)
        logger.info(output.metrics)

        if self.args.tpu_metrics_debug or self.args.debug:
            # tpu-comment: Logging debug metrics for PyTorch/XLA (compile, execute times, ops, etc.)
            xm.master_print(met.metrics_report())

        return output
