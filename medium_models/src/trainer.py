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
from contextlib import nullcontext
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
from src.quzo import make_quzo_direction_pair, quantize_tensor
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
        filename = f"metrics_adaptiveH-{use_ah}_cscale-{use_cs}.csv"
        self._metrics_csv_path = os.path.join(log_dir, filename)
        # 若文件不存在则写入表头
        if not os.path.exists(self._metrics_csv_path):
            with open(self._metrics_csv_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["epoch", "global_step", "train_loss", "train_acc", "eval_ran", "eval_loss", "eval_acc", "eval_loss_avg5"])
        # Track eval loss history for tail-average reporting
        self._eval_loss_history = []

    def _setup_h_estimation_csv(self):
        self._h_estimation_csv_path = None
        self._h_estimation_csv_fields = [
            "global_step",
            "training_h_source",
            "training_h",
            "h_additive",
            "h_two_point",
            "h_two_point_tilde",
            "eps_est",
            "nu3_est",
            "noise_est",
            "delta_tilde",
            "g_tilde",
            "l_tilde",
            "delta_hat",
            "g_hat",
            "l_hat",
            "h2",
            "two_point_precision",
            "additive_noise_precision",
        ]
        if not bool(getattr(self.args, "two_point_h_log_csv", True)):
            return
        if not (self._should_compute_additive_h() or self._should_compute_two_point_h()):
            return
        base_dir = getattr(self.args, "output_dir", "./outputs") or "./outputs"
        os.makedirs(base_dir, exist_ok=True)
        self._h_estimation_csv_path = os.path.join(base_dir, "h_estimation.csv")
        if not os.path.exists(self._h_estimation_csv_path):
            with open(self._h_estimation_csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=self._h_estimation_csv_fields)
                writer.writeheader()

    def _append_h_estimation_row(self, row: Dict[str, Any]):
        path = getattr(self, "_h_estimation_csv_path", None)
        if not path:
            return
        try:
            with open(path, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=self._h_estimation_csv_fields)
                payload = {k: row.get(k) for k in self._h_estimation_csv_fields}
                writer.writerow(payload)
        except Exception as e:
            logger.warning(f"[h_estimation] failed to append CSV row: {type(e).__name__}: {e}")

    def _grad_norm_log_enabled(self) -> bool:
        v = str(os.environ.get("GRAD_NORM_LOG", "0")).strip().lower()
        return v in ("1", "true", "yes", "y", "on")

    def _setup_grad_norm_csv(self):
        base_dir = getattr(self.args, "output_dir", "./outputs") or "./outputs"
        log_dir = os.path.join(base_dir, "metrics_logs")
        os.makedirs(log_dir, exist_ok=True)
        self._grad_norm_csv_path = os.path.join(log_dir, "grad_norms.csv")
        try:
            self._grad_norm_log_every = max(1, int(os.environ.get("GRAD_NORM_LOG_EVERY", "1")))
        except Exception:
            self._grad_norm_log_every = 1
        if not os.path.exists(self._grad_norm_csv_path):
            with open(self._grad_norm_csv_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["epoch", "global_step", "grad_l1_norm", "grad_l2_norm", "source"])

    def _compute_grad_l1_l2_from_param_grads(self, model: nn.Module) -> Tuple[float, float]:
        l1_sum = 0.0
        l2_sq = 0.0
        for _, p in model.named_parameters():
            if p.grad is None:
                continue
            g = p.grad.detach().float()
            l1_sum += float(torch.sum(torch.abs(g)).item())
            l2_sq += float(torch.sum(g * g).item())
        return float(l1_sum), float(math.sqrt(max(l2_sq, 0.0)))

    def _write_grad_norm_row(self, epoch_val: float, global_step: int, grad_l1: Optional[float], grad_l2: Optional[float], source: str):
        path = getattr(self, "_grad_norm_csv_path", None)
        if (not path) or grad_l1 is None or grad_l2 is None:
            return
        every = int(getattr(self, "_grad_norm_log_every", 1))
        if every > 1 and int(global_step) % every != 0:
            return
        try:
            with open(path, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([float(epoch_val), int(global_step), float(grad_l1), float(grad_l2), str(source)])
        except Exception as e:
            logger.warning(f"[grad_norm] failed to append CSV row: {e}")

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
            h0 = getattr(self.args, "zero_order_eps", 1e-3)
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

    def _should_compute_additive_h(self) -> bool:
        return bool(getattr(self.args, "use_adaptive_h", False) or getattr(self.args, "enable_additive_h_estimation", False))

    def _should_compute_two_point_h(self) -> bool:
        return bool(getattr(self.args, "enable_two_point_h_estimation", False))

    def _get_active_h_source(self) -> str:
        src = str(getattr(self.args, "h_estimation_active_source", "fixed")).lower()
        if src == "auto":
            if self._should_compute_additive_h():
                return "additive"
            return "fixed"
        return src

    def _get_current_additive_h(self) -> float:
        val = getattr(self, "additive_h", getattr(self, "adaptive_h", self._get_init_h()))
        try:
            val = float(val)
        except Exception:
            val = self._get_init_h()
        if (not math.isfinite(val)) or val <= 0.0:
            val = self._get_init_h()
        return float(val)

    def _get_current_two_point_h(self) -> float:
        val = getattr(self, "two_point_h", self._get_init_h())
        try:
            val = float(val)
        except Exception:
            val = self._get_init_h()
        if (not math.isfinite(val)) or val <= 0.0:
            val = self._get_init_h()
        return float(val)

    def _get_training_step_size(self) -> float:
        src = self._get_active_h_source()
        if src == "additive" and self._should_compute_additive_h():
            return self._get_current_additive_h()
        if src == "two_point" and self._should_compute_two_point_h():
            return self._get_current_two_point_h()
        return float(getattr(self.args, "zero_order_eps", 1e-3))

    def _should_quantize_training_perturbation(self) -> bool:
        return self._zo_use_quzo() or self._get_active_h_source() == "two_point"

    def _quantize_delta_tensor(self, delta: torch.Tensor, target_dtype: Optional[torch.dtype] = None) -> torch.Tensor:
        if self._zo_use_quzo():
            return quantize_tensor(
                delta,
                self._zo_quant_bits(),
                target_dtype=(delta.dtype if target_dtype is None else target_dtype),
            )
        precision = str(getattr(self.args, "zo_two_point_precision", "fp32")).lower()
        if target_dtype is None:
            target_dtype = delta.dtype
        if precision != "fp16":
            return delta.to(dtype=target_dtype)
        return delta.detach().to(dtype=torch.float16).to(dtype=target_dtype)

    def _copy_probe_inputs(self, inputs):
        if not isinstance(inputs, dict):
            return inputs
        copied = {}
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                copied[k] = v.detach().cpu()
            else:
                copied[k] = v
        return copied

    def _get_two_point_probe_batch(self, current_inputs):
        if bool(getattr(self.args, "two_point_h_fixed_probe_batch", True)):
            cached = getattr(self, "_two_point_fixed_probe_batch", None)
            if cached is None:
                cached = self._copy_probe_inputs(current_inputs)
                self._two_point_fixed_probe_batch = cached
            return dict(cached) if isinstance(cached, dict) else cached
        return dict(current_inputs) if isinstance(current_inputs, dict) else current_inputs

    def _build_named_parameters_to_optim(self, model: nn.Module) -> List[Tuple[str, nn.Parameter]]:
        named_params = []
        for name, param in model.named_parameters():
            if self.should_optim(name, param):
                named_params.append((name, param))
        self.named_parameters_to_optim = named_params
        return named_params

    def _init_h_estimation_state(self):
        h0 = self._get_init_h()
        add_min = float(getattr(self.args, "adaptive_h_min", 1e-5))
        add_max = float(getattr(self.args, "adaptive_h_max", 0.5))
        tp_min = float(getattr(self.args, "two_point_h_min", 1e-5))
        tp_max = float(getattr(self.args, "two_point_h_max", 0.5))
        self.additive_h = float(min(add_max, max(add_min, h0)))
        self.adaptive_h = float(self.additive_h)
        self._additive_prev_h = float(self.additive_h)
        self.two_point_h = float(min(tp_max, max(tp_min, h0)))
        self.two_point_h_tilde = float("nan")
        self._two_point_prev_h = float(self.two_point_h)
        self.epsilon_f = float("nan")
        self.nu3 = float("nan")
        self._additive_last_stats = {
            "h_additive": float(self.additive_h),
            "eps_est": float("nan"),
            "nu3_est": float("nan"),
            "noise_est": float("nan"),
        }
        self._two_point_delta_buf = collections.deque(maxlen=max(1, int(getattr(self.args, "two_point_h_window_delta", 3))))
        self._two_point_g_buf = collections.deque(maxlen=max(1, int(getattr(self.args, "two_point_h_window_g", 5))))
        self._two_point_l_buf = collections.deque(maxlen=max(1, int(getattr(self.args, "two_point_h_window_l", 5))))
        self._two_point_last_stats = {
            "h_two_point": float(self.two_point_h),
            "h_two_point_tilde": float("nan"),
            "delta_tilde": float("nan"),
            "g_tilde": float("nan"),
            "l_tilde": float("nan"),
            "delta_hat": float("nan"),
            "g_hat": float("nan"),
            "l_hat": float("nan"),
            "h2": float("nan"),
        }

    @staticmethod
    def _mean_or_none(xs: List[float]) -> Optional[float]:
        vals = [float(x) for x in xs if math.isfinite(float(x))]
        if len(vals) == 0:
            return None
        return float(sum(vals) / len(vals))

    @staticmethod
    def _median_or_none(xs: List[float]) -> Optional[float]:
        vals = sorted(float(x) for x in xs if math.isfinite(float(x)))
        if len(vals) == 0:
            return None
        n = len(vals)
        if n % 2 == 1:
            return float(vals[n // 2])
        return float(0.5 * (vals[n // 2 - 1] + vals[n // 2]))

    def _current_two_point_delta_hat(self) -> Optional[float]:
        return self._median_or_none(list(getattr(self, "_two_point_delta_buf", [])))

    def _current_two_point_g_hat(self) -> Optional[float]:
        return self._mean_or_none(list(getattr(self, "_two_point_g_buf", [])))

    def _current_two_point_l_hat(self) -> Optional[float]:
        return self._median_or_none(list(getattr(self, "_two_point_l_buf", [])))

    def _update_two_point_buffers(self, delta_tilde=None, g_tilde=None, l_tilde=None):
        if delta_tilde is not None and math.isfinite(float(delta_tilde)) and float(delta_tilde) > 0.0:
            self._two_point_delta_buf.append(float(delta_tilde))
        if g_tilde is not None and math.isfinite(float(g_tilde)) and float(g_tilde) > 0.0:
            self._two_point_g_buf.append(float(g_tilde))
        if l_tilde is not None and math.isfinite(float(l_tilde)) and float(l_tilde) > 0.0:
            self._two_point_l_buf.append(float(l_tilde))

    def _sample_direction_and_delta(self, named_params: List[Tuple[str, nn.Parameter]], h: float) -> Tuple[List[torch.Tensor], float]:
        delta_list = []
        norm_sq = 0.0
        for _, param in named_params:
            z = torch.normal(mean=0, std=1, size=param.data.size(), device=param.data.device, dtype=param.data.dtype)
            norm_sq += float(torch.sum(z.detach().float() * z.detach().float()).item())
            delta_list.append(self._quantize_delta_tensor(z * float(h), target_dtype=param.data.dtype))
        return delta_list, float(norm_sq)

    def _apply_delta_list(self, named_params: List[Tuple[str, nn.Parameter]], delta_list: List[torch.Tensor], multiplier: float):
        with torch.no_grad():
            for (_, param), delta in zip(named_params, delta_list):
                param.data.add_(float(multiplier) * delta)

    def _estimate_delta_rms_sampled(self, model: nn.Module) -> Optional[float]:
        named_params = self._build_named_parameters_to_optim(model)
        if len(named_params) == 0:
            return None
        total_numel = int(sum(param.data.numel() for _, param in named_params))
        if total_numel <= 0:
            return None
        sample_size = max(1, min(int(getattr(self.args, "two_point_h_delta_sample_size", 4096)), total_numel))
        cums = np.cumsum([int(param.data.numel()) for _, param in named_params])
        picks = np.random.randint(0, total_numel, size=sample_size)
        vals = []
        for flat_idx in picks:
            param_idx = int(np.searchsorted(cums, int(flat_idx), side="right"))
            prev = 0 if param_idx == 0 else int(cums[param_idx - 1])
            local_idx = int(flat_idx) - prev
            tensor = named_params[param_idx][1].data.detach().view(-1)[local_idx].float().cpu()
            vals.append(float(tensor.item()))
        if len(vals) == 0:
            return None
        sample = torch.tensor(vals, dtype=torch.float32)
        sample_low = sample.to(dtype=torch.float16)
        sample_next = torch.nextafter(sample_low, torch.full_like(sample_low, float("inf")))
        delta_i = (sample_next - sample_low).abs().to(dtype=torch.float32)
        delta_rms = torch.sqrt(torch.mean(delta_i * delta_i))
        val = float(delta_rms.item())
        if (not math.isfinite(val)) or val <= 0.0:
            return None
        return val

    def _estimate_two_point_g_raw(self, model: nn.Module, probe_inputs, h: float) -> Optional[float]:
        if (not math.isfinite(float(h))) or float(h) <= 0.0:
            return None
        named_params = self._build_named_parameters_to_optim(model)
        if len(named_params) == 0:
            return None
        vals = []
        for _ in range(max(1, int(getattr(self.args, "two_point_h_num_directions_g", 4)))):
            delta_list, _ = self._sample_direction_and_delta(named_params, float(h))
            try:
                self._apply_delta_list(named_params, delta_list, +1.0)
                loss_plus = float(self._zo_two_point_forward(model, probe_inputs).item())
            finally:
                self._apply_delta_list(named_params, delta_list, -1.0)
            try:
                self._apply_delta_list(named_params, delta_list, -1.0)
                loss_minus = float(self._zo_two_point_forward(model, probe_inputs).item())
            finally:
                self._apply_delta_list(named_params, delta_list, +1.0)
            d_hat = (loss_plus - loss_minus) / (2.0 * float(h))
            if math.isfinite(d_hat):
                vals.append(abs(float(d_hat)))
        if len(vals) == 0:
            return None
        return float(math.sqrt(math.pi / 2.0) * (sum(vals) / len(vals)))

    def _estimate_two_point_l_raw(self, model: nn.Module, probe_inputs, delta_hat: float) -> Tuple[Optional[float], Optional[float]]:
        if (not math.isfinite(float(delta_hat))) or float(delta_hat) <= 0.0:
            return None, None
        named_params = self._build_named_parameters_to_optim(model)
        if len(named_params) == 0:
            return None, None
        c2 = float(getattr(self.args, "two_point_h_c2", 1.0))
        eps_num = float(getattr(self.args, "two_point_h_eps_num", 1e-12))
        q_l = min(1.0, max(0.0, float(getattr(self.args, "two_point_h_q_l", 0.5))))
        h2 = float(max(float(delta_hat), c2 * math.sqrt(float(delta_hat))))
        base_loss = float(self._zo_two_point_forward(model, probe_inputs).item())
        if not math.isfinite(base_loss):
            return None, h2
        lambdas = []
        for _ in range(max(1, int(getattr(self.args, "two_point_h_num_directions_l", 4)))):
            delta_list, norm_sq = self._sample_direction_and_delta(named_params, h2)
            try:
                self._apply_delta_list(named_params, delta_list, +1.0)
                loss1 = float(self._zo_two_point_forward(model, probe_inputs).item())
                self._apply_delta_list(named_params, delta_list, +1.0)
                loss2 = float(self._zo_two_point_forward(model, probe_inputs).item())
            finally:
                self._apply_delta_list(named_params, delta_list, -2.0)
            k_hat = (loss2 - 2.0 * loss1 + base_loss) / max(h2 ** 2, 1e-30)
            lam = abs(float(k_hat)) / (float(norm_sq) + eps_num)
            if math.isfinite(lam):
                lambdas.append(float(lam))
        if len(lambdas) == 0:
            return None, h2
        lambdas_arr = np.asarray(lambdas, dtype=np.float64)
        return float(np.quantile(lambdas_arr, q_l)), h2

    def _refresh_additive_h_estimation(self, model, train_dataloader, current_inputs):
        if not self._should_compute_additive_h():
            return None
        nb = int(getattr(self.args, "adaptive_h_estimate_num_batches", 4))
        nd = int(getattr(self.args, "adaptive_h_estimate_num_directions", 3))
        reduce = getattr(self.args, "adaptive_h_estimate_reduce", "mean")
        h_min = float(getattr(self.args, "adaptive_h_min", 1e-5))
        h_max = float(getattr(self.args, "adaptive_h_max", 0.5))
        beta = float(getattr(self.args, "adaptive_h_ema_beta", 0.1))
        inputs_list = self._get_h_estimation_inputs(train_dataloader, current_inputs, nb)
        h_raw, eps_est, nu3_est = self.estimate_adaptive_h_multi(
            model, self.compute_loss, inputs_list,
            layer_name=None, num_directions=nd,
            reduce=reduce, h_min=h_min, h_max=h_max
        )
        if (not math.isfinite(h_raw)) or (h_raw <= 0.0):
            logger.warning(
                f"[additive error estimation][skip] step={self.state.global_step} "
                f"h_raw invalid -> keep h={self._get_current_additive_h():.3e}"
            )
            stats = {
                "h_additive": self._get_current_additive_h(),
                "eps_est": float(eps_est) if math.isfinite(float(eps_est)) else float("nan"),
                "nu3_est": float(nu3_est) if math.isfinite(float(nu3_est)) else float("nan"),
                "noise_est": float(eps_est) if math.isfinite(float(eps_est)) else float("nan"),
            }
            self._additive_last_stats = stats
            return stats
        alpha = float(getattr(self.args, "h_trunc_alpha", 1.0))
        if math.isfinite(alpha) and alpha > 0.0:
            h_raw = float(h_raw) * (alpha ** (-1.0 / 6.0))
        h_sm = self._smooth_h_log_ema(self._additive_prev_h, h_raw, beta, h_min, h_max)
        self.additive_h = float(h_sm)
        self.adaptive_h = float(h_sm)
        self._additive_prev_h = float(h_sm)
        self.epsilon_f = float(eps_est)
        self.nu3 = float(nu3_est)
        stats = {
            "h_additive": float(h_sm),
            "eps_est": float(eps_est),
            "nu3_est": float(nu3_est),
            "noise_est": float(eps_est),
        }
        self._additive_last_stats = stats
        logger.info(
            f"[additive error estimation][update] step={self.state.global_step} "
            f"h_raw={float(h_raw):.3e} -> h={float(h_sm):.3e} "
            f"(eps≈{float(eps_est):.2e}, nu3≈{float(nu3_est):.2e}, nb={nb}, nd={nd}, reduce={reduce})"
        )
        return stats

    def _refresh_two_point_h_estimation(self, model, current_inputs):
        if not self._should_compute_two_point_h():
            return None
        probe_inputs = self._get_two_point_probe_batch(current_inputs)
        delta_tilde = self._estimate_delta_rms_sampled(model)
        delta_for_l = self._current_two_point_delta_hat()
        if delta_for_l is None:
            delta_for_l = delta_tilde
        g_tilde = self._estimate_two_point_g_raw(model, probe_inputs, self._get_current_two_point_h())
        l_tilde, h2 = self._estimate_two_point_l_raw(model, probe_inputs, delta_for_l) if delta_for_l is not None else (None, None)
        self._update_two_point_buffers(delta_tilde=delta_tilde, g_tilde=g_tilde, l_tilde=l_tilde)
        delta_hat = self._current_two_point_delta_hat()
        g_hat = self._current_two_point_g_hat()
        l_hat = self._current_two_point_l_hat()
        h_tilde = None
        if (
            delta_hat is not None and g_hat is not None and l_hat is not None
            and delta_hat > 0.0 and g_hat > 0.0 and l_hat > 0.0
        ):
            named_params = self._build_named_parameters_to_optim(model)
            d_dim = max(1, int(sum(param.data.numel() for _, param in named_params)))
            h_tilde = (
                (float(delta_hat) ** 2 * float(g_hat) ** 2)
                / (16.0 * (float(l_hat) ** 2) * float(d_dim) * float(d_dim + 2))
            ) ** 0.25
            h_min = float(getattr(self.args, "two_point_h_min", 1e-5))
            h_max = float(getattr(self.args, "two_point_h_max", 0.5))
            beta = float(getattr(self.args, "two_point_h_beta", 0.5))
            h_tilde = float(min(h_max, max(h_min, h_tilde)))
            self.two_point_h = float(self._smooth_h_log_ema(self._two_point_prev_h, h_tilde, beta, h_min, h_max))
            self._two_point_prev_h = float(self.two_point_h)
            self.two_point_h_tilde = float(h_tilde)
            logger.info(
                f"[two-point simple estimation][update] step={self.state.global_step} "
                f"h_tilde={float(h_tilde):.3e} -> h={float(self.two_point_h):.3e} "
                f"(Delta≈{float(delta_hat):.2e}, G≈{float(g_hat):.2e}, L≈{float(l_hat):.2e}, h2={float(h2) if h2 is not None else float('nan'):.2e})"
            )
        else:
            logger.warning(
                f"[two-point simple estimation][skip] step={self.state.global_step} "
                f"invalid state -> keep h={self._get_current_two_point_h():.3e}"
            )
        stats = {
            "h_two_point": self._get_current_two_point_h(),
            "h_two_point_tilde": float(h_tilde) if h_tilde is not None and math.isfinite(float(h_tilde)) else float("nan"),
            "delta_tilde": float(delta_tilde) if delta_tilde is not None and math.isfinite(float(delta_tilde)) else float("nan"),
            "g_tilde": float(g_tilde) if g_tilde is not None and math.isfinite(float(g_tilde)) else float("nan"),
            "l_tilde": float(l_tilde) if l_tilde is not None and math.isfinite(float(l_tilde)) else float("nan"),
            "delta_hat": float(delta_hat) if delta_hat is not None and math.isfinite(float(delta_hat)) else float("nan"),
            "g_hat": float(g_hat) if g_hat is not None and math.isfinite(float(g_hat)) else float("nan"),
            "l_hat": float(l_hat) if l_hat is not None and math.isfinite(float(l_hat)) else float("nan"),
            "h2": float(h2) if h2 is not None and math.isfinite(float(h2)) else float("nan"),
        }
        self._two_point_last_stats = stats
        return stats

    def _log_joint_h_estimation_step(self):
        if not (self._should_compute_additive_h() or self._should_compute_two_point_h()):
            return
        additive = dict(getattr(self, "_additive_last_stats", {}) or {})
        two_point = dict(getattr(self, "_two_point_last_stats", {}) or {})
        row = {
            "global_step": int(getattr(self.state, "global_step", 0)),
            "training_h_source": self._get_active_h_source(),
            "training_h": float(self._get_training_step_size()),
            "h_additive": additive.get("h_additive"),
            "h_two_point": two_point.get("h_two_point"),
            "h_two_point_tilde": two_point.get("h_two_point_tilde"),
            "eps_est": additive.get("eps_est"),
            "nu3_est": additive.get("nu3_est"),
            "noise_est": additive.get("noise_est"),
            "delta_tilde": two_point.get("delta_tilde"),
            "g_tilde": two_point.get("g_tilde"),
            "l_tilde": two_point.get("l_tilde"),
            "delta_hat": two_point.get("delta_hat"),
            "g_hat": two_point.get("g_hat"),
            "l_hat": two_point.get("l_hat"),
            "h2": two_point.get("h2"),
            "two_point_precision": str(getattr(self.args, "zo_two_point_precision", "fp32")).lower(),
            "additive_noise_precision": str(getattr(self.args, "zo_two_point_precision", "fp32")).lower(),
        }
        self._append_h_estimation_row(row)

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
    # h-probe (new)
    # Enable via environment variables (no need to touch run.py args parser):
    #   H_PROBE=1
    #   H_PROBE_EVERY=500              (0 => only at global_step==0)
    #   H_PROBE_MIN=1e-8, H_PROBE_MAX=1e-2
    #   H_PROBE_HLIST=...              (optional, comma-separated overrides)
    #   H_PROBE_NDIR=8                 (M directions per probe)
    #   H_PROBE_NB=1                   (1 => eval batch = probe batch; >=2 => use a separate fixed eval batch)
    #   H_PROBE_UNIT_U=1               (normalize direction to unit norm across all params)
    #   H_PROBE_SAVE_PARAMS=1          (best-effort snapshot/restore theta_t; fallback to exact undo)
    # Output: <output_dir>/hprobe.jsonl
    # =============================

    def _hprobe_enabled(self) -> bool:
        v = os.environ.get("H_PROBE", "0").lower()
        return v in ("1", "true", "yes", "y", "on")

    def _hprobe_bool_env(self, key: str, default: str = "0") -> bool:
        v = os.environ.get(key, default)
        if isinstance(v, str):
            return v.lower() in ("1", "true", "yes", "y", "on")
        return bool(v)

    def _hprobe_cfg(self) -> Dict[str, Any]:
        every = int(os.environ.get("H_PROBE_EVERY", "0"))
        num_h = int(os.environ.get("H_PROBE_NUM", "9"))  # kept for backward compat; ignored when H_PROBE_HLIST is empty
        h_min = float(os.environ.get("H_PROBE_MIN", "1e-8"))
        h_max = float(os.environ.get("H_PROBE_MAX", "1e-2"))
        ndir = int(os.environ.get("H_PROBE_NDIR", "8"))
        nbatch = int(os.environ.get("H_PROBE_NB", "1"))
        out_name = os.environ.get("H_PROBE_OUT", "hprobe.jsonl")
        h_list = os.environ.get("H_PROBE_HLIST", "")
        unit_u = self._hprobe_bool_env("H_PROBE_UNIT_U", "1")
        save_params = self._hprobe_bool_env("H_PROBE_SAVE_PARAMS", "1")
        log_each_h = self._hprobe_bool_env("H_PROBE_LOG_EACH_H", "1")
        return {
            "every": every,
            "num_h": max(num_h, 2),
            "h_min": h_min,
            "h_max": h_max,
            "ndir": max(ndir, 1),
            "nbatch": max(nbatch, 1),
            "out_name": out_name,
            "h_list": h_list,
            "unit_u": unit_u,
            "save_params": save_params,
            "log_each_h": log_each_h,
        }

    def _hprobe_h_list(self, cfg: Dict[str, Any]) -> List[float]:
        # Optional explicit override: comma/space-separated list
        h_list = str(cfg.get("h_list", "") or "").strip()
        if h_list:
            hs = []
            for tok in re.split(r"[\s,]+", h_list):
                if not tok:
                    continue
                try:
                    hs.append(float(tok))
                except Exception:
                    pass
            hs = sorted(set([float(x) for x in hs if math.isfinite(x) and x > 0]))
            return hs

        # Default: log-scale set with multipliers {1,3} per decade: 1e-8,3e-8,1e-7,3e-7,...,1e-2
        h_min, h_max = float(cfg["h_min"]), float(cfg["h_max"])
        if (not math.isfinite(h_min)) or (not math.isfinite(h_max)) or h_min <= 0 or h_max <= 0:
            h_min, h_max = 1e-8, 1e-2
        if h_min > h_max:
            h_min, h_max = h_max, h_min

        emin = int(math.floor(math.log10(h_min)))
        emax = int(math.ceil(math.log10(h_max)))
        hs = []
        for e in range(emin, emax + 1):
            for m in (1.0, 3.0):
                h = m * (10.0 ** e)
                if h < h_min or h > h_max:
                    continue
                hs.append(float(h))
        hs = sorted(set(hs))
        if len(hs) == 0:
            hs = [float(h_min), float(h_max)]
        return hs

    def _hprobe_use_wd(self, name: str) -> bool:
        n = name.lower()
        return ("bias" not in n) and ("layer_norm" not in n) and ("layernorm" not in n)

    def _hprobe_get_inv_norm(self, model: nn.Module, seed: int) -> float:
        """Compute 1/||u|| for the direction generated by seed (across all params). Cached within one probe."""
        cache = getattr(self, "_hprobe_inv_norm_cache", None)
        if isinstance(cache, dict) and (seed in cache):
            return float(cache[seed])

        # Ensure parameter list exists
        if (not hasattr(self, "named_parameters_to_optim")) or (self.named_parameters_to_optim is None) or (len(self.named_parameters_to_optim) == 0):
            self.named_parameters_to_optim = [(n, p) for n, p in model.named_parameters() if p.requires_grad]

        torch.manual_seed(seed)
        ss = 0.0
        with torch.no_grad():
            for _, p in self.named_parameters_to_optim:
                z = torch.normal(mean=0, std=1, size=p.data.size(), device=p.data.device, dtype=p.data.dtype)
                ss += float(torch.sum(z.float() * z.float()).item())
        inv = 1.0 / math.sqrt(max(ss, 1e-30))
        if not isinstance(cache, dict):
            cache = {}
        cache[seed] = float(inv)
        self._hprobe_inv_norm_cache = cache
        return float(inv)

    def _hprobe_true_grad(self, model: nn.Module, inputs: Dict[str, Any]) -> Tuple[float, List[Optional[torch.Tensor]]]:
        """Compute loss and true gradient on a fixed probe batch (one backward via autograd.grad)."""
        # Ensure parameter list exists
        if (not hasattr(self, "named_parameters_to_optim")) or (self.named_parameters_to_optim is None) or (len(self.named_parameters_to_optim) == 0):
            self.named_parameters_to_optim = [(n, p) for n, p in model.named_parameters() if p.requires_grad]

        was_training = bool(model.training)
        model.eval()
        inputs = self._prepare_inputs(inputs)
        with self.compute_loss_context_manager():
            loss = self.compute_loss(model, inputs)
        if self.args.n_gpu > 1:
            loss = loss.mean()
        params = [p for _, p in self.named_parameters_to_optim]
        grads = torch.autograd.grad(loss, params, retain_graph=False, create_graph=False, allow_unused=True)
        grads = [g.detach() if g is not None else None for g in grads]
        loss_val = float(loss.detach().item())
        if was_training:
            model.train()
        return loss_val, grads

    def _hprobe_dot_grad_direction(
        self,
        model: nn.Module,
        grads: List[Optional[torch.Tensor]],
        seed: int,
    ) -> float:
        """Compute d_true = g_true \cdot u(seed)."""
        # Ensure parameter list exists
        if (not hasattr(self, "named_parameters_to_optim")) or (self.named_parameters_to_optim is None) or (len(self.named_parameters_to_optim) == 0):
            self.named_parameters_to_optim = [(n, p) for n, p in model.named_parameters() if p.requires_grad]

        inv_norm = 1.0  # raw-z probe: do not normalize

        torch.manual_seed(seed)
        dot = 0.0
        with torch.no_grad():
            for (_, p), g in zip(self.named_parameters_to_optim, grads):
                z = torch.normal(mean=0, std=1, size=p.data.size(), device=p.data.device, dtype=p.data.dtype)
                # if inv_norm != 1.0:
                #     z = z * inv_norm
                if g is None:
                    continue
                dot += float(torch.sum(g.float() * z.float()).item())
        return float(dot)

    def _hprobe_eval_at(self, model: nn.Module, inputs: Dict[str, Any], h: float, seed: int, mult: float, collect_stats: bool = False) -> Tuple[float, Optional[Dict[str, Any]]]:
        """Evaluate f(theta + mult*h*z) and restore params, using the same z via seed.

        If collect_stats=True and a param snapshot is available (H_PROBE_SAVE_PARAMS=1), additionally return
        quantization/perturbation diagnostics:
          - h_eff = ||theta_perturbed - theta|| / ||u||
          - r     = #(theta_i^+ != theta_i) / N
        """
        # Ensure list exists
        if (not hasattr(self, "named_parameters_to_optim")) or (self.named_parameters_to_optim is None) or (len(self.named_parameters_to_optim) == 0):
            self.named_parameters_to_optim = [(n, p) for n, p in model.named_parameters() if p.requires_grad]

        inv_norm = 1.0  # raw-z probe: do not normalize

        # Optional: perturbation/quantization stats (requires param snapshot)
        base_params = getattr(self, "_hprobe_param_backup", None)
        if not (isinstance(base_params, list) and len(base_params) == len(self.named_parameters_to_optim)):
            base_params = None

        u_ss = 0.0
        diff_ss = 0.0
        changed = 0
        numel = 0

        torch.manual_seed(seed)
        with torch.no_grad():
            for idx, (_, p) in enumerate(self.named_parameters_to_optim):
                z = torch.normal(mean=0, std=1, size=p.data.size(), device=p.data.device, dtype=p.data.dtype)
                # if inv_norm != 1.0:
                #     z = z * inv_norm
                p.data.add_(mult * float(h) * z)
                if bool(collect_stats):
                    u_ss += float(torch.sum(z.float() * z.float()).item())
                    if base_params is not None:
                        bp = base_params[idx]
                        try:
                            diff = (p.data - bp).float()
                            diff_ss += float(torch.sum(diff * diff).item())
                        except Exception:
                            diff_ss = float("nan")
                        try:
                            changed += int(torch.sum(p.data != bp).item())
                            numel += int(p.data.numel())
                        except Exception:
                            pass

        val = float(self.zo_forward(model, inputs).item())

        # Restore
        torch.manual_seed(seed)
        with torch.no_grad():
            for _, p in self.named_parameters_to_optim:
                z = torch.normal(mean=0, std=1, size=p.data.size(), device=p.data.device, dtype=p.data.dtype)
                # if inv_norm != 1.0:
                #     z = z * inv_norm
                p.data.add_(-mult * float(h) * z)

        stats = None
        if bool(collect_stats):
            u_norm = float(math.sqrt(max(u_ss, 0.0)))
            h_eff = None
            r = None
            if base_params is not None and math.isfinite(diff_ss) and u_norm > 0.0:
                diff_norm = float(math.sqrt(max(diff_ss, 0.0)))
                h_eff = float(diff_norm / u_norm)
                if numel > 0:
                    r = float(changed / float(numel))
                stats = {
                    "u_norm": float(u_norm),
                    "diff_norm": float(diff_norm),
                    "h_eff": h_eff,
                    "param_changed_ratio": r,
                    "param_changed_count": int(changed),
                    "numel": int(numel),
                }
            else:
                stats = {
                    "u_norm": float(u_norm),
                    "diff_norm": None,
                    "h_eff": None,
                    "param_changed_ratio": None,
                    "param_changed_count": None,
                    "numel": None,
                }

        return val, stats

    def _hprobe_proj_grad_at(self, model: nn.Module, inputs: Dict[str, Any], h: float, seed: int) -> Tuple[float, float, float, float, Optional[Dict[str, Any]]]:
        fp, stats_p = self._hprobe_eval_at(model, inputs, h=h, seed=seed, mult=+1.0, collect_stats=True)
        fm, _ = self._hprobe_eval_at(model, inputs, h=h, seed=seed, mult=-1.0, collect_stats=False)
        delta = float(fp - fm)
        ghat = float(delta / (2.0 * float(h)))
        return fp, fm, delta, ghat, stats_p

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
        inv_norm = 1.0
        # if getattr(self, "_hprobe_unit_u", False):
        #     inv_norm = float(self._hprobe_get_inv_norm(model, seed))

        torch.manual_seed(seed)
        with torch.no_grad():
            for name, p in self.named_parameters_to_optim:
                z = torch.normal(mean=0, std=1, size=p.data.size(), device=p.data.device, dtype=p.data.dtype)
                # if inv_norm != 1.0:
                #     z = z * inv_norm
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

        inv_norm = 1.0
        # if getattr(self, "_hprobe_unit_u", False):
        #     inv_norm = float(self._hprobe_get_inv_norm(model, seed))

        torch.manual_seed(seed)
        with torch.no_grad():
            for name, p in self.named_parameters_to_optim:
                z = torch.normal(mean=0, std=1, size=p.data.size(), device=p.data.device, dtype=p.data.dtype)
                # if inv_norm != 1.0:
                #     z = z * inv_norm
                if self._hprobe_use_wd(name):
                    p.data = (p.data + lr * (projected_grad * z)) / denom
                else:
                    p.data = p.data + lr * (projected_grad * z)

    def _hprobe_pick_batches(self, current_inputs: Dict[str, Any], nbatch: int):
        """Pick and cache a fixed probe batch (and optional fixed eval batch) for the whole run."""
        probe_batch = getattr(self, "_hprobe_fixed_probe_batch", None)
        eval_batch = getattr(self, "_hprobe_fixed_eval_batch", None)
        if probe_batch is not None:
            return probe_batch, (eval_batch if eval_batch is not None else probe_batch)

        buf = getattr(self, "_h_probe_buffer", None)
        picked = []
        nbatch = max(1, int(nbatch))
        if isinstance(buf, list) and len(buf) > 0:
            replace = len(buf) < nbatch
            idxs = np.random.choice(len(buf), size=nbatch, replace=replace)
            for idx in idxs:
                item = buf[int(idx)]
                picked.append(dict(item) if isinstance(item, dict) else item)
        else:
            picked = [current_inputs]

        probe_batch = picked[0]
        eval_batch = picked[1] if len(picked) > 1 else probe_batch
        self._hprobe_fixed_probe_batch = probe_batch
        self._hprobe_fixed_eval_batch = eval_batch
        return probe_batch, eval_batch

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
            probe_batch, eval_batch = self._hprobe_pick_batches(current_inputs, nbatch=nbatch)

            # (1) Save theta_t (best-effort)
            param_backup = None
            if bool(cfg.get("save_params", True)):
                try:
                    with torch.no_grad():
                        param_backup = [p.data.detach().clone() for _, p in self.named_parameters_to_optim]
                except Exception as e:
                    param_backup = None
                    try:
                        logger.warning(f"[hprobe] param snapshot failed; will rely on exact undo. err={type(e).__name__}: {e}")
                    except Exception:
                        pass

            # Expose snapshot to _hprobe_eval_at for perturbation/quantization diagnostics.
            # Note: when snapshot is unavailable, stats will be recorded as None.
            self._hprobe_param_backup = param_backup
            if param_backup is None:
                try:
                    if not getattr(self, "_hprobe_warned_no_backup", False):
                        logger.warning("[hprobe] no param snapshot available for perturb stats. Consider setting H_PROBE_SAVE_PARAMS=1")
                        self._hprobe_warned_no_backup = True
                except Exception:
                    pass

            # (2) Base loss and true gradient on fixed B_probe
            self._hprobe_unit_u = False  # raw-z probe: never normalize directions
            self._hprobe_inv_norm_cache = {}
            base_probe_loss, grads = self._hprobe_true_grad(model, probe_batch)
            base_eval_loss = float(self.zo_forward(model, eval_batch).item())

            grad_sq = 0.0
            for g in grads:
                if g is None:
                    continue
                grad_sq += float(torch.sum(g.float() * g.float()).item())
            grad_norm = float(math.sqrt(max(grad_sq, 0.0)))

            # (3) Sample M directions (fixed within this probe, re-sampled each probe step)
            ndir = int(cfg["ndir"])
            dir_seeds = [int(np.random.randint(0, 1_000_000_000)) for _ in range(max(1, ndir))]

            # raw-z probe: no direction norm cache needed

            d_true_list = [float(self._hprobe_dot_grad_direction(model, grads, s)) for s in dir_seeds]

            lr = float(self._hprobe_get_lr())
            gs = int(getattr(self.state, "global_step", 0))

            rows = []
            for h in hs:
                d_fd_list, e_list = [], []
                delta_list = []
                h_eff_list, r_list = [], []
                u_norm_list, diff_norm_list = [], []
                for j, seed in enumerate(dir_seeds):
                    _, _, _delta, ghat, stats_p = self._hprobe_proj_grad_at(model, probe_batch, h=float(h), seed=seed)
                    d_fd_list.append(float(ghat))
                    e_list.append(float(d_true_list[j] - float(ghat)))
                    delta_list.append(float(_delta))
                    if isinstance(stats_p, dict):
                        h_eff_list.append(stats_p.get("h_eff"))
                        r_list.append(stats_p.get("param_changed_ratio"))
                        u_norm_list.append(stats_p.get("u_norm"))
                        diff_norm_list.append(stats_p.get("diff_norm"))
                    else:
                        h_eff_list.append(None)
                        r_list.append(None)
                        u_norm_list.append(None)
                        diff_norm_list.append(None)

                e_arr = np.asarray(e_list, dtype=np.float64) if len(e_list) > 0 else None
                dtrue_arr = np.asarray(d_true_list, dtype=np.float64) if len(d_true_list) > 0 else None
                dfd_arr = np.asarray(d_fd_list, dtype=np.float64) if len(d_fd_list) > 0 else None

                def _mean_std(arr):
                    if arr is None or arr.size == 0:
                        return None, None
                    return float(arr.mean()), float(arr.std())

                e_mean, e_std = _mean_std(e_arr)
                e_abs_mean, e_abs_std = _mean_std(np.abs(e_arr) if e_arr is not None else None)
                dtrue_mean, dtrue_std = _mean_std(dtrue_arr)
                dfd_mean, dfd_std = _mean_std(dfd_arr)

                # Direction consistency + correlation (between true and FD directional derivatives)
                dir_sign_match = None
                dir_corr = None
                try:
                    if dtrue_arr is not None and dfd_arr is not None and dtrue_arr.size > 0 and dfd_arr.size > 0:
                        st = np.sign(dtrue_arr)
                        sf = np.sign(dfd_arr)
                        mask = (st != 0) & (sf != 0)
                        if mask.any():
                            dir_sign_match = float(np.mean(st[mask] == sf[mask]))
                        # Pearson correlation requires non-zero variance and >=2 samples
                        if dtrue_arr.size >= 2:
                            std_t = float(np.std(dtrue_arr))
                            std_f = float(np.std(dfd_arr))
                            if std_t > 0.0 and std_f > 0.0:
                                dir_corr = float(np.corrcoef(dtrue_arr, dfd_arr)[0, 1])
                except Exception:
                    dir_sign_match = None
                    dir_corr = None

                # Perturbation/quantization diagnostics (ignore None entries)
                def _mean_std_nonnull(x_list):
                    if not isinstance(x_list, list) or len(x_list) == 0:
                        return None, None
                    vals = [float(x) for x in x_list if (x is not None) and isinstance(x, (int, float)) and math.isfinite(float(x))]
                    if len(vals) == 0:
                        return None, None
                    arr = np.asarray(vals, dtype=np.float64)
                    return float(arr.mean()), float(arr.std())

                h_eff_mean, h_eff_std = _mean_std_nonnull(h_eff_list)
                r_mean, r_std = _mean_std_nonnull(r_list)
                delta_zero_frac = None
                delta_unique = None
                try:
                    if isinstance(delta_list, list) and len(delta_list) > 0:
                        dz = np.asarray(delta_list, dtype=np.float64)
                        delta_zero_frac = float(np.mean(dz == 0.0))
                        delta_unique = int(len(set([float(x) for x in delta_list])))
                except Exception:
                    delta_zero_frac = None
                    delta_unique = None

                # (5) Virtual one-step update: theta'(h) = theta - lr * g_fd(h), evaluate on fixed B_eval
                lr_per_dir = lr / float(max(1, len(dir_seeds)))
                for j, seed in enumerate(dir_seeds):
                    self._hprobe_apply_update(model, seed=seed, projected_grad=d_fd_list[j], lr=lr_per_dir, weight_decay=0.0)
                loss_after = float(self.zo_forward(model, eval_batch).item())
                for j, seed in enumerate(dir_seeds):
                    self._hprobe_undo_update(model, seed=seed, projected_grad=d_fd_list[j], lr=lr_per_dir, weight_decay=0.0)
                deltaL = float(loss_after - base_eval_loss)

                row = {
                    "global_step": gs,
                    "h": float(h),
                    "lr": lr,
                    "unit_u": bool(self._hprobe_unit_u),
                    "ndir": int(len(dir_seeds)),
                    "probe_loss": float(base_probe_loss),
                    "train_loss": float(base_probe_loss),
                    "eval_loss": float(base_eval_loss),
                    "grad_true_norm": float(grad_norm),
                    "d_true_mean": dtrue_mean,
                    "d_true_std": dtrue_std,
                    "d_fd_mean": dfd_mean,
                    "d_fd_std": dfd_std,
                    "e_d_mean": e_mean,
                    "e_d_std": e_std,
                    "e_d_abs_mean": e_abs_mean,
                    "e_d_abs_std": e_abs_std,
                    "dir_sign_match": dir_sign_match,
                    "dir_corr": dir_corr,
                    "deltaL": float(deltaL),
                    "loss_after": float(loss_after),
                    "dir_seeds": dir_seeds,
                    "d_true_list": d_true_list,
                    "d_fd_list": d_fd_list,
                    "e_d_list": e_list,
                    "delta_list": delta_list,
                    "h_eff_mean": h_eff_mean,
                    "h_eff_std": h_eff_std,
                    "h_eff_list": h_eff_list,
                    "param_changed_ratio_mean": r_mean,
                    "param_changed_ratio_std": r_std,
                    "param_changed_ratio_list": r_list,
                    "u_norm_list": u_norm_list,
                    "diff_norm_list": diff_norm_list,
                    "delta_zero_frac": delta_zero_frac,
                    "delta_unique": delta_unique,
                    "quant_noise_indicator": delta_zero_frac,
                }
                rows.append(row)

                if bool(cfg.get("log_each_h", True)):
                    try:
                        logger.info(
                            f"[hprobe] step={gs} h={float(h):.3e} "
                            f"probe_loss={base_probe_loss:.6e} eval_loss={base_eval_loss:.6e} "
                            f"e_abs_mean={e_abs_mean if e_abs_mean is not None else float('nan'):.6e} "
                            f"deltaL={deltaL:.6e} "
                            f"h_eff_mean={h_eff_mean if h_eff_mean is not None else float('nan'):.3e} "
                            f"r_mean={r_mean if r_mean is not None else float('nan'):.3e} "
                            f"d0={delta_zero_frac if delta_zero_frac is not None else float('nan'):.3e}"
                        )
                    except Exception:
                        pass

            out_path = os.path.join(getattr(self.args, "output_dir", "./outputs") or "./outputs", cfg["out_name"])
            self._hprobe_write_jsonl(rows, out_path)
            try:
                logger.info(f"[hprobe] wrote {len(rows)} rows to {out_path} (step={gs})")
            except Exception:
                pass

            # Restore theta_t (if snapshot succeeded)
            if param_backup is not None:
                try:
                    with torch.no_grad():
                        for (_, p), saved in zip(self.named_parameters_to_optim, param_backup):
                            p.data.copy_(saved)
                except Exception:
                    pass

        finally:
            # cleanup per-probe caches
            try:
                if hasattr(self, "_hprobe_inv_norm_cache"):
                    self._hprobe_inv_norm_cache = None
                if hasattr(self, "_hprobe_unit_u"):
                    self._hprobe_unit_u = False
                if hasattr(self, "_hprobe_param_backup"):
                    self._hprobe_param_backup = None
            except Exception:
                pass
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
        # Run probe only on local process zero to avoid duplicated heavy work
        try:
            if hasattr(self.args, "local_rank") and getattr(self.args, "local_rank", -1) not in (-1, 0):
                return
        except Exception:
            pass
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
        # Legacy additive error estimation:
        # ε_f is estimated from finite differences and, for this experiment, those difference evaluations
        # must follow the fp16 two-point path.
        if eps_f_override is not None:
            eps_f = float(eps_f_override)
        else:
            try:
                eps_f = float(getattr(self, "epsilon_f", None))
                if not math.isfinite(eps_f) or eps_f <= 0:
                    raise ValueError("invalid epsilon_f")
            except Exception:
                eps_f = float(self.estimate_noise(model, self.compute_loss, inputs, layer_name=layer_name))
                logger.info(f"[additive error estimation][estimate_nu3] on-the-fly epsilon_f = {eps_f:.3e}")
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
            f0 = float(self._zo_two_point_forward(model, inputs))
        def eval_at(alpha: float) -> float:
            try:
                with torch.no_grad():
                    set_params(alpha)
                    val = float(self._zo_two_point_forward(model, inputs))
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
        eps_train = float(self._get_current_additive_h())
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
        # Legacy additive error estimation:
        # keep parameter arithmetic in float64 for stability, but evaluate the finite differences
        # through the fp16 two-point path (_zo_two_point_forward).
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
                    f_vals.append(float(self._zo_two_point_forward(model, inputs)))
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
        logger.info(f"[additive error estimation][noise] epsilon_f={epsilon_f}")
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

    def _zo_quant_bits(self) -> int:
        return int(getattr(self.args, "zo_quantization_bits", 32))

    def _zo_use_quzo(self) -> bool:
        # Keep 16-bit on the plain MeZO path; only 8/4-bit use QuZO-specific bundle/requantize logic.
        return self._zo_quant_bits() in {8, 4}

    def _quzo_get_bundle(
        self,
        name: str,
        param: nn.Parameter,
        random_vector: Optional[Dict[str, Any]] = None,
        random_seed: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        if random_vector is not None and name in random_vector:
            return random_vector[name]
        step_seed = int(random_seed if random_seed is not None else np.random.randint(2147483647))
        bundle = make_quzo_direction_pair(
            param.data,
            bits=self._zo_quant_bits(),
            key=name,
            step_seed=step_seed,
            target_dtype=param.data.dtype,
        )
        if random_vector is not None:
            random_vector[name] = bundle
        return bundle

    def _quzo_quantize_param_from_bundle(self, param: nn.Parameter, bundle: Dict[str, torch.Tensor]) -> None:
        if self._zo_quant_bits() == 16:
            return
        seed_val = bundle.get("state_seed", None)
        seed = int(seed_val.item()) if isinstance(seed_val, torch.Tensor) else None
        param.data.copy_(
            quantize_tensor(
                param.data,
                self._zo_quant_bits(),
                seed=seed,
                target_dtype=param.data.dtype,
            )
        )

    def _quzo_apply_update_to_param(
        self,
        name: str,
        param: nn.Parameter,
        direction: torch.Tensor,
        projected_grad: Union[torch.Tensor, float],
        learning_rate: float,
        weight_decay: float,
        bundle: Optional[Dict[str, torch.Tensor]] = None,
    ) -> None:
        pg = float(projected_grad.detach().float().item()) if isinstance(projected_grad, torch.Tensor) else float(projected_grad)
        if self._zo_quant_bits() == 16:
            if weight_decay != 0.0 and self._hprobe_use_wd(name):
                param.data.mul_(1.0 - float(learning_rate) * float(weight_decay))
            param.data.add_(direction.detach().to(dtype=param.data.dtype), alpha=-(float(learning_rate) * pg))
            return
        update = float(learning_rate) * (pg * direction.detach().float())
        if weight_decay != 0.0 and self._hprobe_use_wd(name):
            update = update + float(learning_rate) * float(weight_decay) * param.data.detach().float()
        seed_val = None if bundle is None else bundle.get("state_seed", None)
        seed = int(seed_val.item()) if isinstance(seed_val, torch.Tensor) else None
        param.data.copy_(
            quantize_tensor(
                param.data.detach().float() - update,
                self._zo_quant_bits(),
                seed=seed,
                target_dtype=param.data.dtype,
            )
        )

    def _zo_two_point_autocast_context(self):
        precision = str(getattr(self.args, "zo_two_point_precision", "fp32")).lower()
        if precision == "fp16":
            if torch.cuda.is_available() and _use_native_amp and ("autocast" in globals()):
                return autocast(dtype=torch.float16)
            if not getattr(self, "_zo_two_point_fp16_warned", False):
                logger.warning(
                    "[zo] zo_two_point_precision=fp16 requested but CUDA AMP autocast is unavailable; "
                    "falling back to fp32 for two-point evaluations."
                )
                self._zo_two_point_fp16_warned = True
        return nullcontext()

    def _zo_two_point_forward(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
        with self._zo_two_point_autocast_context():
            return self.zo_forward(model, inputs)

    def _zo_fd_projected_grad(self, loss1: torch.Tensor, loss2: torch.Tensor, eps: float) -> torch.Tensor:
        return (loss1.float() - loss2.float()) / (2.0 * float(eps))

    def _setup_zo_probe_csv(self):
        self._zo_probe_csv_path = None
        self._zo_probe_csv_fields = [
            "global_step",
            "eps",
            "zo_two_point_precision",
            "fd_mean",
            "td_mean",
            "mae",
            "mse",
            "rmse",
            "sign_acc",
            "corr",
            "probe_num_seeds",
        ]

        every = int(getattr(self.args, "zo_probe_every", 0))
        if every <= 0 or (not bool(getattr(self.args, "zo_probe_log_csv", True))):
            return

        base_dir = getattr(self.args, "output_dir", "./outputs") or "./outputs"
        os.makedirs(base_dir, exist_ok=True)
        self._zo_probe_csv_path = os.path.join(base_dir, "zo_directional_probe.csv")
        if not os.path.exists(self._zo_probe_csv_path):
            with open(self._zo_probe_csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=self._zo_probe_csv_fields)
                writer.writeheader()

    def _zo_probe_should_run(self) -> bool:
        every = int(getattr(self.args, "zo_probe_every", 0))
        if every <= 0:
            return False
        try:
            if hasattr(self.args, "local_rank") and getattr(self.args, "local_rank", -1) not in (-1, 0):
                return False
        except Exception:
            return False

        global_step = int(getattr(self.state, "global_step", 0))
        if global_step <= 0 or (global_step % every) != 0:
            return False
        if getattr(self, "_zo_probe_last_step", None) == global_step:
            return False
        self._zo_probe_last_step = global_step
        return True

    def _zo_probe_seed_list(self, global_step: int, num_seeds: int) -> List[int]:
        base_seed = int(getattr(self.args, "seed", 0))
        mixed_seed = (base_seed * 1000003 + int(global_step) * 9176 + 97) % 2147483647
        rng = np.random.RandomState(int(mixed_seed))
        return [int(x) for x in rng.randint(0, 2147483647, size=max(1, int(num_seeds)))]

    def _zo_probe_append_csv_row(self, row: Dict[str, Any]):
        if getattr(self, "_zo_probe_csv_path", None) is None:
            return
        try:
            with open(self._zo_probe_csv_path, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=self._zo_probe_csv_fields)
                writer.writerow(row)
        except Exception as e:
            logger.warning(f"[zo_probe] failed to append CSV row: {type(e).__name__}: {e}")

    def _zo_maybe_run_directional_probe(self, model: nn.Module, inputs: Dict[str, Any]):
        if self._zo_use_quzo():
            return
        if not self._zo_probe_should_run():
            return

        global_step = int(getattr(self.state, "global_step", 0))
        num_seeds = max(1, int(getattr(self.args, "zo_probe_num_seeds", 16)))
        eps = float(self._get_training_step_size())
        precision = str(getattr(self.args, "zo_two_point_precision", "fp32")).lower()
        dir_seeds = self._zo_probe_seed_list(global_step, num_seeds)

        fd_vals: List[float] = []
        td_vals: List[float] = []

        zo_forward_step_backup = int(getattr(self.state, "zo_forward_step", 0))
        torch_state = torch.random.get_rng_state()
        cuda_states = None
        try:
            if torch.cuda.is_available():
                cuda_states = torch.cuda.get_rng_state_all()
        except Exception:
            cuda_states = None

        try:
            for seed in dir_seeds:
                if self.args.efficient_zero_order:
                    with torch.no_grad():
                        model = self.efficient_perturb_parameters(model, seed)
                        loss1 = self._zo_two_point_forward(model, inputs)
                        model = self.efficient_perturb_parameters(model, seed, scaling_factor=-2)
                        loss2 = self._zo_two_point_forward(model, inputs)
                        model = self.efficient_perturb_parameters(model, seed, scaling_factor=1)
                    _, td = self.zo_true_directional_derivative(model, inputs, random_seed=seed)
                else:
                    # Deterministically sample the same z direction for FD and true-directional checks.
                    torch.manual_seed(int(seed))
                    if torch.cuda.is_available():
                        torch.cuda.manual_seed_all(int(seed))

                    random_vector = None
                    with torch.no_grad():
                        if self.args.zo_variant is not None:
                            model, random_vector = self.norm_perturb_parameters(model)
                        else:
                            model, random_vector = self.perturb_parameters(model)
                        loss1 = self._zo_two_point_forward(model, inputs)

                        if self.args.zo_variant is not None:
                            model, random_vector = self.norm_perturb_parameters(model, random_vector=random_vector, scaling_factor=-2)
                        else:
                            model, random_vector = self.perturb_parameters(model, random_vector=random_vector, scaling_factor=-2)
                        loss2 = self._zo_two_point_forward(model, inputs)

                        if self.args.zo_variant is not None:
                            model, random_vector = self.norm_perturb_parameters(model, random_vector=random_vector, scaling_factor=1)
                        else:
                            model, random_vector = self.perturb_parameters(model, random_vector=random_vector, scaling_factor=1)
                    _, td = self.zo_true_directional_derivative(model, inputs, random_vector=random_vector)

                fd = self._zo_fd_projected_grad(loss1, loss2, eps)
                fd_vals.append(float(fd.detach().item()))
                td_vals.append(float(td.detach().float().item()))

            fd_arr = np.asarray(fd_vals, dtype=np.float64)
            td_arr = np.asarray(td_vals, dtype=np.float64)
            valid = np.isfinite(fd_arr) & np.isfinite(td_arr)
            fd_arr = fd_arr[valid]
            td_arr = td_arr[valid]
            if fd_arr.size == 0:
                logger.warning(f"[zo_probe] step={global_step}: no finite probe pairs; skipping row.")
                return

            diff = fd_arr - td_arr
            fd_mean = float(np.mean(fd_arr))
            td_mean = float(np.mean(td_arr))
            mae = float(np.mean(np.abs(diff)))
            mse = float(np.mean(diff * diff))
            rmse = float(np.sqrt(np.mean(diff * diff)))
            sign_acc = float(np.mean(np.sign(fd_arr) == np.sign(td_arr)))

            corr = float("nan")
            if fd_arr.size >= 2:
                std_fd = float(np.std(fd_arr))
                std_td = float(np.std(td_arr))
                if std_fd > 0.0 and std_td > 0.0:
                    corr = float(np.corrcoef(fd_arr, td_arr)[0, 1])

            row = {
                "global_step": int(global_step),
                "eps": float(eps),
                "zo_two_point_precision": precision,
                "fd_mean": fd_mean,
                "td_mean": td_mean,
                "mae": mae,
                "mse": mse,
                "rmse": rmse,
                "sign_acc": sign_acc,
                "corr": corr,
                "probe_num_seeds": int(num_seeds),
            }
            self._zo_probe_append_csv_row(row)
            logger.info(
                "[zo_probe] step=%d eps=%.3e precision=%s n=%d mae=%.6e mse=%.6e rmse=%.6e sign_acc=%.4f corr=%s",
                int(global_step),
                float(eps),
                precision,
                int(fd_arr.size),
                mae,
                mse,
                rmse,
                sign_acc,
                f"{corr:.6f}" if math.isfinite(corr) else "nan",
            )
        except Exception as e:
            logger.warning(f"[zo_probe] step={global_step} failed: {type(e).__name__}: {e}")
        finally:
            self.state.zo_forward_step = zo_forward_step_backup
            torch.random.set_rng_state(torch_state)
            try:
                if cuda_states is not None and torch.cuda.is_available():
                    torch.cuda.set_rng_state_all(cuda_states)
            except Exception:
                pass

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

    def zo_true_directional_derivative(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        random_vector: Optional[Dict[str, torch.Tensor]] = None,
        random_seed: Optional[int] = None,
        layer_name: Optional[str] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute the true directional derivative <grad, z> for MeZO.

        This is an ablation utility: keep the same sampled direction z, but replace the finite-difference
        directional derivative (loss1-loss2)/(2*eps) with the true directional derivative <∇L(θ), z>.

        Notes:
        - If random_seed is provided, z will be regenerated on-the-fly using torch.manual_seed(random_seed)
          (mirrors efficient_zero_order behavior).
        - Otherwise random_vector should map parameter name -> z tensor.
        - Uses torch.autograd.grad so it does NOT overwrite/clear param.grad (important for grad accumulation).
        """
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

        named_params = self.named_parameters_to_optim
        if layer_name is not None:
            named_params = [(n, p) for (n, p) in self.named_parameters_to_optim if self.retrieve_c(n) == layer_name]

        params = [p for _, p in named_params]
        grads = torch.autograd.grad(loss, params, retain_graph=False, create_graph=False, allow_unused=True)

        # Match the z used later when writing pseudo-gradients.
        apply_c_scale = (
            getattr(self.args, "use_c_scale", False)
            and getattr(self.args, "zero_order_use_trainer_optim", False)
            and (self.args.zo_variant is not None)
            and (not self.args.change_grad_estimate)
        )

        proj = None
        for (name, param), g in zip(named_params, grads):
            if g is None:
                # keep RNG consumption in sync for efficient mode
                if random_seed is not None:
                    if self._zo_use_quzo():
                        _ = self._quzo_get_bundle(name, param, random_seed=random_seed)
                    else:
                        _ = torch.normal(
                            mean=0,
                            std=1,
                            size=param.data.size(),
                            device=param.data.device,
                            dtype=param.data.dtype,
                        )
                continue

            if self._zo_use_quzo():
                if random_seed is not None:
                    bundle = self._quzo_get_bundle(name, param, random_seed=random_seed)
                elif random_vector is None or name not in random_vector:
                    bundle = self._quzo_get_bundle(name, param)
                    if random_vector is not None:
                        random_vector[name] = bundle
                else:
                    bundle = random_vector[name]
                z = bundle["u2"]
            elif random_seed is not None:
                z = torch.normal(mean=0, std=1, size=param.data.size(), device=param.data.device, dtype=param.data.dtype)
            else:
                if random_vector is None or name not in random_vector:
                    z = torch.normal(mean=0, std=1, size=param.data.size(), device=param.data.device, dtype=param.data.dtype)
                else:
                    z = random_vector[name]

            if apply_c_scale:
                cname = self.retrieve_c(name)
                if cname in self.cs:
                    c_val = self.cs[cname]
                    if isinstance(c_val, torch.Tensor):
                        c_val = float(c_val.item())
                    if c_val != 0.0:
                        z = z * c_val

            contrib = torch.sum(g.detach().float() * z.detach().float())
            proj = contrib if proj is None else proj + contrib

        if proj is None:
            proj = torch.tensor(0.0, device=loss.device)

        self.state.zo_forward_step += 1
        return loss.detach(), proj.detach()

    def efficient_perturb_parameters(self, model: nn.Module, random_seed: int, scaling_factor=1):
        if self._zo_use_quzo():
            for name, param in self.named_parameters_to_optim:
                bundle = self._quzo_get_bundle(name, param, random_seed=random_seed)
                eps = float(self._get_training_step_size())
                delta = bundle["u1"] * (float(scaling_factor) * eps)
                param.data = param.data + delta
                self._quzo_quantize_param_from_bundle(param, bundle)
            return model

        torch.manual_seed(random_seed)
        # 需要 name 以支持按层操作
        for name, param in self.named_parameters_to_optim:
            z = torch.normal(mean=0, std=1, size=param.data.size(), device=param.data.device, dtype=param.data.dtype)
            # === Begin Adaptive h (Berahas et al.) ===
            eps = float(self._get_training_step_size())
            delta = z * (float(scaling_factor) * eps)
            if self._should_quantize_training_perturbation():
                delta = self._quantize_delta_tensor(delta, target_dtype=param.data.dtype)
            param.data = param.data + delta
            # === End Adaptive h ===
        return model

    def norm_perturb_parameters(self, model: nn.Module, random_vector=None, scaling_factor=1):
        if random_vector is None:
            random_vector = {}

        for name, param in self.named_parameters_to_optim:
            bundle = None
            if self._zo_use_quzo():
                bundle = self._quzo_get_bundle(name, param, random_vector=random_vector)
                z = bundle["u1"]
            else:
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
            eps = float(self._get_training_step_size())
            delta = z * (float(scaling_factor) * eps)
            if (not self._zo_use_quzo()) and self._should_quantize_training_perturbation():
                delta = self._quantize_delta_tensor(delta, target_dtype=param.data.dtype)
            param.data = param.data + delta
            if self._zo_use_quzo():
                self._quzo_quantize_param_from_bundle(param, bundle)
            # === End Adaptive h ===

        return model, random_vector

    def perturb_parameters(self, model: nn.Module, random_vector=None, scaling_factor=1):
        if random_vector is None:
            random_vector = {}

        for name, param in self.named_parameters_to_optim:
            bundle = None
            if self._zo_use_quzo():
                bundle = self._quzo_get_bundle(name, param, random_vector=random_vector)
                z = bundle["u1"]
            else:
                if name in random_vector:
                    z = random_vector[name]
                else:
                    z = torch.normal(mean=0, std=1, size=param.data.size(), device=param.data.device, dtype=param.data.dtype)
                    random_vector[name] = z
            # === Begin Adaptive h (Berahas et al.) ===
            eps = float(self._get_training_step_size())
            delta = z * (float(scaling_factor) * eps)
            if (not self._zo_use_quzo()) and self._should_quantize_training_perturbation():
                delta = self._quantize_delta_tensor(delta, target_dtype=param.data.dtype)
            param.data = param.data + delta
            if self._zo_use_quzo():
                self._quzo_quantize_param_from_bundle(param, bundle)
            # === End Adaptive h ===

        return model, random_vector

    def perturb_single_layer(self, model, layer_name, random_vector=None, scaling_factor=1):
        if random_vector is None:
            random_vector = {}

        for name, param in self.named_parameters_to_optim:
            cname = self.retrieve_c(name)
            if cname == layer_name:
                bundle = None
                if self._zo_use_quzo():
                    bundle = self._quzo_get_bundle(name, param, random_vector=random_vector)
                    z = bundle["u1"]
                else:
                    if name in random_vector:
                        z = random_vector[name]
                    else:
                        z = torch.normal(mean=0, std=1, size=param.data.size(), device=param.data.device, dtype=param.data.dtype)
                        random_vector[name] = z
                # === Begin Adaptive h (Berahas et al.) ===
                eps = float(self._get_training_step_size())
                delta = z * (float(scaling_factor) * eps)
                if (not self._zo_use_quzo()) and self._should_quantize_training_perturbation():
                    delta = self._quantize_delta_tensor(delta, target_dtype=param.data.dtype)
                param.data = param.data + delta
                if self._zo_use_quzo():
                    self._quzo_quantize_param_from_bundle(param, bundle)
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

                eps = self._get_training_step_size()
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
        将参数名映射到“层键”（用于分层 c / 分层优化）。
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

        # 2) 其次使用已构造的 layer_names
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
        logger.info(
            "[zo-config] zo_two_point_precision=%s | zero_order_eps=%s | zo_use_true_directional_derivative=%s",
            str(getattr(self.args, "zo_two_point_precision", "fp32")).lower(),
            getattr(self.args, "zero_order_eps", 1e-3),
            bool(getattr(self.args, "zo_use_true_directional_derivative", False)),
        )

        self.state = TrainerState()
        # 初始化 CSV 日志文件
        self._setup_metrics_csv()
        self._setup_zo_probe_csv()
        if int(getattr(self.args, "zo_probe_every", 0)) > 0:
            logger.info(
                "[zo_probe] enabled: every=%d, num_seeds=%d, csv=%s",
                int(getattr(self.args, "zo_probe_every", 0)),
                int(getattr(self.args, "zo_probe_num_seeds", 16)),
                self._zo_probe_csv_path if getattr(self, "_zo_probe_csv_path", None) else "disabled",
            )
        grad_norm_logging = self._grad_norm_log_enabled()
        if grad_norm_logging:
            self._setup_grad_norm_csv()
            logger.info(
                f"[grad_norm] enabled: path={self._grad_norm_csv_path}, every={getattr(self, '_grad_norm_log_every', 1)}"
            )
        else:
            self._grad_norm_csv_path = None
        self._setup_h_estimation_csv()
        _csv_pending = None  # 暂存本 step 的训练度量，待是否有 eval 再一起写入
        self.state.global_step = 0
        start_time = time.time()
        self.state.zo_forward_step = 0
        self._init_h_estimation_state()
        logger.info(
            "[h_estimation][init] h0=%.3e active_source=%s additive=%s two_point=%s",
            float(self._get_init_h()),
            self._get_active_h_source(),
            bool(self._should_compute_additive_h()),
            bool(self._should_compute_two_point_h()),
        )
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
                if self._should_compute_additive_h() or self._should_compute_two_point_h() or self._hprobe_enabled():
                    self._update_h_probe_buffer(inputs)
                # --- h-probes (Probe 1/2/3): stability / delta-loss floor / one-step gain ---
                self._hprobe_maybe_run(model, inputs)

                if self.args.zero_order_optim:
                    # Get parameters that should be optimized (for layer-wise optimization and prefix-tuning)
                    self.named_parameters_to_optim = []
                    for name, param in model.named_parameters():
                        if self.should_optim(name, param):
                            self.named_parameters_to_optim.append((name, param))
                    self._zo_maybe_run_directional_probe(model, inputs)

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
                                if getattr(self.args, "zo_use_true_directional_derivative", False):
                                    # Sample z for this layer (no perturbation), then compute true directional derivative <grad, z>.
                                    model, random_vector = self.perturb_single_layer(model, layer, scaling_factor=0.0)

                                    model.eval()
                                    _in = self._prepare_inputs(inputs)
                                    if self.args.optimize_acc:
                                        loss, logits = model(**_in)
                                        preds = F.softmax(logits, dim=-1)
                                        acc = torch.sum(torch.argmax(preds, 1) == _in['labels']) / len(preds)
                                        loss = -acc
                                    else:
                                        with self.compute_loss_context_manager():
                                            loss = self.compute_loss(model, _in)
                                        if self.args.n_gpu > 1:
                                            loss = loss.mean()

                                    self.state.zo_forward_step += 1

                                    layer_named_params = [(n, p) for (n, p) in self.named_parameters_to_optim if self.retrieve_c(n) == layer]
                                    layer_params = [p for _, p in layer_named_params]
                                    layer_grads = torch.autograd.grad(loss, layer_params, retain_graph=False, create_graph=False, allow_unused=True)

                                    proj = None
                                    for (n, _), g in zip(layer_named_params, layer_grads):
                                        if g is None:
                                            continue
                                        z_base = random_vector[n]["u2"] if self._zo_use_quzo() else random_vector[n]
                                        z_tilde = z_base * (c_i_val if getattr(self.args, "use_c_scale", False) else 1.0)
                                        contrib = torch.sum(g.detach().float() * z_tilde.detach().float())
                                        proj = contrib if proj is None else proj + contrib

                                    projected_grad = proj if proj is not None else torch.tensor(0.0, device=loss.device)
                                    loss1 = loss.detach()
                                    loss2 = loss1
                                else:
                                    model, random_vector = self.perturb_single_layer(model, layer, scaling_factor=1.0/c_i_val)
                                    loss1 = self._zo_two_point_forward(model, inputs)
                                    model, random_vector = self.perturb_single_layer(model, layer, random_vector=random_vector, scaling_factor=-2.0/c_i_val)
                                    loss2 = self._zo_two_point_forward(model, inputs)
                                    model, random_vector = self.perturb_single_layer(model, layer, random_vector=random_vector, scaling_factor=1.0/c_i_val)

                                # Debugging: check for NaN in losses
                                if torch.isnan(loss1).item() or torch.isnan(loss2).item():
                                    logger.warning("NaN encountered in loss during ZO forward step.")

                                # === Begin Adaptive h (Berahas et al.) ===
                                eps = self._get_training_step_size()
                                if not getattr(self.args, "zo_use_true_directional_derivative", False):
                                    projected_grad = self._zo_fd_projected_grad(loss1, loss2, eps)
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
                                        z_base = random_vector[name]["u2"] if self._zo_use_quzo() else random_vector[name]
                                        z_tilde = z_base * (c_i_val if getattr(self.args, "use_c_scale", False) else 1.0)
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

                            if getattr(self.args, "zo_use_true_directional_derivative", False):
                                # Fixed-h ablation: use the true directional derivative <grad, z> for the same z direction.
                                # We do NOT perturb parameters in this mode (but we keep the same z sampling).
                                if self.args.efficient_zero_order:
                                    loss1, projected_grad = self.zo_true_directional_derivative(model, inputs, random_seed=random_seed)
                                elif self.args.zo_variant is not None:
                                    model, random_vector = self.norm_perturb_parameters(model, scaling_factor=0.0)
                                    loss1, projected_grad = self.zo_true_directional_derivative(model, inputs, random_vector=random_vector)
                                else:
                                    model, random_vector = self.perturb_parameters(model, scaling_factor=0.0)
                                    loss1, projected_grad = self.zo_true_directional_derivative(model, inputs, random_vector=random_vector)
                                loss2 = loss1
                            else:
                                with torch.no_grad():
                                    # first function evaluation
                                    if self.args.efficient_zero_order:
                                        model = self.efficient_perturb_parameters(model, random_seed)
                                    elif self.args.zo_variant is not None:
                                        model, random_vector = self.norm_perturb_parameters(model)
                                    else:
                                        model, random_vector = self.perturb_parameters(model)
                                    loss1 = self._zo_two_point_forward(model, inputs)

                                    # second function evaluation
                                    if self.args.efficient_zero_order:
                                        model = self.efficient_perturb_parameters(model, random_seed, scaling_factor=-2)
                                    elif self.args.zo_variant is not None:
                                        model, random_vector = self.norm_perturb_parameters(model, random_vector, scaling_factor=-2)
                                    else:
                                        model, random_vector = self.perturb_parameters(model, random_vector, scaling_factor=-2)
                                    loss2 = self._zo_two_point_forward(model, inputs)

                            # Debugging: check for NaN in losses
                            if torch.isnan(loss1).item() or torch.isnan(loss2).item():
                                logger.warning("NaN encountered in loss during ZO forward step.")

                            # === Begin Adaptive h (Berahas et al.) ===
                            eps = self._get_training_step_size()
                            if not getattr(self.args, "zo_use_true_directional_derivative", False):
                                # === Original Code ===
                                projected_grad = self._zo_fd_projected_grad(loss1, loss2, eps)
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
                                if self.args.efficient_zero_order and (not self._zo_use_quzo()):
                                    # print(random_seed)
                                    torch.manual_seed(random_seed)

                                for name, param in self.named_parameters_to_optim:
                                    # recover noise used in perturbations
                                    if self.args.efficient_zero_order:
                                        if self._zo_use_quzo():
                                            z = self._quzo_get_bundle(name, param, random_seed=random_seed)["u2"]
                                        else:
                                            z = torch.normal(mean=0, std=1, size=param.data.size(), device=param.data.device, dtype=param.data.dtype)
                                    else:
                                        z = random_vector[name]["u2"] if self._zo_use_quzo() else random_vector[name]

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
                            if not getattr(self.args, "zo_use_true_directional_derivative", False):
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
                            grad_l1_step = None
                            grad_l2_step = None
                            if grad_norm_logging:
                                grad_l1_step, grad_l2_step = self._compute_grad_l1_l2_from_param_grads(model)
                                self._write_grad_norm_row(
                                    epoch_val=float(self.epoch),
                                    global_step=int(self.state.global_step),
                                    grad_l1=grad_l1_step,
                                    grad_l2=grad_l2_step,
                                    source="zo_trainer",
                                )
                            # Gradient norm clipping
                            if self.args.zero_order_clip_grad:
                                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), self.args.max_grad_norm)

                            # Update the parameters and step scheduler
                            optimizer.step()
                            if self._zo_use_quzo():
                                with torch.no_grad():
                                    for name, param in self.named_parameters_to_optim:
                                        bundle = self._quzo_get_bundle(name, param, random_seed=int(self.state.global_step))
                                        self._quzo_quantize_param_from_bundle(param, bundle)
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
                                if grad_l1_step is not None and grad_l2_step is not None:
                                    logs["grad_l1_norm"] = float(grad_l1_step)
                                    logs["grad_l2_norm"] = float(grad_l2_step)
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
                        grad_l1_step = None
                        grad_l2_step = None
                        grad_l1_acc = 0.0
                        grad_l2_sq_acc = 0.0
                        for name, param in self.named_parameters_to_optim:
                            bundle = None
                            if self.args.efficient_zero_order:
                                if self._zo_use_quzo():
                                    bundle = self._quzo_get_bundle(name, param, random_seed=random_seed)
                                    z = bundle["u2"]
                                else:
                                    z = torch.normal(mean=0, std=1, size=param.data.size(), device=param.data.device, dtype=param.data.dtype)
                            else:
                                if self._zo_use_quzo():
                                    bundle = random_vector[name]
                                    z = bundle["u2"]
                                else:
                                    z = random_vector[name]
                            grad_est = projected_grad * z
                            if grad_norm_logging:
                                g = grad_est.detach().float()
                                grad_l1_acc += float(torch.sum(torch.abs(g)).item())
                                grad_l2_sq_acc += float(torch.sum(g * g).item())
                            if self._zo_use_quzo():
                                self._quzo_apply_update_to_param(
                                    name,
                                    param,
                                    z,
                                    projected_grad,
                                    float(self.args.learning_rate),
                                    float(self.args.weight_decay),
                                    bundle=bundle,
                                )
                            else:
                                param.data = param.data - self.args.learning_rate * (grad_est + self.args.weight_decay * param.data)

                        if grad_norm_logging:
                            grad_l1_step = float(grad_l1_acc)
                            grad_l2_step = float(math.sqrt(max(grad_l2_sq_acc, 0.0)))
                            self._write_grad_norm_row(
                                epoch_val=float(self.epoch),
                                global_step=int(self.state.global_step),
                                grad_l1=grad_l1_step,
                                grad_l2=grad_l2_step,
                                source="zo_direct",
                            )

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
                                if grad_l1_step is not None and grad_l2_step is not None:
                                    logs["grad_l1_norm"] = float(grad_l1_step)
                                    logs["grad_l2_norm"] = float(grad_l2_step)
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

                        grad_l1_step = None
                        grad_l2_step = None
                        if grad_norm_logging:
                            grad_l1_step, grad_l2_step = self._compute_grad_l1_l2_from_param_grads(model)
                            self._write_grad_norm_row(
                                epoch_val=float(self.epoch),
                                global_step=int(self.state.global_step),
                                grad_l1=grad_l1_step,
                                grad_l2=grad_l2_step,
                                source="fo",
                            )

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
                            if grad_l1_step is not None and grad_l2_step is not None:
                                logs["grad_l1_norm"] = float(grad_l1_step)
                                logs["grad_l2_norm"] = float(grad_l2_step)
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

                additive_refreshed = False
                if (
                    self._should_compute_additive_h()
                    and self.state.global_step > 0
                    and (self.state.global_step % update_noise_every == 0)
                ):
                    self._refresh_additive_h_estimation(model, train_dataloader, inputs)
                    additive_refreshed = True

                two_point_every = max(1, int(getattr(self.args, "two_point_h_refresh_every", update_noise_every)))
                two_point_refreshed = False
                if (
                    self._should_compute_two_point_h()
                    and self.state.global_step > 0
                    and (self.state.global_step % two_point_every == 0)
                ):
                    self._refresh_two_point_h_estimation(model, inputs)
                    two_point_refreshed = True

                if additive_refreshed or two_point_refreshed:
                    self._log_joint_h_estimation_step()

                if self.args.max_steps > 0 and self.state.global_step > self.args.max_steps or (self.args.max_zo_forward_steps > 0 and self.state.zo_forward_step > self.args.max_zo_forward_steps):
                    epoch_iterator.close()
                    break

                # Optional: force eval on the last K steps (each step once)
                try:
                    tail_eval_steps = int(os.environ.get("FINAL_EVAL_STEPS", "0"))
                except Exception:
                    tail_eval_steps = 0

                eval_reason = None
                do_eval = False
                if getattr(self.args, "evaluate_during_training", False) and self.state.global_step % self.args.eval_steps == 0:
                    do_eval = True
                    eval_reason = "YES"
                if tail_eval_steps > 0 and getattr(self.args, "max_steps", 0) and int(self.args.max_steps) > 0:
                    remaining = int(self.args.max_steps) - int(self.state.global_step)
                    if remaining < int(tail_eval_steps) and int(self.state.global_step) > 0:
                        do_eval = True
                        eval_reason = "TAIL"

                if do_eval:
                    output = self.evaluate()
                    metrics = output.metrics
                    objective = self.dev_objective(metrics)
                    # === CSV：本步触发了评估，把评估度量与训练度量一并写入 ===
                    try:
                        eval_loss = float(metrics.get("eval_loss", float("nan"))) if isinstance(metrics, dict) else float("nan")
                        eval_loss_avg5 = None
                        try:
                            if math.isfinite(eval_loss):
                                if not hasattr(self, "_eval_loss_history") or self._eval_loss_history is None:
                                    self._eval_loss_history = []
                                self._eval_loss_history.append((int(self.state.global_step), float(eval_loss)))
                                if len(self._eval_loss_history) >= 5:
                                    last5 = [v for _, v in self._eval_loss_history[-5:]]
                                    eval_loss_avg5 = float(sum(last5) / float(len(last5)))
                        except Exception:
                            eval_loss_avg5 = None
                        eval_acc = self._extract_eval_acc(metrics)
                        with open(self._metrics_csv_path, "a", newline="") as f:
                            writer = csv.writer(f)
                            row = [
                                _csv_pending.get("epoch") if _csv_pending else float(self.epoch),
                                _csv_pending.get("global_step") if _csv_pending else int(self.state.global_step),
                                _csv_pending.get("train_loss") if _csv_pending else float("nan"),
                                _csv_pending.get("train_acc") if _csv_pending else None,
                                (eval_reason if eval_reason is not None else "YES"),
                                eval_loss,
                                (None if eval_acc is None else float(eval_acc)),
                                eval_loss_avg5,
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

        # Optional: extra evals at the very end for stability
        try:
            final_eval_repeat = int(os.environ.get("FINAL_EVAL_REPEAT", "0"))
        except Exception:
            final_eval_repeat = 0
        if final_eval_repeat > 0:
            for i in range(final_eval_repeat):
                try:
                    output = self.evaluate()
                    metrics = output.metrics
                    eval_loss = float(metrics.get("eval_loss", float("nan"))) if isinstance(metrics, dict) else float("nan")
                    eval_acc = self._extract_eval_acc(metrics) if isinstance(metrics, dict) else None

                    eval_loss_avg5 = None
                    try:
                        if math.isfinite(eval_loss):
                            if not hasattr(self, "_eval_loss_history") or self._eval_loss_history is None:
                                self._eval_loss_history = []
                            self._eval_loss_history.append((int(self.state.global_step), float(eval_loss)))
                            if len(self._eval_loss_history) >= 5:
                                last5 = [v for _, v in self._eval_loss_history[-5:]]
                                eval_loss_avg5 = float(sum(last5) / float(len(last5)))
                    except Exception:
                        eval_loss_avg5 = None

                    try:
                        with open(self._metrics_csv_path, "a", newline="") as f:
                            writer = csv.writer(f)
                            row = [
                                float(self.epoch),
                                int(self.state.global_step),
                                float("nan"),
                                None,
                                "FINAL",
                                eval_loss,
                                (None if eval_acc is None else float(eval_acc)),
                                eval_loss_avg5,
                            ]
                            writer.writerow(row)
                    except Exception as e:
                        logger.warning(f"[CSV] failed to write final eval row: {e}")

                    logger.info(f"[final_eval] {i+1}/{final_eval_repeat} eval_loss={eval_loss:.6e}")
                except Exception as e:
                    try:
                        logger.warning(f"[final_eval] failed: {e}")
                    except Exception:
                        pass

        # Write tail-avg eval loss summary (mean of last 5 eval losses)
        try:
            if hasattr(self.args, "local_rank") and getattr(self.args, "local_rank", -1) not in (-1, 0):
                pass
            else:
                hist = getattr(self, "_eval_loss_history", None)
                if isinstance(hist, list) and len(hist) > 0:
                    count = min(5, len(hist))
                    last = hist[-count:]
                    last_steps = [int(s) for s, _ in last]
                    last_vals = [float(v) for _, v in last]
                    avg = float(sum(last_vals) / float(count))
                    summary = {
                        "eval_loss_last5_mean": avg,
                        "eval_loss_last5_count": int(count),
                        "eval_loss_last5_steps": last_steps,
                        "eval_loss_last5_values": last_vals,
                    }
                    out_dir = getattr(self.args, "output_dir", "./outputs") or "./outputs"
                    out_path = os.path.join(out_dir, "eval_loss_last5.json")
                    with open(out_path, "w", encoding="utf-8") as f:
                        json.dump(summary, f, ensure_ascii=False)
                    logger.info(f"[eval_loss_last5] mean={avg:.6e} count={count} steps={last_steps}")
        except Exception as e:
            try:
                logger.warning(f"[eval_loss_last5] failed to write summary: {e}")
            except Exception:
                pass

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
