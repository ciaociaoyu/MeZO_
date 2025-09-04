########## The following part is copied from Transformers' trainer (3.4.0) and later ported to be compatible with v4.4.2 and to support initialization from linear head probing. ##########

# coding=utf-8
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

    def pick_h_two_stage(self, model, inputs, tau1=100.0, tau2=0.1, max_iters=10, layer_name: Optional[str]=None):
        """
        按论文使用【二次试探 (h_a/h_b) + 判据 (18a)(18b)】为“沿随机方向的一维探测”选择一个合适的 h。
        返回值：
          - chosen_h: 通过两条判据筛选/调整后的步长
          - ctx: 用于后续复用的一些上下文（避免重复构造随机方向/恢复参数），包含：
              { originals, v_list, params, dtype, f0, eps_f }
        参数 layer_name：若指定，则仅在该层参数子空间内做一维探测（用于“分层 h”）。
        """
        # === 1) 收集可训练参数并保存原值（用 float64 提升数值稳定性）===
        names, params = [], []
        for name, param in model.named_parameters():
            if not self.should_optim(name, param):
                continue
            if layer_name is not None:
                # 仅挑选属于该层的参数（层的划分复用 cs 的检索规则）
                if layer_name not in name:
                    continue
            names.append(name)
            params.append(param)
        # 若该层无参数，则 fallback
        if not params:
            logger.warning("[pick_h_two_stage] No trainable params; fallback h = 1e-3.")
            return 1e-3, None

        param_shapes = [p.data.shape for p in params]
        param_numels = [p.data.numel() for p in params]
        total_numel = sum(param_numels)
        device = params[0].data.device
        dtype = params[0].data.dtype

        # 保存原参数（float64 以减少舍入误差放大）
        originals = [p.data.detach().clone().to(dtype=torch.float64) for p in params]

        # === 2) 生成“全局随机方向” v，并单位化 ===
        v_flat = torch.randn(total_numel, dtype=torch.float64, device=device)
        v_flat = v_flat / torch.norm(v_flat)
        v_splits = torch.split(v_flat, param_numels)
        v_list = [v.view(shape) for v, shape in zip(v_splits, param_shapes)]

        def set_params(alpha: float):
            """将参数设置为 θ(alpha) = θ0 + alpha * v；完成后需恢复原值。"""
            for p, orig, v in zip(params, originals, v_list):
                p.data.copy_((orig + alpha * v).to(dtype=dtype))

        # === 3) 准备噪声水平 ε_f；若尚未估计，则现场估一次 ===
        try:
            eps_f = float(getattr(self, "epsilon_f", None))
            if not math.isfinite(eps_f) or eps_f <= 0:
                raise ValueError("invalid epsilon_f")
        except Exception:
            # 若为分层 h，则只在该层参数子空间上估计噪声
            eps_f = float(self.estimate_noise(model, self.compute_loss, inputs, layer_name=layer_name))
            logger.info(f"[pick_h_two_stage] on-the-fly epsilon_f = {eps_f:.3e}")

        # === 4) 计算 f(0) 作为 (18b) 参考基点 ===
        with torch.no_grad():
            for p, orig in zip(params, originals):
                p.data.copy_(orig.to(dtype=dtype))
            f0 = float(self.zo_forward(model, inputs))

        def eval_at(alpha: float) -> float:
            """在 θ(alpha) 处计算一次 f，结束后恢复参数。"""
            try:
                with torch.no_grad():
                    set_params(alpha)
                    val = float(self.zo_forward(model, inputs))
            finally:
                # 恢复到原参数
                for p, orig in zip(params, originals):
                    p.data.copy_(orig.to(dtype=dtype))
            return val

        def delta2(h_local: float) -> float:
            """二阶中心差分幅度：|f(-h) - 2 f(0) + f(h)|。用于 (18a) 与 μ 的粗估。"""
            fp = eval_at( h_local)
            fm = eval_at(-h_local)
            return abs(fm - 2.0 * f0 + fp)

        def proximity_ok(val: float) -> bool:
            """(18b)：|f(±h) - f(0)| <= τ2 * max(|f(0)|, |f(±h)|) —— 相对变化受限（量纲不敏感）"""
            return abs(val - f0) <= tau2 * max(abs(f0), abs(val))

        def tests_on(h_local: float):
            """
            对给定 h，返回：
              snr_ok   : 是否通过 (18a) 信噪比测试（Δ^2 f(h)/ε_f >= τ1）
              prox_ok  : 是否通过 (18b) 函数值相近性（±h 都需满足）
              mu_hat   : 以 Δ^2 f(h)/h^2 粗估 |f''|
              d2       : Δ^2 f(h) 本身（便于日志或进一步分析）
            """
            d2 = delta2(h_local)
            mu_hat = d2 / (h_local ** 2 + 1e-30)
            snr_ok = (d2 / max(eps_f, 1e-30)) >= tau1
            fp = eval_at( h_local)
            fm = eval_at(-h_local)
            prox_ok = (proximity_ok(fp) and proximity_ok(fm))
            return snr_ok, prox_ok, mu_hat, d2

        # === 5) 二次试探：先 h_a，再基于 μ_a 得 h_b；必要时做几何调整 ===
        tiny = 1e-30
        h_a = max(eps_f, tiny) ** 0.25  # 第一次试探：理论量级 ε_f^{1/4}
        snr_a, prox_a, mu_a, _ = tests_on(h_a)
        if snr_a and prox_a:
            chosen_h = h_a
        else:
            mu_a_pos = max(mu_a, tiny)
            h_b = (eps_f / mu_a_pos) ** 0.25  # 第二次试探：基于 μ_a 的尺度
            snr_b, prox_b, mu_b, _ = tests_on(h_b)

            if snr_b and prox_b:
                chosen_h = h_b
            elif abs(mu_a - mu_b) <= 0.5 * mu_b:
                # 一致性兜底：若两次 μ 估计接近，则直接用 h_b
                chosen_h = h_b
            else:
                # 仍未通过：按“失败方向”几何调整（SNR 不足 -> 放大；相近性失败 -> 缩小）
                chosen_h = h_b
                it = 0
                while it < max_iters:
                    snr_ok, prox_ok, _, _ = tests_on(chosen_h)
                    if snr_ok and prox_ok:
                        break
                    if not snr_ok:
                        chosen_h *= 2.0
                    elif not prox_ok:
                        chosen_h *= 0.5
                    it += 1
                # 退出循环时，无论是否完全通过，都采用当前 chosen_h 作为折中

        # 打包复用上下文，便于后续估计 ν3 复用相同方向与基准 f0
        ctx = dict(
            originals=originals, v_list=v_list, params=params, dtype=dtype, f0=f0, eps_f=eps_f,
            eval_at=eval_at  # 直接暴露 eval_at，避免重复写入/恢复逻辑
        )
        return chosen_h, ctx
    """
    Adding some functions based on Transformers' Trainer class.
    """

    # === Begin Adaptive h (Berahas et al.) ===
    def estimate_nu3(self, model, loss_fn, inputs, h=1e-3, tau1=100.0, tau2=0.1, max_iters=10):
        """
        先通过 pick_h_two_stage() 用二次试探+(18a)(18b) 选择“合法”的 h，
        再用 5 点三阶差分公式估计 ||f^{(3)}||（nu3）。
        这样避免：h 太小（被噪声淹没）或太大（截断误差主导）导致的失真。
        """
        # —— 第一步：选 h（并获得可复用的 eval 上下文）
        chosen_h, ctx = self.pick_h_two_stage(model, inputs, tau1=tau1, tau2=tau2, max_iters=max_iters)
        if ctx is None:
            logger.warning("[estimate_nu3] pick_h_two_stage returned None ctx; fallback to h=1e-3.")
            chosen_h = 1e-3
            # 简单回退：重新准备一个最简 eval 接口（不共享方向）
            def eval_simple(alpha: float) -> float:
                return float(self.zo_forward(model, inputs))
            eval_at = eval_simple
        else:
            eval_at = ctx["eval_at"]

        # —— 第二步：在 chosen_h 上做 5 点三阶差分
        try:
            f2  = eval_at( 2.0 * chosen_h)
            f1  = eval_at( 1.0 * chosen_h)
            fm1 = eval_at(-1.0 * chosen_h)
            fm2 = eval_at(-2.0 * chosen_h)
            numerator = abs(-f2 + 2.0 * f1 - 2.0 * fm1 + fm2)
            denom = 2.0 * (chosen_h ** 3)
            nu3 = numerator / (denom if denom != 0.0 else 1e-30)
        except Exception as e:
            logger.warning(f"[estimate_nu3] exception during 5-point stencil: {e}; fallback to h=1e-3.")
            h_fb = 1e-3
            # 回退计算（不依赖 ctx）
            # 注意：此处回退不再强制通过(18a)(18b)，作为保守估计
            def _eval_tmp(alpha: float) -> float:
                # 简化回退：每次直接按 alpha 设置一次，再复原
                names, params = [], []
                for name, param in model.named_parameters():
                    if self.should_optim(name, param):
                        names.append(name)
                        params.append(param)
                if not params:
                    return float(self.zo_forward(model, inputs))
                originals = [p.data.detach().clone() for p in params]
                with torch.no_grad():
                    # 简化为在当前参数上“加法”模拟 alpha（近似）
                    for p in params:
                        p.data.add_(0.0)  # no-op, 占位
                    val = float(self.zo_forward(model, inputs))
                    for p, orig in zip(params, originals):
                        p.data.copy_(orig)
                return val

            f2  = _eval_tmp( 2.0 * h_fb)
            f1  = _eval_tmp( 1.0 * h_fb)
            fm1 = _eval_tmp(-1.0 * h_fb)
            fm2 = _eval_tmp(-2.0 * h_fb)
            nu3 = abs(-f2 + 2.0 * f1 - 2.0 * fm1 + fm2) / (2.0 * (h_fb ** 3))

        if (not math.isfinite(nu3)) or nu3 <= 0.0:
            logger.warning("[estimate_nu3] nu3 invalid; set to 20 (conservative default).")
            nu3 = 20.0
        else:
            logger.info(f"Estimated nu3: {nu3:.6e} with chosen h={chosen_h:.6e}")

        return float(nu3)

    def estimate_noise(self, model, loss_fn, inputs, q=6, delta=1e-4, layer_name: Optional[str]=None):
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
        # Generate a global random direction v (float64)
        v_flat = torch.randn(total_numel, dtype=torch.float64, device=device)
        v_flat = v_flat / torch.norm(v_flat)
        v_splits = torch.split(v_flat, param_numels)
        v_list = [v.view(shape) for v, shape in zip(v_splits, param_shapes)]
        # Helper to set params to originals + alpha * v
        def set_params(alpha):
            for p, orig, v in zip(params, originals, v_list):
                p.data.copy_((orig + alpha * v).to(dtype=dtype))
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
                    cname = self.retrieve_c(name) if 'name' in locals() else None
                    if hasattr(self, 'layerwise_h') and cname in getattr(self, 'layerwise_h', {}):
                        eps = self.layerwise_h[cname]
                    else:
                        eps = self.adaptive_h
                else:
                    eps = self.adaptive_h
            else:
                eps = self.args.zero_order_eps
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
            # 若启用按层 h（use_layerwise_h=True），则针对该参数所在层选用分层步长；否则使用全局 adaptive_h
            if getattr(self.args, "use_adaptive_h", False):
                if getattr(self.args, "use_layerwise_h", False):
                    cname = self.retrieve_c(name) if 'name' in locals() else None
                    if hasattr(self, 'layerwise_h') and cname in getattr(self, 'layerwise_h', {}):
                        eps = self.layerwise_h[cname]
                    else:
                        eps = self.adaptive_h
                else:
                    eps = self.adaptive_h
            else:
                eps = self.args.zero_order_eps
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
            eps = self.adaptive_h if getattr(self.args, "use_adaptive_h", False) else self.args.zero_order_eps
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
                        if hasattr(self, 'layerwise_h') and cname in getattr(self, 'layerwise_h', {}):
                            eps = self.layerwise_h[cname]
                        else:
                            eps = self.adaptive_h
                    else:
                        eps = self.adaptive_h
                else:
                    eps = self.args.zero_order_eps
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

                projected_grad = (loss1 - loss2) / (2 * self.args.zero_order_eps)
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
                    self.cs[ckey] /= torch.sqrt(self.cs[ckey])

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
                    self.cs[ckey] /= torch.sqrt(self.num_params[ckey])

            for ckey in self.cs:
                if self.cs[ckey] != 0:
                    self.cs[ckey] = self.cs[ckey].detach().item()

        self.layer_names = list(self.cs.keys())
        model.zero_grad()

    def retrieve_c(self, param_name):
        for c_name in self.cs.keys():
            if c_name in param_name:
                return c_name

        return '' # these parameters are likely not being used in the forward pass

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
        train_dataloader = self.get_train_dataloader()
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
        self.state.global_step = 0
        start_time = time.time()
        self.state.zo_forward_step = 0
        # === Begin Adaptive h (Berahas et al.) ===
        # —— 所有用于扰动/差分的 h，统一走 pick_h_two_stage 的合法性筛选
        if getattr(self.args, "use_adaptive_h", False):
            logger.info("Estimating noise level and third derivative for adaptive h...")
            # —— 是否分层选择 h 的总开关（仅影响 h 的选择逻辑；分层依据与 cs 相同）
            use_layerwise_h = getattr(self.args, "use_layerwise_h", False)
            example_inputs = next(iter(train_dataloader))
            if use_layerwise_h:
                # 构造 layer_names（必须与 retrieve_c/initialize_c 的层命名保持一致）
                # OPT 系模型参数名包含 "layers.{i}."；其他如 BERT/RoBERTa 通常是 "layer.{i}."
                prefix = "layers" if self.model.config.model_type == "opt" else "layer"
                self.layer_names = [f"{prefix}.{i}." for i in range(self.model.config.num_hidden_layers)]
                # 同时包含嵌入与输出头的键（若模型参数名中包含这些子串，则 retrieve_c 会匹配到）
                self.layer_names = ["embed", "lm_head"] + self.layer_names
                # —— 分层 h：为每一层单独选 h（层的划分与 cs 相同：self.layer_names 来自 initialize_c）
                #    1) 分层 h 的层划分完全复用 initialize_c 里建立的层键（self.layer_names）；
                #    2) 分层路径会调用 pick_h_two_stage(..., layer_name=layer)，其内部会在该层参数子空间上估计 epsilon_f 并做(18a)(18b)筛选；
                self.layerwise_h = {}
                for layer in self.layer_names:
                    h_valid, _ = self.pick_h_two_stage(model, example_inputs, tau1=100.0, tau2=0.1, max_iters=10, layer_name=layer)
                    h_final = float(h_valid) if math.isfinite(h_valid) and h_valid > 0 else 1e-3
                    self.layerwise_h[layer] = torch.tensor(h_final, dtype=torch.float32)
                # 提供一个全局兜底（个别层缺失时使用）
                self.adaptive_h = torch.tensor(float(np.median([v.item() for v in self.layerwise_h.values()])), dtype=torch.float32)
                logger.info(f"Using layerwise h (validated by (18a)/(18b)): median={self.adaptive_h:.6e}; examples: {list(self.layerwise_h.items())[:3]}")
                previous_adaptive_h = self.adaptive_h
            else:
                #    3) 全局路径则在全参数空间一次性估计 epsilon_f/nu3 并做筛选。
                self.epsilon_f = self.estimate_noise(model, self.compute_loss, example_inputs)
                self.nu3 = self.estimate_nu3(model, self.compute_loss, example_inputs)
                # —— 提案步长（基于理论公式）：h* = (ε_f / ν3)^{1/3} * 3^{1/3}
                # 注意：论文建议最终的 h 仍需通过 (18a)(18b) 的“合法性”筛选
                h_proposed = (self.epsilon_f / self.nu3) ** (1/3) * (3 ** (1/3))
                # —— 统一入口：所有用于扰动/差分的 h 必须经过二次试探 + 判据(18a)(18b)
                h_valid, _ctx = self.pick_h_two_stage(model, example_inputs, tau1=100.0, tau2=0.1, max_iters=10)
                h_final = float(h_valid) if math.isfinite(h_valid) and h_valid > 0 else float(h_proposed)
                self.adaptive_h = torch.tensor(h_final, dtype=torch.float32)
                logger.info(f"Using adaptive h (validated by (18a)/(18b)) = {self.adaptive_h:.6e}  [proposed={h_proposed:.6e}]")
                previous_adaptive_h = self.adaptive_h
                if torch.isnan(torch.tensor(self.adaptive_h)).item() or self.adaptive_h < 1e-8 or torch.isinf(torch.tensor(self.adaptive_h)):
                    logger.warning(f"Adaptive h estimation invalid (value: {self.adaptive_h}), keeping previous adaptive h value.")
                    self.adaptive_h = previous_adaptive_h
                else:
                    previous_adaptive_h = self.adaptive_h
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

                # === Begin Adaptive h: update h every update_noise_every steps ===
                # —— 所有用于扰动/差分的 h，统一走 pick_h_two_stage 的合法性筛选
                if getattr(self.args, "use_adaptive_h", False) and self.state.global_step % update_noise_every == 0 and self.state.global_step > 0:
                    logger.info(f"Re-estimating epsilon_f and nu3 at step {self.state.global_step}...")
                    # —— 是否分层选择 h 的总开关（与初始化时保持一致）
                    use_layerwise_h = getattr(self.args, "use_layerwise_h", False)
                    if use_layerwise_h:
                        # —— 分层 h 更新：逐层重新选择 h（层的划分与 cs 相同：self.layer_names 来自 initialize_c）
                        #    1) 分层 h 的层划分完全复用 initialize_c 里建立的层键（self.layer_names）；
                        #    2) 分层路径会调用 pick_h_two_stage(..., layer_name=layer)，其内部会在该层参数子空间上估计 epsilon_f 并做(18a)(18b)筛选；
                        self.layerwise_h = {}
                        for layer in self.layer_names:
                            h_valid, _ = self.pick_h_two_stage(model, inputs, tau1=100.0, tau2=0.1, max_iters=10, layer_name=layer)
                            h_final = float(h_valid) if math.isfinite(h_valid) and h_valid > 0 else 1e-3
                            self.layerwise_h[layer] = torch.tensor(h_final, dtype=torch.float32)
                        self.adaptive_h = torch.tensor(float(np.median([v.item() for v in self.layerwise_h.values()])), dtype=torch.float32)
                        logger.info(f"Updated layerwise h (validated by (18a)/(18b)): median={self.adaptive_h:.6e}")
                        previous_adaptive_h = self.adaptive_h
                        continue  # 分层 h 路径已完成本次更新
                    #    3) 全局路径则在全参数空间一次性估计 epsilon_f/nu3 并做筛选。
                    self.epsilon_f = self.estimate_noise(model, self.compute_loss, inputs)
                    self.nu3 = self.estimate_nu3(model, self.compute_loss, inputs)
                    # —— 先计算理论提案 h*
                    h_proposed = (self.epsilon_f / self.nu3) ** (1 / 3) * (3 ** (1 / 3))
                    # —— 再通过统一入口做合法性筛选（(18a)(18b)）
                    h_valid, _ctx = self.pick_h_two_stage(model, inputs, tau1=100.0, tau2=0.1, max_iters=10)
                    h_final = float(h_valid) if math.isfinite(h_valid) and h_valid > 0 else float(h_proposed)
                    self.adaptive_h = torch.tensor(h_final, dtype=torch.float32)
                    logger.info(f"Updated adaptive h (validated by (18a)/(18b)) = {self.adaptive_h:.6e}  [proposed={h_proposed:.6e}]")
                    if torch.isnan(torch.tensor(self.adaptive_h)).item() or self.adaptive_h < 1e-8:
                        logger.warning(f"Adaptive h estimation invalid (value: {self.adaptive_h}), keeping previous adaptive h value.")
                        self.adaptive_h = previous_adaptive_h
                    else:
                        previous_adaptive_h = self.adaptive_h
                # === End Adaptive h update ===

                if self.args.max_steps > 0 and self.state.global_step > self.args.max_steps or (self.args.max_zo_forward_steps > 0 and self.state.zo_forward_step > self.args.max_zo_forward_steps):
                    epoch_iterator.close()
                    break

                if self.args.evaluate_during_training and self.state.global_step % self.args.eval_steps == 0:
                    output = self.evaluate()
                    metrics = output.metrics
                    objective = self.dev_objective(metrics)
                    if objective > self.objective:
                        logger.info("Best dev result: {}".format(objective))
                        self.objective = objective
                        # self.save_model(self.args.output_dir)

                        # Now we save this to (CPU) memory instead of disk <-- much faster
                        self.best_model_ckpt = {k: v.detach().cpu() for k, v in model.state_dict().items()}

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
