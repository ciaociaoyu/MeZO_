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

import contextlib
import functools
import glob
import inspect
import math
import os
import random
import re
import shutil
import sys
import time
import warnings
from collections.abc import Mapping
from collections import deque
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union
import copy
 # ------------------------------------------------------------------
# Local F1 helper (previously imported from a non-existent `metrics` module)
# Used only in `zo_forward_nondiff` for SQuAD-style evaluation.
# ------------------------------------------------------------------

def _simple_f1(prediction: str, ground_truth: str) -> float:
    """Compute token-level F1 score between two strings.

    This is a lightweight replacement to avoid an external `metrics` dependency.
    """
    pred_tokens = prediction.strip().split()
    gt_tokens = ground_truth.strip().split()

    if len(pred_tokens) == 0 and len(gt_tokens) == 0:
        return 1.0
    if len(pred_tokens) == 0 or len(gt_tokens) == 0:
        return 0.0

    common = {}
    for t in pred_tokens:
        common[t] = common.get(t, 0) + 1

    num_same = 0
    for t in gt_tokens:
        if common.get(t, 0) > 0:
            num_same += 1
            common[t] -= 1

    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(gt_tokens)
    return 2 * precision * recall / (precision + recall)
import numpy as np

from tqdm.auto import tqdm
from transformers import Trainer
from sklearn.linear_model import LinearRegression, LogisticRegression, LogisticRegressionCV

# Integrations must be imported before ML frameworks:
from transformers.integrations import (  # isort: split
    default_hp_search_backend,
    get_reporting_integration_callbacks,
    hp_params,
    is_fairscale_available,
    is_optuna_available,
    is_ray_tune_available,
    is_sigopt_available,
    is_wandb_available,
    run_hp_search_optuna,
    run_hp_search_ray,
    run_hp_search_sigopt,
    run_hp_search_wandb,
)

import numpy as np
import torch
import torch.distributed as dist
from packaging import version
from torch import nn
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

from huggingface_hub import Repository

from transformers import __version__
from transformers.configuration_utils import PretrainedConfig
from transformers.data.data_collator import DataCollator, DataCollatorWithPadding, default_data_collator
from transformers.debug_utils import DebugOption, DebugUnderflowOverflow
from transformers.deepspeed import deepspeed_init, is_deepspeed_zero3_enabled
from transformers.dependency_versions_check import dep_version_check
from transformers.modelcard import TrainingSummary
from transformers.modeling_utils import PreTrainedModel, load_sharded_checkpoint, unwrap_model
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES, MODEL_MAPPING_NAMES
from transformers.optimization import Adafactor, get_scheduler
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS, is_torch_greater_or_equal_than_1_10, \
    is_torch_less_than_1_11
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.trainer_callback import (
    CallbackHandler,
    DefaultFlowCallback,
    PrinterCallback,
    ProgressCallback,
    TrainerCallback,
    TrainerControl,
    TrainerState,
)
from transformers.trainer_pt_utils import (
    DistributedLengthGroupedSampler,
    DistributedSamplerWithLoop,
    DistributedTensorGatherer,
    IterableDatasetShard,
    LabelSmoother,
    LengthGroupedSampler,
    SequentialDistributedSampler,
    ShardSampler,
    distributed_broadcast_scalars,
    distributed_concat,
    find_batch_size,
    get_module_class_from_name,
    get_parameter_names,
    nested_concat,
    nested_detach,
    nested_numpify,
    nested_truncate,
    nested_xla_mesh_reduce,
    reissue_pt_warnings,
)
from transformers.trainer_utils import (
    PREFIX_CHECKPOINT_DIR,
    BestRun,
    EvalLoopOutput,
    EvalPrediction,
    FSDPOption,
    HPSearchBackend,
    HubStrategy,
    IntervalStrategy,
    PredictionOutput,
    RemoveColumnsCollator,
    ShardedDDPOption,
    TrainerMemoryTracker,
    TrainOutput,
    default_compute_objective,
    default_hp_space,
    denumpify_detensorize,
    enable_full_determinism,
    find_executable_batch_size,
    get_last_checkpoint,
    has_length,
    number_of_arguments,
    seed_worker,
    set_seed,
    speed_metrics,
)
from transformers.training_args import OptimizerNames, ParallelMode, TrainingArguments
from transformers.utils import (
    CONFIG_NAME,
    WEIGHTS_INDEX_NAME,
    WEIGHTS_NAME,
    find_labels,
    get_full_repo_name,
    is_apex_available,
    is_datasets_available,
    is_in_notebook,
    is_ipex_available,
    is_sagemaker_dp_enabled,
    is_sagemaker_mp_enabled,
    is_torch_tensorrt_fx_available,
    is_torch_tpu_available,
    is_torchdynamo_available,
    logging,
)
from transformers.utils.generic import ContextManagers

_is_native_cpu_amp_available = is_torch_greater_or_equal_than_1_10

DEFAULT_CALLBACKS = [DefaultFlowCallback]
DEFAULT_PROGRESS_CALLBACK = ProgressCallback

if is_in_notebook():
    from .utils.notebook import NotebookProgressCallback

    DEFAULT_PROGRESS_CALLBACK = NotebookProgressCallback

if is_apex_available():
    from apex import amp

if is_datasets_available():
    import datasets

if is_torch_tpu_available(check_device=False):
    import torch_xla.core.xla_model as xm
    import torch_xla.debug.metrics as met
    import torch_xla.distributed.parallel_loader as pl

if is_fairscale_available():
    dep_version_check("fairscale")
    import fairscale
    from fairscale.nn.data_parallel import FullyShardedDataParallel as FullyShardedDDP
    from fairscale.nn.data_parallel import ShardedDataParallel as ShardedDDP
    from fairscale.nn.wrap import auto_wrap
    from fairscale.optim import OSS
    from fairscale.optim.grad_scaler import ShardedGradScaler

if is_sagemaker_mp_enabled():
    import smdistributed.modelparallel.torch as smp
    from smdistributed.modelparallel import __version__ as SMP_VERSION

    IS_SAGEMAKER_MP_POST_1_10 = version.parse(SMP_VERSION) >= version.parse("1.10")

    from .trainer_pt_utils import smp_forward_backward, smp_forward_only, smp_gather, smp_nested_concat
else:
    IS_SAGEMAKER_MP_POST_1_10 = False

if TYPE_CHECKING:
    import optuna

logger = logging.get_logger(__name__)

# Name of the files used for checkpointing
TRAINING_ARGS_NAME = "training_args.bin"
TRAINER_STATE_NAME = "trainer_state.json"
OPTIMIZER_NAME = "optimizer.pt"
SCHEDULER_NAME = "scheduler.pt"
SCALER_NAME = "scaler.pt"


class OurTrainer(Trainer):
    from transformers.trainer_pt_utils import _get_learning_rate, log_metrics, metrics_format, save_metrics, save_state

    # ---------------------------------------------------------------------
    # Sampler override ("sample" 相关):
    #   - dataloader_shuffle=True  -> RandomSampler (shuffle)
    #   - dataloader_shuffle=False -> SequentialSampler (no shuffle)
    # 这样可以避免训练阶段一直使用 SequentialSampler 导致的分布漂移/不稳定。
    # ---------------------------------------------------------------------
    def _get_train_sampler(self) -> Optional[torch.utils.data.Sampler]:
        """Override HF Trainer's train sampler selection.

        Notes
        -----
        - We keep HF's default behavior for group_by_length.
        - For distributed training we use DistributedSampler with shuffle controlled by args.dataloader_shuffle.
        """
        if self.train_dataset is None or not has_length(self.train_dataset):
            return None

        # Let HF handle special samplers (e.g., LengthGroupedSampler)
        if getattr(self.args, "group_by_length", False):
            return super()._get_train_sampler()

        shuffle = bool(getattr(self.args, "dataloader_shuffle", True))

        if self.args.local_rank != -1:
            return DistributedSampler(self.train_dataset, shuffle=shuffle, seed=self.args.seed)

        if shuffle:
            # 用独立 generator 固定 DataLoader 的随机性，避免被 MeZO 里频繁 torch.manual_seed(...) 干扰
            data_seed = getattr(self.args, "data_seed", None)
            if data_seed is None:
                data_seed = self.args.seed
            g = torch.Generator()
            g.manual_seed(int(data_seed))
            return RandomSampler(self.train_dataset, generator=g)

        return SequentialSampler(self.train_dataset)
    # ---------------------------------------------------------------------
    # Adaptive eps ("alpha" 相关):
    # 在用 (eps_f, nu3) 计算训练 eps* 时，引入 alpha (<1) 来下调截断误差项的权重，
    # 使 eps* 变大： eps* <- alpha^{-1/6} * eps*.
    #
    # 同时，为了避免 probe batch 绑定某个固定方向/固定 batch，我们使用滚动 buffer + 随机抽样。
    # ---------------------------------------------------------------------

    def _get_base_zo_eps(self) -> float:
        """Return the *configured* (non-adaptive) eps.

        We support both naming conventions:
        - args.zo_eps (MeZO scripts)
        - args.zero_order_eps (kernel/other scripts)
        """
        if hasattr(self.args, "zo_eps") and self.args.zo_eps is not None:
            return float(self.args.zo_eps)
        if hasattr(self.args, "zero_order_eps"):
            return float(self.args.zero_order_eps)
        # last-resort fallback
        return 1e-3

    def _get_current_zo_eps(self) -> float:
        """Return the eps actually used for ZO gradient estimation."""
        if getattr(self.args, "use_adaptive_h", False) and hasattr(self, "adaptive_h"):
            return float(self.adaptive_h)
        return self._get_base_zo_eps()

    def _adaptive_h_init(self):
        """Initialize adaptive-h state (called once at train start)."""
        # EMA smoothing beta (externalizable; default 0.1)
        self._h_beta = float(getattr(self.args, "h_beta", 0.1))
        # truncation-error down-weight alpha (<1 => larger eps*)
        self._h_trunc_alpha = float(getattr(self.args, "h_trunc_alpha", 1.0))

        # rolling buffer of recent batches (randomly subsampled for probe)
        buf_size = int(getattr(self.args, "h_probe_buffer", 64))
        self._h_probe_buffer = deque(maxlen=max(buf_size, 1))

        # (nb, nd) for averaging: nb batches, nd directions
        self._h_nb = int(getattr(self.args, "h_nb", 1))
        self._h_nd = int(getattr(self.args, "h_nd", 3))
        self._h_reduce = str(getattr(self.args, "h_reduce", "mean"))

        # thresholds for local-scale tests (SNR & proximity)
        self._h_tau1 = float(getattr(self.args, "h_tau1", 5.0))
        self._h_tau2 = float(getattr(self.args, "h_tau2", 0.2))

        # internal guard
        self._h_last_update_step = -1

        # init value: start from configured eps
        self.adaptive_h = self._get_base_zo_eps()

        if self.is_world_process_zero():
            logger.info(
                f"[adaptive h][init] h0={self.adaptive_h:.3e} beta={self._h_beta} alpha={self._h_trunc_alpha} "
                f"(nb={self._h_nb}, nd={self._h_nd}, buf={self._h_probe_buffer.maxlen}, reduce={self._h_reduce})"
            )

    def _adaptive_h_buffer_add(self, inputs: Dict[str, Any]):
        """Add one batch into rolling probe buffer (CPU-cloned)."""
        if not hasattr(self, "_h_probe_buffer"):
            return
        try:
            cloned: Dict[str, Any] = {}
            for k, v in inputs.items():
                if isinstance(v, torch.Tensor):
                    # store on CPU to keep GPU memory stable
                    cloned[k] = v.detach().cpu().clone()
                else:
                    cloned[k] = copy.deepcopy(v)
            self._h_probe_buffer.append(cloned)
        except Exception as e:
            if self.is_world_process_zero():
                logger.warning(f"[adaptive h] failed to buffer a batch: {e}")

    def _adaptive_h_eval_at(self, model, inputs: Dict[str, Any], h: float, seed: int, mult: int) -> float:
        """Evaluate f(theta + mult * h * z) and restore theta back."""
        # Make sure we have parameter list ready
        if not hasattr(self, "named_parameters_to_optim") or not self.named_parameters_to_optim:
            self.named_parameters_to_optim = [(n, p) for n, p in model.named_parameters() if p.requires_grad]

        self.zo_perturb_parameters(random_seed=seed, scaling_factor=mult, eps=h)
        loss = self.zo_forward(model, inputs)
        self.zo_perturb_parameters(random_seed=seed, scaling_factor=-mult, eps=h)
        return float(loss.item() if isinstance(loss, torch.Tensor) else loss)

    def _adaptive_h_estimate_eps_f(self, model, batches: List[Dict[str, Any]], q: int = 8, delta: float = 1e-6, trials: int = 6) -> float:
        """Estimate epsilon_f using a forward-difference table (More & Wild-style).

        We evaluate f(theta + i*delta*z) for i=0..q along a *fixed* Gaussian direction z (controlled by seed),
        build a forward-difference table, and use a higher-order difference column to estimate the noise scale.

        Returns
        -------
        epsilon_f_hat : float
            Estimated objective noise scale.
        """
        if len(batches) == 0:
            return 0.0

        # guard: q must be >= 3 because we use j=3
        q = int(q)
        if q < 3:
            q = 3
        delta = float(delta)
        if (not math.isfinite(delta)) or (delta <= 0.0):
            delta = 1e-6

        # Parameters for noise estimator (paper uses j=3)
        j = 3
        gamma = 1.0 / (2.0 * (2.0 * j + 1.0) * (j + 1.0) ** 2)

        vals: List[float] = []
        trials = max(int(trials), 1)

        for _ in range(trials):
            inputs = random.choice(batches)
            seed = int(np.random.randint(0, 1_000_000_000))

            # Evaluate f(theta + i*delta*z) for i=0..q using incremental perturbations
            f_vals: List[float] = []

            # i=0
            f0 = self.zo_forward(model, inputs)
            f_vals.append(float(f0.item()) if isinstance(f0, torch.Tensor) else float(f0))

            # i=1..q : theta <- theta + delta*z each step
            for _i in range(1, q + 1):
                self.zo_perturb_parameters(random_seed=seed, scaling_factor=1, eps=delta)
                fi = self.zo_forward(model, inputs)
                f_vals.append(float(fi.item()) if isinstance(fi, torch.Tensor) else float(fi))

            # restore theta back to original
            self.zo_perturb_parameters(random_seed=seed, scaling_factor=-q, eps=delta)

            # Build forward difference table (only keep diagonal entries we need)
            T = [f_vals]
            for k in range(1, q + 1):
                prev = T[k - 1]
                cur = [prev[i + 1] - prev[i] for i in range(len(prev) - 1)]
                T.append(cur)

            # Use higher-order difference statistic for noise estimate
            denom = float(q + 1 - j)
            if denom <= 0:
                continue

            # sum_{i=0}^{q-j} (T_{i,j})^2  where T[j][i] is j-th forward difference at position i
            try:
                sj2 = gamma / denom * sum((T[j][i] ** 2) for i in range(q + 1 - j))
            except Exception:
                continue

            if sj2 > 0 and math.isfinite(sj2):
                vals.append(math.sqrt(sj2))

        if not vals:
            return 0.0
        return float(np.median(np.asarray(vals, dtype=np.float64)))

    def _adaptive_h_nu3_tests(self, model, inputs: Dict[str, Any], h: float, seed: int, eps_f: float):
        """Run local-scale tests and estimate nu3 for a given probe h."""
        f0 = float(self.zo_forward(model, inputs).item())
        fp = self._adaptive_h_eval_at(model, inputs, h=h, seed=seed, mult=1)
        fm = self._adaptive_h_eval_at(model, inputs, h=h, seed=seed, mult=-1)

        # Paper-style SNR test uses Δ(h) = |f(-h) - 2 f0 + f(+h)|
        delta2 = abs(fm - 2.0 * f0 + fp)
        # Auxiliary (for debugging): D(h) = |f(+h)-f0| + |f(-h)-f0|
        D = abs(fp - f0) + abs(fm - f0)

        snr_val = delta2 / max(eps_f, 1e-30)
        snr_ok = snr_val >= self._h_tau1

        # Proximity: relative change should be small (local region)
        prox_plus = abs(fp - f0) / max(abs(f0), abs(fp), 1e-30)
        prox_minus = abs(fm - f0) / max(abs(f0), abs(fm), 1e-30)
        prox_ok = (prox_plus <= self._h_tau2) and (prox_minus <= self._h_tau2)

        # nu3 estimate via third difference
        f2 = self._adaptive_h_eval_at(model, inputs, h=h, seed=seed, mult=2)
        fm2 = self._adaptive_h_eval_at(model, inputs, h=h, seed=seed, mult=-2)
        delta3 = abs(-f2 + 2.0 * fp - 2.0 * fm + fm2)
        nu3_hat = delta3 / (2.0 * (h ** 3 + 1e-30))

        return {
            "snr_ok": snr_ok,
            "prox_ok": prox_ok,
            "nu3_hat": float(nu3_hat),
            "delta3": float(delta3),
            "snr_val": float(snr_val),
            "delta2": float(delta2),
            "D": float(D),
            "prox": (float(prox_plus), float(prox_minus)),
            "f0": float(f0),
        }

    def _adaptive_h_estimate_nu3(self, model, inputs: Dict[str, Any], eps_f: float, h_init: float, seed: int) -> \
    Optional[float]:
        """Estimate nu3 with simple scale-search driven by (snr, prox)."""
        h = float(h_init)
        h_min = float(getattr(self.args, "h_probe_min", 1e-6))
        h_max = float(getattr(self.args, "h_probe_max", 5e-1))
        max_tries = int(getattr(self.args, "h_probe_max_tries", 12))

        best = None
        for _ in range(max_tries):
            h = float(min(max(h, h_min), h_max))
            out = self._adaptive_h_nu3_tests(model, inputs, h=h, seed=seed, eps_f=eps_f)

            # Prefer shrinking when prox fails (h too large).
            if not out["prox_ok"]:
                h *= 0.5
                best = out
                continue
            if not out["snr_ok"]:
                h *= 2.0
                best = out
                continue

            best = out
            break

        if best is None:
            return None

        # IMPORTANT: 如果最终没有找到同时满足 (snr_ok & prox_ok) 的 h，丢弃该 direction
        if (not bool(best.get("snr_ok", False))) or (not bool(best.get("prox_ok", False))):
            return None

        nu3_hat = float(best.get("nu3_hat", 0.0))
        if not math.isfinite(nu3_hat) or nu3_hat <= 0.0:
            return None
        return nu3_hat

    def _adaptive_h_update_if_needed(self, model):
        """Periodically re-estimate (eps_f, nu3) and update adaptive_h."""
        if not getattr(self.args, "use_adaptive_h", False):
            return

        every = int(getattr(self.args, "update_noise_every", 0))
        if every <= 0:
            return

        gs = int(getattr(self.state, "global_step", 0))
        if gs <= 0 or (gs % every) != 0:
            return

        if getattr(self, "_h_last_update_step", -1) == gs:
            return
        self._h_last_update_step = gs

        # Choose probe batches from rolling buffer (random sampling)
        buf = list(getattr(self, "_h_probe_buffer", []))
        if not buf:
            return

        nb = max(int(getattr(self, "_h_nb", 1)), 1)
        nd = max(int(getattr(self, "_h_nd", 1)), 1)

        # eps_f: noise scale
        eps_f_hat = self._adaptive_h_estimate_eps_f(model, buf, q=8, delta=1e-6, trials=max(2 * nb, 6))

        # nu3: curvature/third-derivative scale (average over batches & directions)
        nu3_list: List[float] = []

        # Probe h init: smaller-scale than training eps. Heuristic: min(eps_train, eps_f^{1/5}).
        eps_train_now = self._get_current_zo_eps()
        h_probe_init = min(eps_train_now, max(1e-6, eps_f_hat ** 0.2))

        for _ in range(nb):
            batch = random.choice(buf)
            for _ in range(nd):
                seed = int(np.random.randint(0, 1_000_000_000))
                nu3_hat = self._adaptive_h_estimate_nu3(model, batch, eps_f=eps_f_hat, h_init=h_probe_init, seed=seed)
                if nu3_hat is not None and math.isfinite(nu3_hat):
                    nu3_list.append(float(nu3_hat))

        if not nu3_list:
            if self.is_world_process_zero():
                logger.warning(f"[adaptive h][update] step={gs} nu3_list empty -> keep h={self.adaptive_h:.3e}")
            return

        if self._h_reduce == "median":
            nu3_hat = float(np.median(np.asarray(nu3_list, dtype=np.float64)))
        else:
            nu3_hat = float(np.mean(np.asarray(nu3_list, dtype=np.float64)))

        # Guard: tiny nu3 makes eps_star explode
        if not math.isfinite(nu3_hat) or nu3_hat <= 1e-12:
            if self.is_world_process_zero():
                logger.warning(
                    f"[adaptive h][update] step={gs} nu3_hat={nu3_hat:.3e} too small/invalid -> keep h={self.adaptive_h:.3e}"
                )
            return

        # Compute eps* for training (central difference 1st-derivative MSE optimum):
        #   h* = alpha^{-1/6} * (3*eps_f/nu3)^{1/3}
        alpha = max(float(self._h_trunc_alpha), 1e-12)
        alpha_scale = alpha ** (-1.0 / 6.0)
        h_raw = alpha_scale * ((3.0 * max(eps_f_hat, 1e-30) / nu3_hat) ** (1.0 / 3.0))

        # Clamp (avoid runaway)
        h_min = float(getattr(self.args, "h_train_min", 1e-6))
        h_max = float(getattr(self.args, "h_train_max", 5e-2))
        h_raw = float(min(max(h_raw, h_min), h_max))

        # EMA smoothing
        h_old = float(self.adaptive_h)
        beta = float(self._h_beta)
        self.adaptive_h = float((1.0 - beta) * h_old + beta * h_raw)

        if self.is_world_process_zero():
            logger.info(
                f"[adaptive h][update] step={gs} h_raw={h_raw:.3e} -> h_ema={self.adaptive_h:.3e} "
                f"(eps_f={eps_f_hat:.3e}, nu3={nu3_hat:.3e}, nb={nb}, nd={nd}, buf={len(buf)}, reduce={self._h_reduce}, alpha={alpha})"
            )

    def _inner_training_loop(
            self, batch_size=None, args=None, resume_from_checkpoint=None, trial=None, ignore_keys_for_eval=None
    ):
        """
        We overload the original training loop to add linear probing and MeZO. Search key word "MeZO added"
        for those updates.
        """
        self._train_batch_size = batch_size
        # Data loader and number of training steps
        train_dataloader = self.get_train_dataloader()

        # --- Inspect sampler type (RandomSampler / SequentialSampler / DistributedSampler) ---
        try:
            sampler = getattr(train_dataloader, "sampler", None)
            batch_sampler = getattr(train_dataloader, "batch_sampler", None)
            logger.info(f"[dataloader] sampler={type(sampler).__name__}, batch_sampler={type(batch_sampler).__name__}")
        except Exception:
            pass

        # --- Adaptive h init (only when running MeZO) ---
        is_mezo = (getattr(self.args, "trainer", None) == "zo") or bool(getattr(self.args, "zero_order_optim", False))
        if is_mezo and getattr(self.args, "use_adaptive_h", False):
            self._adaptive_h_init()

        # MeZO added: Linear probing
        if self.args.linear_probing:

            def _get_token_prediction_layer(model):
                if model.config.model_type == "opt":
                    return model.lm_head
                else:
                    raise NotImplementedError(model.config.model_type)

            def _extract_features(model, *args, **kwargs):
                """some magic for getting features pre last layer"""
                features = {}

                def __hook(model_, input_, output_):
                    features["features"] = input_[0].detach()

                _get_token_prediction_layer(model).register_forward_hook(__hook)
                model.forward(*args, **kwargs)
                return features["features"]

            logger.info("Linear probing")
            logger.info("Starting to get features for training dataset")
            targets = []
            features = []
            with torch.inference_mode():
                for step, inputs in enumerate(tqdm(train_dataloader)):
                    for k, v in inputs.items():
                        if isinstance(v, torch.Tensor):
                            inputs[k] = v.to(self.model.device)

                    feature = _extract_features(self.model, **inputs)
                    target = inputs["labels"]

                    # Shift the target (bc it's autoregressive LM) and add the corresponding part
                    assert not self.args.train_as_classification and self.args.only_train_option
                    feature, target = feature[:, :-1], target[:, 1:]
                    for _i, _len in enumerate(inputs["option_len"]):
                        features.append(feature[_i, -_len:])
                        targets.append(target[_i, -_len:])

            logger.info("Finished getting features for training dataset")

            features = torch.cat(features, dim=0).cpu().numpy()
            targets = torch.cat(targets, dim=0).cpu().numpy()
            # Whether to use bias
            if self.model.config.model_type in ["opt", "gpt2"]:
                use_bias = False
            else:
                raise NotImplementedError
            # Set early stopping
            tol = 0.01 if self.args.lp_early_stopping else 1e-4  # 1e-4 is scipy default
            max_iter = 1000 if self.args.lp_early_stopping else 5000

            logger.info("Fitting logistic regression...")
            reg = LogisticRegressionCV(max_iter=max_iter, fit_intercept=use_bias, multi_class="multinomial",
                                       random_state=0, tol=tol, n_jobs=-1).fit(features, targets)
            logger.info("Done")

            logger.info("Assigning weights to model")
            decoder = _get_token_prediction_layer(self.model)
            coef_torch = torch.tensor(reg.coef_, device=decoder.weight.device, dtype=decoder.weight.dtype)
            if use_bias:
                bias_torch = torch.tensor(reg.intercept_, device=decoder.weight.device, dtype=decoder.weight.dtype)
            if coef_torch.shape[0] == 1:  # The regressor only detects two classes
                assert len(reg.classes_) == 2
                coef_torch = torch.cat([-coef_torch / 2, coef_torch / 2], dim=0)
                if use_bias:
                    bias_torch = torch.cat([-bias_torch / 2, bias_torch / 2], dim=0)

            for _i, token_id in enumerate(reg.classes_):
                decoder.weight.data[token_id] = coef_torch[_i]
                if use_bias:
                    decoder.bias.data[token_id] = bias_torch[_i]

            return None

        # Setting up training control variables:
        # number of training epochs: num_train_epochs
        # number of training steps per epoch: num_update_steps_per_epoch
        # total number of training steps to execute: max_steps
        total_train_batch_size = args.train_batch_size * args.gradient_accumulation_steps * args.world_size

        len_dataloader = None
        if has_length(train_dataloader):
            len_dataloader = len(train_dataloader)
            num_update_steps_per_epoch = len_dataloader // args.gradient_accumulation_steps
            num_update_steps_per_epoch = max(num_update_steps_per_epoch, 1)
            num_examples = self.num_examples(train_dataloader)
            if args.max_steps > 0:
                max_steps = args.max_steps
                num_train_epochs = args.max_steps // num_update_steps_per_epoch + int(
                    args.max_steps % num_update_steps_per_epoch > 0
                )
                # May be slightly incorrect if the last batch in the training dataloader has a smaller size but it's
                # the best we can do.
                num_train_samples = args.max_steps * total_train_batch_size
            else:
                max_steps = math.ceil(args.num_train_epochs * num_update_steps_per_epoch)
                num_train_epochs = math.ceil(args.num_train_epochs)
                num_train_samples = self.num_examples(train_dataloader) * args.num_train_epochs
        elif args.max_steps > 0:  # Rely on max_steps when dataloader does not have a working size
            max_steps = args.max_steps
            # Setting a very large number of epochs so we go as many times as necessary over the iterator.
            num_train_epochs = sys.maxsize
            num_update_steps_per_epoch = max_steps
            num_examples = total_train_batch_size * args.max_steps
            num_train_samples = args.max_steps * total_train_batch_size
        else:
            raise ValueError(
                "args.max_steps must be set to a positive value if dataloader does not have a length, was"
                f" {args.max_steps}"
            )

        if DebugOption.UNDERFLOW_OVERFLOW in self.args.debug:
            if self.args.n_gpu > 1:
                # nn.DataParallel(model) replicates the model, creating new variables and module
                # references registered here no longer work on other gpus, breaking the module
                raise ValueError(
                    "Currently --debug underflow_overflow is not supported under DP. Please use DDP"
                    " (torch.distributed.launch)."
                )
            else:
                debug_overflow = DebugUnderflowOverflow(self.model)  # noqa

        delay_optimizer_creation = (
                self.sharded_ddp is not None
                and self.sharded_ddp != ShardedDDPOption.SIMPLE
                or is_sagemaker_mp_enabled()
                or self.fsdp is not None
        )
        if args.deepspeed:
            deepspeed_engine, optimizer, lr_scheduler = deepspeed_init(
                self, num_training_steps=max_steps, resume_from_checkpoint=resume_from_checkpoint
            )
            self.model = deepspeed_engine.module
            self.model_wrapped = deepspeed_engine
            self.deepspeed = deepspeed_engine
            self.optimizer = optimizer
            self.lr_scheduler = lr_scheduler
        elif not delay_optimizer_creation:
            self.create_optimizer_and_scheduler(num_training_steps=max_steps)

        self.state = TrainerState()
        self.state.is_hyper_param_search = trial is not None

        # Activate gradient checkpointing if needed
        if args.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

        model = self._wrap_model(self.model_wrapped)

        if is_sagemaker_mp_enabled() and resume_from_checkpoint is not None:
            self._load_from_checkpoint(resume_from_checkpoint, model)

        # for the rest of this function `model` is the outside model, whether it was wrapped or not
        if model is not self.model:
            self.model_wrapped = model

        if delay_optimizer_creation:
            self.create_optimizer_and_scheduler(num_training_steps=max_steps)

        # Check if saved optimizer or scheduler states exist
        self._load_optimizer_and_scheduler(resume_from_checkpoint)

        # important: at this point:
        # self.model         is the Transformers Model
        # self.model_wrapped is DDP(Transformers Model), Deepspeed(Transformers Model), etc.

        # Train!
        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {num_examples}")
        logger.info(f"  Num Epochs = {num_train_epochs}")
        logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size}")
        logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {max_steps}")
        logger.info(
            f"  Number of trainable parameters = {sum(p.numel() for p in model.parameters() if p.requires_grad)}"
        )

        self.state.epoch = 0
        start_time = time.time()
        epochs_trained = 0
        steps_trained_in_current_epoch = 0
        steps_trained_progress_bar = None

        # Check if continuing training from a checkpoint
        if resume_from_checkpoint is not None and os.path.isfile(
                os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME)
        ):
            self.state = TrainerState.load_from_json(os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME))
            epochs_trained = self.state.global_step // num_update_steps_per_epoch
            if not args.ignore_data_skip:
                steps_trained_in_current_epoch = self.state.global_step % (num_update_steps_per_epoch)
                steps_trained_in_current_epoch *= args.gradient_accumulation_steps
            else:
                steps_trained_in_current_epoch = 0

            logger.info("  Continuing training from checkpoint, will skip to saved global_step")
            logger.info(f"  Continuing training from epoch {epochs_trained}")
            logger.info(f"  Continuing training from global step {self.state.global_step}")
            if not args.ignore_data_skip:
                logger.info(
                    f"  Will skip the first {epochs_trained} epochs then the first {steps_trained_in_current_epoch} "
                    "batches in the first epoch. If this takes a lot of time, you can add the `--ignore_data_skip` "
                    "flag to your launch command, but you will resume the training on data already seen by your model."
                )
                if self.is_local_process_zero() and not args.disable_tqdm:
                    steps_trained_progress_bar = tqdm(total=steps_trained_in_current_epoch)
                    steps_trained_progress_bar.set_description("Skipping the first batches")

        # Update the references
        self.callback_handler.model = self.model
        self.callback_handler.optimizer = self.optimizer
        self.callback_handler.lr_scheduler = self.lr_scheduler
        self.callback_handler.train_dataloader = train_dataloader
        if self.hp_name is not None and self._trial is not None:
            # use self._trial because the SigOpt/Optuna hpo only call `_hp_search_setup(trial)` instead of passing trial
            # parameter to Train when using DDP.
            self.state.trial_name = self.hp_name(self._trial)
        if trial is not None:
            assignments = trial.assignments if self.hp_search_backend == HPSearchBackend.SIGOPT else trial
            self.state.trial_params = hp_params(assignments)
        else:
            self.state.trial_params = None
        # This should be the same if the state has been saved but in case the training arguments changed, it's safer
        # to set this after the load.
        self.state.max_steps = max_steps
        self.state.num_train_epochs = num_train_epochs
        self.state.is_local_process_zero = self.is_local_process_zero()
        self.state.is_world_process_zero = self.is_world_process_zero()

        # tr_loss is a tensor to avoid synchronization of TPUs through .item()
        tr_loss = torch.tensor(0.0).to(args.device)
        # _total_loss_scalar is updated everytime .item() has to be called on tr_loss and stores the sum of all losses
        self._total_loss_scalar = 0.0
        self._globalstep_last_logged = self.state.global_step
        model.zero_grad()

        self.control = self.callback_handler.on_train_begin(args, self.state, self.control)

        # Skip the first epochs_trained epochs to get the random state of the dataloader at the right point.
        if not args.ignore_data_skip:
            for epoch in range(epochs_trained):
                is_random_sampler = hasattr(train_dataloader, "sampler") and isinstance(
                    train_dataloader.sampler, RandomSampler
                )
                if is_torch_less_than_1_11 or not is_random_sampler:
                    # We just need to begin an iteration to create the randomization of the sampler.
                    # That was before PyTorch 1.11 however...
                    for _ in train_dataloader:
                        break
                else:
                    # Otherwise we need to call the whooooole sampler cause there is some random operation added
                    # AT THE VERY END!
                    _ = list(train_dataloader.sampler)

        for epoch in range(epochs_trained, num_train_epochs):
            if isinstance(train_dataloader, DataLoader) and isinstance(train_dataloader.sampler, DistributedSampler):
                train_dataloader.sampler.set_epoch(epoch)
            elif hasattr(train_dataloader, "dataset") and isinstance(train_dataloader.dataset, IterableDatasetShard):
                train_dataloader.dataset.set_epoch(epoch)

            if is_torch_tpu_available():
                parallel_loader = pl.ParallelLoader(train_dataloader, [args.device]).per_device_loader(args.device)
                epoch_iterator = parallel_loader
            else:
                epoch_iterator = train_dataloader

            # Reset the past mems state at the beginning of each epoch if necessary.
            if args.past_index >= 0:
                self._past = None

            steps_in_epoch = (
                len(epoch_iterator)
                if len_dataloader is not None
                else args.max_steps * args.gradient_accumulation_steps
            )
            self.control = self.callback_handler.on_epoch_begin(args, self.state, self.control)

            if epoch == epochs_trained and resume_from_checkpoint is not None and steps_trained_in_current_epoch == 0:
                self._load_rng_state(resume_from_checkpoint)

            step = -1
            for step, inputs in enumerate(epoch_iterator):

                # Skip past any already trained steps if resuming training
                if steps_trained_in_current_epoch > 0:
                    steps_trained_in_current_epoch -= 1
                    if steps_trained_progress_bar is not None:
                        steps_trained_progress_bar.update(1)
                    if steps_trained_in_current_epoch == 0:
                        self._load_rng_state(resume_from_checkpoint)
                    continue
                elif steps_trained_progress_bar is not None:
                    steps_trained_progress_bar.close()
                    steps_trained_progress_bar = None
                if step % args.gradient_accumulation_steps == 0:
                    self.control = self.callback_handler.on_step_begin(args, self.state, self.control)

                # # ===== DEBUG 2025-11-08: 记录本 step batch size 开始 =====
                # try:
                #     debug_batch_size = None
                #     for _v in inputs.values():
                #         if isinstance(_v, torch.Tensor):
                #             debug_batch_size = _v.size(0)
                #             break
                #     if debug_batch_size is not None and self.is_world_process_zero():
                #         logger.info(f"[DEBUG] step={step}, global_step={self.state.global_step}, batch_size={debug_batch_size}")
                # except Exception as e:
                #     if self.is_world_process_zero():
                #         logger.warning(f"[DEBUG] 计算 batch size 失败: step={step}, err={e}")
                # # ===== DEBUG 2025-11-08: 记录本 step batch size 结束 =====

                # --- Adaptive h: rolling buffer + periodic update (随机抽样) ---
                if is_mezo and getattr(args, "use_adaptive_h", False):
                    self._adaptive_h_buffer_add(inputs)
                    self._adaptive_h_update_if_needed(model)

                # MeZO added: estimate gradient
                if is_mezo:
                    tr_loss_step = self.zo_step(model, inputs)
                else:
                    if (
                            ((step + 1) % args.gradient_accumulation_steps != 0)
                            and args.local_rank != -1
                            and args._no_sync_in_gradient_accumulation
                    ):
                        # Avoid unnecessary DDP synchronization since there will be no backward pass on this example.
                        with model.no_sync():
                            tr_loss_step = self.training_step(model, inputs)
                    else:
                        tr_loss_step = self.training_step(model, inputs)

                # # ===== DEBUG 2025-11-08: 记录本 step loss 开始 =====
                # if self.is_world_process_zero():
                #     try:
                #         if isinstance(tr_loss_step, torch.Tensor):
                #             debug_loss_value = tr_loss_step.detach().item()
                #         else:
                #             debug_loss_value = float(tr_loss_step)
                #         logger.info(f"[DEBUG] step={step}, global_step={self.state.global_step}, step_loss={debug_loss_value}")
                #     except Exception as e:
                #         logger.warning(f"[DEBUG] 记录 step loss 失败: step={step}, err={e}")
                # # ===== DEBUG 2025-11-08: 记录本 step loss 结束 =====

                if (
                        args.logging_nan_inf_filter
                        and not is_torch_tpu_available()
                        and (torch.isnan(tr_loss_step) or torch.isinf(tr_loss_step))
                ):
                    # if loss is nan or inf simply add the average of previous logged losses
                    tr_loss += tr_loss / (1 + self.state.global_step - self._globalstep_last_logged)
                else:
                    tr_loss += tr_loss_step

                self.current_flos += float(self.floating_point_ops(inputs))

                # Optimizer step for deepspeed must be called on every step regardless of the value of gradient_accumulation_steps
                if self.deepspeed:
                    self.deepspeed.step()

                if (step + 1) % args.gradient_accumulation_steps == 0 or (
                        # last step in epoch but step is always smaller than gradient_accumulation_steps
                        steps_in_epoch <= args.gradient_accumulation_steps
                        and (step + 1) == steps_in_epoch
                ):
                    # MeZO added: update model with the estimated gradient
                    if is_mezo:
                        self.zo_update(model)
                    else:
                        # Gradient clipping
                        if args.max_grad_norm is not None and args.max_grad_norm > 0 and not self.deepspeed:
                            # deepspeed does its own clipping

                            if self.do_grad_scaling:
                                # Reduce gradients first for XLA
                                if is_torch_tpu_available():
                                    gradients = xm._fetch_gradients(self.optimizer)
                                    xm.all_reduce("sum", gradients, scale=1.0 / xm.xrt_world_size())
                                # AMP: gradients need unscaling
                                self.scaler.unscale_(self.optimizer)

                            if is_sagemaker_mp_enabled() and args.fp16:
                                self.optimizer.clip_master_grads(args.max_grad_norm)
                            elif hasattr(self.optimizer, "clip_grad_norm"):
                                # Some optimizers (like the sharded optimizer) have a specific way to do gradient clipping
                                self.optimizer.clip_grad_norm(args.max_grad_norm)
                            elif hasattr(model, "clip_grad_norm_"):
                                # Some models (like FullyShardedDDP) have a specific way to do gradient clipping
                                model.clip_grad_norm_(args.max_grad_norm)
                            else:
                                # Revert to normal clipping otherwise, handling Apex or full precision
                                nn.utils.clip_grad_norm_(
                                    amp.master_params(self.optimizer) if self.use_apex else model.parameters(),
                                    args.max_grad_norm,
                                )

                        # Optimizer step
                        optimizer_was_run = True
                        if self.deepspeed:
                            pass  # called outside the loop
                        elif is_torch_tpu_available():
                            if self.do_grad_scaling:
                                self.scaler.step(self.optimizer)
                                self.scaler.update()
                            else:
                                xm.optimizer_step(self.optimizer)
                        elif self.do_grad_scaling:
                            scale_before = self.scaler.get_scale()
                            self.scaler.step(self.optimizer)
                            self.scaler.update()
                            scale_after = self.scaler.get_scale()
                            optimizer_was_run = scale_before <= scale_after
                        else:
                            self.optimizer.step()

                        if optimizer_was_run and not self.deepspeed:
                            self.lr_scheduler.step()
                        model.zero_grad()

                    self.state.global_step += 1
                    self.state.epoch = epoch + (step + 1) / steps_in_epoch
                    self.control = self.callback_handler.on_step_end(args, self.state, self.control)

                    # # ===== DEBUG 2025-11-08: 调用 _maybe_log_save_evaluate 前记录样本数(按 step 粗略估计) 开始 =====
                    # if self.is_world_process_zero():
                    #     try:
                    #         debug_seen_samples = self.state.global_step * total_train_batch_size
                    #         logger.info(f"[DEBUG] 调用 _maybe_log_save_evaluate(step) 时 global_step={self.state.global_step}, approx_seen_samples={debug_seen_samples}")
                    #     except Exception as e:
                    #         logger.warning(f"[DEBUG] 计算 approx_seen_samples 失败(step): err={e}")
                    # # ===== DEBUG 2025-11-08: 调用 _maybe_log_save_evaluate 前记录样本数(按 step 粗略估计) 结束 =====

                    self._maybe_log_save_evaluate(tr_loss, model, trial, epoch, ignore_keys_for_eval)
                else:
                    self.control = self.callback_handler.on_substep_end(args, self.state, self.control)

                if self.control.should_epoch_stop or self.control.should_training_stop:
                    break
            if step < 0:
                logger.warning(
                    "There seems to be not a single sample in your epoch_iterator, stopping training at step"
                    f" {self.state.global_step}! This is expected if you're using an IterableDataset and set"
                    f" num_steps ({max_steps}) higher than the number of available samples."
                )
                self.control.should_training_stop = True

            self.control = self.callback_handler.on_epoch_end(args, self.state, self.control)

            # # ===== DEBUG 2025-11-08: 调用 _maybe_log_save_evaluate 前记录样本数(按 epoch 粗略估计) 开始 =====
            # if self.is_world_process_zero():
            #     try:
            #         debug_seen_samples = self.state.global_step * total_train_batch_size
            #         logger.info(f"[DEBUG] 调用 _maybe_log_save_evaluate(epoch_end) 时 global_step={self.state.global_step}, approx_seen_samples={debug_seen_samples}")
            #     except Exception as e:
            #         logger.warning(f"[DEBUG] 计算 approx_seen_samples 失败(epoch_end): err={e}")
            # # ===== DEBUG 2025-11-08: 调用 _maybe_log_save_evaluate 前记录样本数(按 epoch 粗略估计) 结束 =====

            self._maybe_log_save_evaluate(tr_loss, model, trial, epoch, ignore_keys_for_eval)

            if DebugOption.TPU_METRICS_DEBUG in self.args.debug:
                if is_torch_tpu_available():
                    # tpu-comment: Logging debug metrics for PyTorch/XLA (compile, execute times, ops, etc.)
                    xm.master_print(met.metrics_report())
                else:
                    logger.warning(
                        "You enabled PyTorch/XLA debug metrics but you don't have a TPU "
                        "configured. Check your training configuration if this is unexpected."
                    )
            if self.control.should_training_stop:
                break

        if args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of training
            delattr(self, "_past")

        logger.info("\n\nTraining completed. Do not forget to share your model on huggingface.co/models =)\n\n")
        if args.load_best_model_at_end and self.state.best_model_checkpoint is not None:
            # Wait for everyone to get here so we are sur the model has been saved by process 0.
            if is_torch_tpu_available():
                xm.rendezvous("load_best_model_at_end")
            elif args.local_rank != -1:
                dist.barrier()
            elif is_sagemaker_mp_enabled():
                smp.barrier()

            self._load_best_model()

        # add remaining tr_loss
        self._total_loss_scalar += tr_loss.item()
        train_loss = self._total_loss_scalar / self.state.global_step

        metrics = speed_metrics("train", start_time, num_samples=num_train_samples, num_steps=self.state.max_steps)
        self.store_flos()
        metrics["total_flos"] = self.state.total_flos
        metrics["train_loss"] = train_loss

        self.is_in_train = False

        self._memory_tracker.stop_and_update_metrics(metrics)

        self.log(metrics)

        run_dir = self._get_output_dir(trial)
        checkpoints_sorted = self._sorted_checkpoints(use_mtime=False, output_dir=run_dir)

        # Delete the last checkpoint when save_total_limit=1 if it's different from the best checkpoint.
        if self.state.best_model_checkpoint is not None and self.args.save_total_limit == 1:
            for checkpoint in checkpoints_sorted:
                if checkpoint != self.state.best_model_checkpoint:
                    logger.info(f"Deleting older checkpoint [{checkpoint}] due to args.save_total_limit")
                    shutil.rmtree(checkpoint)

        self.control = self.callback_handler.on_train_end(args, self.state, self.control)

        return TrainOutput(self.state.global_step, train_loss, metrics)

    ############## MeZO ##############

    def zo_perturb_parameters(self, random_seed=None, scaling_factor=1, eps: Optional[float] = None):
        """
        Perturb the parameters with random vector z.
        Input:
        - random_seed: random seed for MeZO in-place perturbation (if it's None, we will use self.zo_random_seed)
        - scaling_factor: theta = theta + scaling_factor * z * eps
        """

        # Set the random seed to ensure that we sample the same z for perturbation/update
        torch.manual_seed(random_seed if random_seed is not None else self.zo_random_seed)

        # eps may be dynamically adjusted (adaptive h)
        if eps is None:
            eps = self._get_current_zo_eps()
        eps = float(eps)

        for name, param in self.named_parameters_to_optim:
            z = torch.normal(mean=0, std=1, size=param.data.size(), device=param.data.device, dtype=param.data.dtype)
            param.data = param.data + scaling_factor * z * eps

    def zo_forward(self, model, inputs):
        """
        Get (no gradient) loss from the model. Dropout is turned off too.
        """
        model.eval()
        if self.args.non_diff:
            # Non-differentiable objective (may require autoregressive generation)
            return self.zo_forward_nondiff(model, inputs)

        with torch.inference_mode():
            inputs = self._prepare_inputs(inputs)
            with self.compute_loss_context_manager():
                loss = self.compute_loss(model, inputs)
            if self.args.n_gpu > 1:
                # Warning: this is copied from the original Huggingface Trainer. Untested.
                loss = loss.mean()  # mean() to average on multi-gpu parallel training

        # # ===== DEBUG 2025-11-08: 记录 zo_forward loss 开始 =====
        # if self.is_world_process_zero():
        #     try:
        #         logger.info(f"[DEBUG] zo_forward loss={loss.detach().item()}")
        #     except Exception as e:
        #         logger.warning(f"[DEBUG] 记录 zo_forward loss 失败: err={e}")
        # # ===== DEBUG 2025-11-08: 记录 zo_forward loss 结束 =====

        return loss.detach()

    def zo_forward_nondiff(self, model, inputs):
        """
        Get (no gradient) non-diffiable loss from the model.
        """
        model.eval()
        assert self.args.task_name == "SQuAD", "Non differentiable objective only supports SQuAD for now."

        with torch.inference_mode():
            inputs = self._prepare_inputs(inputs)
            args = self.args
            outputs = self.model.generate(
                inputs["input_ids"], do_sample=args.sampling, temperature=args.temperature,
                num_beams=args.num_beams, top_p=args.top_p, top_k=args.top_k,
                max_new_tokens=min(args.max_new_tokens, args.max_length - inputs["input_ids"].size(1)),
                num_return_sequences=1,
                eos_token_id=[self.tokenizer.encode(args.eos_token, add_special_tokens=False)[-1],
                              self.tokenizer.eos_token_id],
            )
            output_text = []
            for i in range(len(outputs)):
                output_text.append(
                    self.tokenizer.decode(outputs[i][inputs["input_ids"].size(1):], skip_special_tokens=True).strip())
            f1s = [_simple_f1(output_text[i], inputs['gold'][i]) for i in range(len(output_text))]

        # # ===== DEBUG 2025-11-08: 记录 zo_forward_nondiff mean F1 开始 =====
        # debug_mean_f1 = np.mean(f1s)
        # if self.is_world_process_zero():
        #     try:
        #         logger.info(f"[DEBUG] zo_forward_nondiff mean_F1={debug_mean_f1}")
        #     except Exception as e:
        #         logger.warning(f"[DEBUG] 记录 zo_forward_nondiff mean F1 失败: err={e}")
        # # ===== DEBUG 2025-11-08: 记录 zo_forward_nondiff mean F1 结束 =====

        # 计算 mean F1（之前 debug_mean_f1 被注释掉会导致 NameError）
        mean_f1 = float(np.mean(f1s)) if len(f1s) > 0 else 0.0
        device = self.model.device if hasattr(self.model, "device") else torch.device("cpu")
        return -torch.tensor(mean_f1, dtype=torch.float32, device=device)

    def zo_step(self, model, inputs):
        """
        Estimate gradient by MeZO. Return the loss from f(theta + z)
        """
        args = self.args

        # What parameters to optimize
        self.named_parameters_to_optim = []
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.named_parameters_to_optim.append((name, param))

        # Sample the random seed for sampling z
        self.zo_random_seed = np.random.randint(1000000000)

        # eps can be dynamically adjusted (adaptive h)
        eps = self._get_current_zo_eps()
        self._last_zo_eps = float(eps)

        # First function evaluation
        self.zo_perturb_parameters(scaling_factor=1, eps=eps)
        loss1 = self.zo_forward(model, inputs)

        # Second function evaluation
        self.zo_perturb_parameters(scaling_factor=-2, eps=eps)
        loss2 = self.zo_forward(model, inputs)

        self.projected_grad = ((loss1 - loss2) / (2 * eps)).item()

        # # ===== DEBUG 2025-11-08: 记录 MeZO 两次 loss 和 projected_grad 开始 =====
        # if self.is_world_process_zero():
        #     try:
        #         l1 = loss1.item() if isinstance(loss1, torch.Tensor) else float(loss1)
        #         l2 = loss2.item() if isinstance(loss2, torch.Tensor) else float(loss2)
        #         logger.info(f"[DEBUG] zo_step loss1={l1}, loss2={l2}, projected_grad={self.projected_grad}, zo_eps={self.args.zo_eps}")
        #     except Exception as e:
        #         logger.warning(f"[DEBUG] 记录 zo_step 信息失败: err={e}")
        # # ===== DEBUG 2025-11-08: 记录 MeZO 两次 loss 和 projected_grad 结束 =====

        # No gradient accumulation support
        assert self.args.gradient_accumulation_steps == 1

        # Reset model back to its parameters at start of step
        self.zo_perturb_parameters(scaling_factor=1, eps=eps)

        return loss1

    def zo_update(self, model):
        """
        Update the parameters with the estimated gradients.
        """
        args = self.args

        # Reset the random seed for sampling zs
        torch.manual_seed(self.zo_random_seed)

        for name, param in self.named_parameters_to_optim:
            # Resample z
            z = torch.normal(mean=0, std=1, size=param.data.size(), device=param.data.device, dtype=param.data.dtype)
            if "bias" not in name and "layer_norm" not in name and "layernorm" not in name:
                param.data = param.data - self._get_learning_rate() * (
                            self.projected_grad * z + args.weight_decay * param.data)
            else:
                param.data = param.data - self._get_learning_rate() * (self.projected_grad * z)

        self.lr_scheduler.step()

    ############## Misc overload functions ##############

    def _set_signature_columns_if_needed(self):
        """
        We overload this function for non-differentiable objective training to pass "gold" -- the gold text for the task
        """
        if self._signature_columns is None:
            # Inspect model forward signature to keep only the arguments it accepts.
            signature = inspect.signature(self.model.forward)
            self._signature_columns = list(signature.parameters.keys())
            # Labels may be named label or label_ids, the default data collator handles that.
            self._signature_columns += list(set(["label", "label_ids"] + self.label_names))
            self._signature_columns += ["gold"]

    def save_model(self, output_dir: Optional[str] = None, _internal_call: bool = False):
        """
        We overload this function to fix an FSDP saving bug (before fix, it will likely cause OOM)
        """

        if output_dir is None:
            output_dir = self.args.output_dir

        if is_torch_tpu_available():
            self._save_tpu(output_dir)
        elif is_sagemaker_mp_enabled():
            # Calling the state_dict needs to be done on the wrapped model and on all processes.
            os.makedirs(output_dir, exist_ok=True)
            state_dict = self.model_wrapped.state_dict()
            if self.args.should_save:
                self._save(output_dir, state_dict=state_dict)
            if IS_SAGEMAKER_MP_POST_1_10:
                # 'user_content.pt' indicates model state_dict saved with smp >= 1.10
                Path(os.path.join(output_dir, "user_content.pt")).touch()
        elif (
                ShardedDDPOption.ZERO_DP_2 in self.args.sharded_ddp
                or ShardedDDPOption.ZERO_DP_3 in self.args.sharded_ddp
                or self.fsdp is not None
        ):
            from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, StateDictType, FullStateDictConfig
            full_state_dict_config = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)

            # Fix the FSDP loading bug
            with FSDP.state_dict_type(self.model, StateDictType.FULL_STATE_DICT, full_state_dict_config):
                state_dict = self.model.state_dict()
            # state_dict = self.model.state_dict()

            if self.args.should_save:
                self._save(output_dir, state_dict=state_dict)
        elif self.deepspeed:
            # this takes care of everything as long as we aren't under zero3
            if self.args.should_save:
                self._save(output_dir)

            if is_deepspeed_zero3_enabled():
                # It's too complicated to try to override different places where the weights dump gets
                # saved, so since under zero3 the file is bogus, simply delete it. The user should
                # either user deepspeed checkpoint to resume or to recover full weights use
                # zero_to_fp32.py stored in the checkpoint.
                if self.args.should_save:
                    file = os.path.join(output_dir, WEIGHTS_NAME)
                    if os.path.isfile(file):
                        # logger.info(f"deepspeed zero3: removing {file}, see zero_to_fp32.py to recover weights")
                        os.remove(file)

                # now save the real model if stage3_gather_16bit_weights_on_model_save=True
                # if false it will not be saved.
                # This must be called on all ranks
                if not self.deepspeed.save_16bit_model(output_dir, WEIGHTS_NAME):
                    logger.warning(
                        "deepspeed.save_16bit_model didn't save the model, since"
                        " stage3_gather_16bit_weights_on_model_save=false. Saving the full checkpoint instead, use"
                        " zero_to_fp32.py to recover weights"
                    )
                    self.deepspeed.save_checkpoint(output_dir)

        elif self.args.should_save:
            self._save(output_dir)

        # Push to the Hub when `save_model` is called by the user.
        if self.args.push_to_hub and not _internal_call:
            self.push_to_hub(commit_message="Model save")


# -----------------------------------------------------------------------------
# Backwards-compatible export: the entrypoint imports `Trainer` from this module.
# -----------------------------------------------------------------------------
Trainer = OurTrainer
