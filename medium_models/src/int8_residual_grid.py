import math
from typing import Dict, Iterable, Optional, Tuple, Union

import torch
from torch import nn


RESIDUAL_DTYPE_MAP = {
    "fp16": torch.float16,
    "float16": torch.float16,
    "bf16": torch.bfloat16,
    "bfloat16": torch.bfloat16,
    "fp32": torch.float32,
    "float32": torch.float32,
}


def normalize_residual_dtype(value: str) -> torch.dtype:
    key = str(value or "fp32").strip().lower()
    if key not in RESIDUAL_DTYPE_MAP:
        raise ValueError(f"Unsupported residual_dtype={value!r}. Expected one of fp16, bf16, fp32.")
    return RESIDUAL_DTYPE_MAP[key]


def normalize_commit_mode(value: str) -> str:
    mode = str(value or "round").strip().lower()
    if mode not in {"round", "floor", "stochastic"}:
        raise ValueError(f"Unsupported residual_commit_mode={value!r}. Expected round, floor, or stochastic.")
    return mode


def normalize_commit_select(value: str) -> str:
    mode = str(value or "all").strip().lower()
    if mode not in {"all", "top_abs_acc", "norm_budget"}:
        raise ValueError(f"Unsupported residual_commit_select={value!r}. Expected all, top_abs_acc, or norm_budget.")
    return mode


def normalize_budget_reference(value: str) -> str:
    ref = str(value or "acc").strip().lower()
    if ref not in {"delta", "acc"}:
        raise ValueError(f"Unsupported residual_budget_reference={value!r}. Expected delta or acc.")
    return ref


def normalize_scale_mode(value: str) -> str:
    mode = str(value or "tensor").strip().lower()
    if mode not in {"tensor", "channel", "block"}:
        raise ValueError(f"Unsupported residual_scale_mode={value!r}. Expected tensor, channel, or block.")
    return mode


class ResidualGridUpdater:
    """Error-feedback updater for fake-quantized symmetric low-bit parameter grids."""

    def __init__(
        self,
        named_parameters: Iterable[Tuple[str, nn.Parameter]],
        *,
        bits: int = 8,
        residual_dtype: str = "fp32",
        commit_mode: str = "round",
        max_code_step: int = 0,
        freeze_scale: bool = True,
        scale_floor: float = 0.0,
        commit_threshold: float = 0.0,
        commit_select: str = "all",
        target_active_frac: float = 0.0,
        actual_norm_ratio_cap: float = 0.0,
        budget_reference: str = "acc",
        residual_decay: float = 1.0,
        scale_mode: str = "tensor",
        block_size: int = 0,
    ) -> None:
        self.bits = int(bits)
        if self.bits not in {8, 4}:
            raise ValueError(f"ResidualGridUpdater only supports low-bit QuZO bits 8/4, got {bits}.")
        self.qmax = float((1 << (self.bits - 1)) - 1)
        self.qmin = -self.qmax
        self.residual_dtype = normalize_residual_dtype(residual_dtype)
        self.commit_mode = normalize_commit_mode(commit_mode)
        self.max_code_step = max(0, int(max_code_step))
        self.freeze_scale = bool(freeze_scale)
        self.scale_floor = max(0.0, float(scale_floor or 0.0))
        self.commit_threshold = max(0.0, float(commit_threshold or 0.0))
        self.commit_select = normalize_commit_select(commit_select)
        self.target_active_frac = max(0.0, float(target_active_frac or 0.0))
        self.actual_norm_ratio_cap = max(0.0, float(actual_norm_ratio_cap or 0.0))
        self.budget_reference = normalize_budget_reference(budget_reference)
        self.residual_decay = float(residual_decay if residual_decay is not None else 1.0)
        if (not math.isfinite(self.residual_decay)) or self.residual_decay < 0.0:
            raise ValueError(f"residual_decay must be a finite non-negative float, got {residual_decay!r}.")
        self.scale_mode = normalize_scale_mode(scale_mode)
        self.block_size = int(block_size or 0)
        if self.scale_mode == "block" and self.block_size <= 0:
            raise ValueError("residual_block_size must be > 0 when residual_scale_mode=block.")
        if self.scale_mode != "tensor":
            # The CLI exposes this as scaffolding, but this prototype keeps the
            # existing per-tensor scale behavior until channel/block storage is
            # implemented end to end.
            raise NotImplementedError("residual_scale_mode channel/block is not implemented yet; use tensor.")
        self.scales: Dict[str, torch.Tensor] = {}
        self.initial_scales: Dict[str, torch.Tensor] = {}
        self.residuals: Dict[str, torch.Tensor] = {}
        self.initial_snap_stats: Dict[str, Dict[str, float]] = {}

        with torch.no_grad():
            for name, param in named_parameters:
                if (not param.requires_grad) or (not torch.is_floating_point(param.data)):
                    continue
                scale = self._compute_scale(param)
                self.scales[name] = scale.detach().clone()
                self.initial_scales[name] = scale.detach().clone()
                self.residuals[name] = torch.zeros_like(param.data, dtype=self.residual_dtype, device=param.data.device)
                self.initial_snap_stats[name] = self.snap_param_to_grid(name, param)

    def _compute_scale(self, param: nn.Parameter) -> torch.Tensor:
        data = param.data.detach().float()
        if data.numel() == 0:
            return torch.tensor(1.0 / self.qmax, device=param.data.device, dtype=torch.float32)
        finite = torch.isfinite(data)
        if not bool(finite.all().item()):
            data = torch.where(finite, data, torch.zeros_like(data))
        max_abs = float(torch.max(torch.abs(data)).item()) if data.numel() > 0 else 0.0
        if (not math.isfinite(max_abs)) or max_abs <= 0.0:
            scale = self.scale_floor if self.scale_floor > 0.0 else (1.0 / self.qmax)
        else:
            scale = max_abs / self.qmax
            if self.scale_floor > 0.0:
                scale = max(scale, self.scale_floor)
        return torch.tensor(scale, device=param.data.device, dtype=torch.float32)

    def _broadcast_scale(self, scale: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        scale = scale.to(device=target.device, dtype=torch.float32)
        if scale.numel() == 1:
            return scale
        try:
            out_shape = torch.broadcast_shapes(tuple(scale.shape), tuple(target.shape))
        except RuntimeError as exc:
            raise ValueError(f"Scale shape {tuple(scale.shape)} cannot broadcast to parameter shape {tuple(target.shape)}") from exc
        if tuple(out_shape) != tuple(target.shape):
            raise ValueError(f"Scale shape {tuple(scale.shape)} broadcasts to {tuple(out_shape)}, not parameter shape {tuple(target.shape)}")
        return scale

    def _scale_for(self, name: str, param: nn.Parameter) -> torch.Tensor:
        if self.freeze_scale and name in self.scales:
            return self.scales[name].to(device=param.data.device)
        scale = self._compute_scale(param)
        self.scales[name] = scale.detach().clone()
        if name not in self.initial_scales:
            self.initial_scales[name] = scale.detach().clone()
        return scale

    def _grid_error_stats(self, tensor: torch.Tensor, scale: torch.Tensor) -> Tuple[float, float, float]:
        q = self.quantize_to_code(tensor, scale)
        snapped = self.dequantize_from_code(q, scale)
        err = torch.nan_to_num(tensor.detach().float() - snapped.float(), nan=0.0, posinf=0.0, neginf=0.0)
        err_sq = float(torch.sum(err * err).item())
        err_max = float(torch.max(torch.abs(err)).item()) if err.numel() > 0 else 0.0
        return err_sq, math.sqrt(max(err_sq, 0.0)), err_max

    def snap_param_to_grid(self, name: str, param: nn.Parameter) -> Dict[str, float]:
        scale = self._scale_for(name, param)
        before_sq, before_norm, before_max = self._grid_error_stats(param.data, scale)
        q = self.quantize_to_code(param.data, scale)
        snapped = self.dequantize_from_code(q, scale).to(dtype=param.data.dtype)
        param.data.copy_(snapped)
        after_sq, after_norm, after_max = self._grid_error_stats(param.data, scale)
        return {
            "grid_error_sq_before_snap": before_sq,
            "grid_error_norm_before_snap": before_norm,
            "grid_error_max_before_snap": before_max,
            "grid_error_sq_after_snap": after_sq,
            "grid_error_norm_after_snap": after_norm,
            "grid_error_max_after_snap": after_max,
        }

    def quantize_to_code(self, tensor: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        scale = self._broadcast_scale(scale, tensor)
        q = torch.round(torch.nan_to_num(tensor.detach().float() / scale, nan=0.0, posinf=self.qmax, neginf=self.qmin))
        return torch.clamp(q, self.qmin, self.qmax)

    def dequantize_from_code(self, code: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        scale = self._broadcast_scale(scale, code)
        return code.to(dtype=torch.float32) * scale

    def _top_abs_acc_select(self, k: torch.Tensor, score: torch.Tensor, target_active_frac: float) -> torch.Tensor:
        if target_active_frac <= 0.0 or k.numel() == 0:
            return k
        flat_k = k.reshape(-1)
        active_mask = flat_k != 0
        active_count = int(torch.count_nonzero(active_mask).item())
        if active_count <= 0:
            return k
        keep_count = min(active_count, int(math.ceil(float(target_active_frac) * float(flat_k.numel()))))
        if keep_count <= 0:
            return torch.zeros_like(k)
        if keep_count >= active_count:
            return k
        flat_score = score.reshape(-1)
        masked_score = torch.where(active_mask, flat_score, torch.full_like(flat_score, -float("inf")))
        _, keep_idx = torch.topk(masked_score, k=keep_count, largest=True, sorted=False)
        keep_mask = torch.zeros_like(active_mask)
        keep_mask[keep_idx] = True
        return torch.where(keep_mask.reshape_as(k), k, torch.zeros_like(k))

    def _norm_budget_select(
        self,
        k: torch.Tensor,
        score: torch.Tensor,
        scale: torch.Tensor,
        acc: torch.Tensor,
        delta: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, float, float]:
        reference = acc.float()
        if self.budget_reference == "delta" and delta is not None:
            reference = delta.float()
        reference = torch.nan_to_num(reference, nan=0.0, posinf=0.0, neginf=0.0)
        reference_norm = float(torch.linalg.vector_norm(reference.reshape(-1)).item()) if reference.numel() > 0 else 0.0
        cap = float(self.actual_norm_ratio_cap)
        if cap <= 0.0 or k.numel() == 0:
            return k, reference_norm, 0.0

        budget = cap * reference_norm
        if not math.isfinite(budget) or budget <= 0.0:
            return torch.zeros_like(k), reference_norm, budget

        actual_candidate = torch.nan_to_num(k.float() * scale.float(), nan=0.0, posinf=0.0, neginf=0.0)
        actual_candidate_norm = float(torch.linalg.vector_norm(actual_candidate.reshape(-1)).item()) if actual_candidate.numel() > 0 else 0.0
        if actual_candidate_norm <= budget:
            return k, reference_norm, budget

        flat_k = k.reshape(-1)
        active_idx = torch.nonzero(flat_k != 0, as_tuple=False).reshape(-1)
        if active_idx.numel() == 0:
            return k, reference_norm, budget

        flat_score = score.reshape(-1)
        order = torch.argsort(flat_score[active_idx], descending=True)
        ordered_idx = active_idx[order]
        scale_flat = scale.float().expand_as(k).reshape(-1) if scale.numel() == 1 else scale.reshape(-1).float()
        actual_flat = (flat_k.float() * scale_flat)[ordered_idx]
        sq = torch.nan_to_num(actual_flat * actual_flat, nan=0.0, posinf=float("inf"), neginf=0.0)
        keep_ordered = torch.cumsum(sq, dim=0) <= float(budget * budget)
        if not bool(torch.any(keep_ordered).item()):
            return torch.zeros_like(k), reference_norm, budget
        keep_idx = ordered_idx[keep_ordered]
        keep_mask = torch.zeros_like(flat_k, dtype=torch.bool)
        keep_mask[keep_idx] = True
        return torch.where(keep_mask.reshape_as(k), k, torch.zeros_like(k)), reference_norm, budget

    def _commit_codes(
        self,
        acc: torch.Tensor,
        scale: torch.Tensor,
        *,
        delta: Optional[torch.Tensor] = None,
        return_stats: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, float]]]:
        scale = self._broadcast_scale(scale, acc)
        a = acc.float() / scale
        if self.commit_mode == "round":
            k = torch.round(a)
        elif self.commit_mode == "floor":
            k = torch.sign(a) * torch.floor(torch.abs(a))
        else:
            abs_a = torch.abs(a)
            base = torch.floor(abs_a)
            prob = torch.clamp(abs_a - base, 0.0, 1.0)
            k_abs = base + (torch.rand_like(prob, dtype=torch.float32) < prob).to(dtype=base.dtype)
            k = torch.sign(a) * k_abs
        if self.max_code_step > 0:
            k = torch.clamp(k, -float(self.max_code_step), float(self.max_code_step))
        k = torch.nan_to_num(k, nan=0.0, posinf=float(self.max_code_step or self.qmax), neginf=-float(self.max_code_step or self.qmax))

        numel = int(k.numel())
        candidate_active_count = int(torch.count_nonzero(k != 0).item()) if numel > 0 else 0
        if self.commit_threshold > 0.0:
            k = torch.where(torch.abs(a) >= float(self.commit_threshold), k, torch.zeros_like(k))
        active_after_threshold_count = int(torch.count_nonzero(k != 0).item()) if numel > 0 else 0
        actual_before = torch.nan_to_num(k.float() * scale.float(), nan=0.0, posinf=0.0, neginf=0.0)
        actual_norm_before = float(torch.linalg.vector_norm(actual_before.reshape(-1)).item()) if actual_before.numel() > 0 else 0.0

        norm_budget_reference_norm = 0.0
        norm_budget_cap = 0.0
        score = torch.nan_to_num(torch.abs(a), nan=0.0, posinf=float(self.qmax), neginf=0.0)
        if self.commit_select == "top_abs_acc":
            k = self._top_abs_acc_select(k, score, self.target_active_frac)
        elif self.commit_select == "norm_budget":
            if self.target_active_frac > 0.0:
                k = self._top_abs_acc_select(k, score, self.target_active_frac)
            k, norm_budget_reference_norm, norm_budget_cap = self._norm_budget_select(k, score, scale, acc, delta)
        elif self.budget_reference == "delta" and delta is not None:
            ref = torch.nan_to_num(delta.float(), nan=0.0, posinf=0.0, neginf=0.0)
            norm_budget_reference_norm = float(torch.linalg.vector_norm(ref.reshape(-1)).item()) if ref.numel() > 0 else 0.0
        else:
            ref = torch.nan_to_num(acc.float(), nan=0.0, posinf=0.0, neginf=0.0)
            norm_budget_reference_norm = float(torch.linalg.vector_norm(ref.reshape(-1)).item()) if ref.numel() > 0 else 0.0

        selected_active_count = int(torch.count_nonzero(k != 0).item()) if numel > 0 else 0
        actual_after = torch.nan_to_num(k.float() * scale.float(), nan=0.0, posinf=0.0, neginf=0.0)
        actual_norm_after = float(torch.linalg.vector_norm(actual_after.reshape(-1)).item()) if actual_after.numel() > 0 else 0.0
        stats = {
            "candidate_active_count": float(candidate_active_count),
            "candidate_active_frac": (float(candidate_active_count) / float(numel)) if numel > 0 else 0.0,
            "candidate_active_frac_before_threshold": (float(candidate_active_count) / float(numel)) if numel > 0 else 0.0,
            "active_after_threshold_count": float(active_after_threshold_count),
            "active_frac_after_threshold": (float(active_after_threshold_count) / float(numel)) if numel > 0 else 0.0,
            "selected_active_count": float(selected_active_count),
            "selected_active_frac": (float(selected_active_count) / float(numel)) if numel > 0 else 0.0,
            "selection_dropped_count": float(max(candidate_active_count - selected_active_count, 0)),
            "selection_dropped_frac": (float(max(candidate_active_count - selected_active_count, 0)) / float(numel)) if numel > 0 else 0.0,
            "norm_budget_cap": float(norm_budget_cap),
            "norm_budget_reference_norm": float(norm_budget_reference_norm),
            "norm_budget_reference_sq": float(norm_budget_reference_norm * norm_budget_reference_norm),
            "actual_norm_before_selection": float(actual_norm_before),
            "actual_norm_after_selection": float(actual_norm_after),
            "actual_sq_before_selection": float(actual_norm_before * actual_norm_before),
            "actual_sq_after_selection": float(actual_norm_after * actual_norm_after),
        }
        if return_stats:
            return k, stats
        return k

    def _scale_stats(self, scale: torch.Tensor) -> Dict[str, float]:
        scale_f = torch.nan_to_num(scale.detach().float().reshape(-1), nan=0.0, posinf=0.0, neginf=0.0)
        if scale_f.numel() == 0:
            return {
                "scale_min": 0.0,
                "scale_p01": 0.0,
                "scale_median": 0.0,
                "scale_p99": 0.0,
                "scale_max": 0.0,
                "num_scale_zero": 0.0,
                "num_scale_near_zero": 0.0,
            }
        quantile_input = scale_f
        quantiles = torch.quantile(
            quantile_input,
            torch.tensor([0.01, 0.5, 0.99], device=quantile_input.device, dtype=torch.float32),
        )
        return {
            "scale_min": float(torch.min(scale_f).item()),
            "scale_p01": float(quantiles[0].item()),
            "scale_median": float(quantiles[1].item()),
            "scale_p99": float(quantiles[2].item()),
            "scale_max": float(torch.max(scale_f).item()),
            "num_scale_zero": float(torch.count_nonzero(scale_f == 0).item()),
            "num_scale_near_zero": float(torch.count_nonzero(torch.abs(scale_f) <= 1e-12).item()),
        }

    def _residual_over_scale_stats(
        self,
        residual: torch.Tensor,
        scale: torch.Tensor,
        q_new: Optional[torch.Tensor] = None,
    ) -> Dict[str, float]:
        if residual.numel() == 0:
            return {
                "residual_over_scale_p50": 0.0,
                "residual_over_scale_p90": 0.0,
                "residual_over_scale_p99": 0.0,
                "residual_over_scale_max": 0.0,
                "residual_over_scale_all_max": 0.0,
                "unsaturated_residual_bound_violation_count": 0.0,
                "unsaturated_count": 0.0,
                "unsaturated_residual_bound_violation_frac": 0.0,
                "residual_bound_check_applicable": 1.0 if self.max_code_step == 0 and self.commit_mode in {"round", "floor"} else 0.0,
            }
        scale = self._broadcast_scale(scale, residual)
        ratio = torch.abs(residual.detach().float() / scale)
        ratio = torch.nan_to_num(ratio, nan=0.0, posinf=float("inf"), neginf=0.0).reshape(-1)
        finite_ratio = torch.where(torch.isfinite(ratio), ratio, torch.full_like(ratio, float(self.qmax)))
        if q_new is None:
            unsat_mask = torch.ones_like(finite_ratio, dtype=torch.bool)
        else:
            q_flat = q_new.reshape(-1)
            unsat_mask = (q_flat > self.qmin) & (q_flat < self.qmax)
        quantile_input = finite_ratio[unsat_mask]
        unsat_count = int(quantile_input.numel())
        max_quantile_elems = 1_000_000
        if unsat_count > max_quantile_elems:
            stride = int(math.ceil(float(quantile_input.numel()) / float(max_quantile_elems)))
            quantile_input = quantile_input[::stride][:max_quantile_elems]
        if unsat_count > 0:
            quantiles = torch.quantile(quantile_input, torch.tensor([0.5, 0.9, 0.99], device=quantile_input.device, dtype=torch.float32))
            unsat_max = float(torch.max(finite_ratio[unsat_mask]).item())
        else:
            quantiles = torch.zeros(3, device=finite_ratio.device, dtype=torch.float32)
            unsat_max = 0.0
        bound = None
        if self.max_code_step == 0 and self.commit_mode == "round":
            bound = 0.5001
        elif self.max_code_step == 0 and self.commit_mode == "floor":
            bound = 1.0001
        violation_count = 0
        if unsat_count > 0 and bound is not None:
            violation_count = int(torch.count_nonzero((finite_ratio > bound) & unsat_mask).item())
        return {
            "residual_over_scale_p50": float(quantiles[0].item()),
            "residual_over_scale_p90": float(quantiles[1].item()),
            "residual_over_scale_p99": float(quantiles[2].item()),
            "residual_over_scale_max": unsat_max,
            "residual_over_scale_all_max": float(torch.max(finite_ratio).item()),
            "unsaturated_residual_bound_violation_count": float(violation_count),
            "num_violation": float(violation_count),
            "unsaturated_count": float(unsat_count),
            "num_unsaturated": float(unsat_count),
            "unsaturated_residual_bound_violation_frac": (float(violation_count) / float(unsat_count)) if unsat_count > 0 else 0.0,
            "residual_bound_check_applicable": 1.0 if bound is not None else 0.0,
        }

    def scale_drift_stats(self, name: str, param: nn.Parameter) -> Dict[str, float]:
        current = self._scale_for(name, param).detach().float().to(device=param.data.device)
        initial = self.initial_scales.get(name)
        if initial is None:
            initial = current.detach().clone()
            self.initial_scales[name] = initial
        initial = initial.detach().float().to(device=current.device)
        diff = torch.nan_to_num(current - initial, nan=0.0, posinf=0.0, neginf=0.0)
        diff_sq = float(torch.sum(diff * diff).item())
        return {
            "scale_delta_norm": math.sqrt(max(diff_sq, 0.0)),
            "scale_delta_max": float(torch.max(torch.abs(diff)).item()) if diff.numel() > 0 else 0.0,
        }

    def apply_update(
        self,
        name: str,
        param: nn.Parameter,
        direction: torch.Tensor,
        projected_grad: Union[torch.Tensor, float],
        learning_rate: float,
        *,
        weight_decay: float = 0.0,
        weight_decay_direction: Optional[torch.Tensor] = None,
    ) -> Dict[str, float]:
        with torch.no_grad():
            pg = float(projected_grad.detach().float().item()) if isinstance(projected_grad, torch.Tensor) else float(projected_grad)
            if not math.isfinite(pg):
                return {"skipped": 1.0, "skip_reason": "nonfinite_projected_grad", "numel": float(param.data.numel())}

            if name not in self.residuals or self.residuals[name].shape != param.data.shape or self.residuals[name].device != param.data.device:
                self.residuals[name] = torch.zeros_like(param.data, dtype=self.residual_dtype, device=param.data.device)
                self.scales[name] = self._compute_scale(param).detach().clone()
                self.snap_param_to_grid(name, param)

            scale = self._scale_for(name, param)
            residual = self.residuals[name]
            param_f = torch.nan_to_num(param.data.detach().float(), nan=0.0, posinf=0.0, neginf=0.0)
            if not bool(torch.isfinite(param.data).all().item()):
                param.data.copy_(param_f.to(dtype=param.data.dtype))
            snap_stats = self.snap_param_to_grid(name, param)

            update_direction = pg * direction.detach().float()
            if weight_decay != 0.0 and weight_decay_direction is not None:
                update_direction = update_direction + float(weight_decay) * weight_decay_direction.detach().float()
            delta = torch.nan_to_num(-float(learning_rate) * update_direction, nan=0.0, posinf=0.0, neginf=0.0)

            residual_before_f = residual.detach().float()
            residual_before_sq = float(torch.sum(residual_before_f * residual_before_f).item())
            if abs(float(self.residual_decay) - 1.0) <= 1e-12:
                acc = torch.nan_to_num(residual + delta.to(dtype=self.residual_dtype), nan=0.0, posinf=0.0, neginf=0.0)
            else:
                acc = torch.nan_to_num(
                    float(self.residual_decay) * residual.float() + delta.float(),
                    nan=0.0,
                    posinf=0.0,
                    neginf=0.0,
                ).to(dtype=self.residual_dtype)
            k, commit_stats = self._commit_codes(acc, scale, delta=delta, return_stats=True)
            q_current = self.quantize_to_code(param.data, scale)
            q_delta_requested = k
            q_new = torch.clamp(q_current + q_delta_requested, self.qmin, self.qmax)
            q_delta = q_new - q_current
            actual_delta = self.dequantize_from_code(q_delta, scale)
            new_param = self.dequantize_from_code(q_new, scale)
            param.data.copy_(torch.nan_to_num(new_param, nan=0.0, posinf=0.0, neginf=0.0).to(dtype=param.data.dtype))

            new_residual = torch.nan_to_num(acc.float() - actual_delta.float(), nan=0.0, posinf=0.0, neginf=0.0)
            residual.copy_(new_residual.to(dtype=self.residual_dtype))

            acc_f = acc.float()
            intended_f = delta.float()
            actual_f = actual_delta.float()
            residual_after_f = residual.detach().float()
            ef_error = torch.nan_to_num(residual_after_f - (acc_f - actual_f), nan=0.0, posinf=0.0, neginf=0.0)
            grid_error_sq, grid_error_norm, grid_error_max = self._grid_error_stats(param.data, scale)
            acc_sq = float(torch.sum(acc_f * acc_f).item())
            intended_sq = float(torch.sum(intended_f * intended_f).item())
            actual_sq = float(torch.sum(actual_f * actual_f).item())
            residual_sq = float(torch.sum(residual_after_f * residual_after_f).item())
            residual_after_sq = residual_sq
            dot = float(torch.sum(intended_f * actual_f).item())
            acc_actual_dot = float(torch.sum(acc_f * actual_f).item())
            eps = 1e-12
            acc_actual_cos = (
                acc_actual_dot / (math.sqrt(max(acc_sq, 0.0)) * math.sqrt(max(actual_sq, 0.0)) + eps)
                if acc_sq > 0.0 or actual_sq > 0.0 else None
            )
            actual_over_acc_norm_ratio = (
                math.sqrt(max(actual_sq, 0.0)) / (math.sqrt(max(acc_sq, 0.0)) + eps)
                if acc_sq > 0.0 else None
            )
            ef_error_sq = float(torch.sum(ef_error * ef_error).item())
            ef_error_norm = math.sqrt(max(ef_error_sq, 0.0))
            ef_error_max = float(torch.max(torch.abs(ef_error)).item()) if ef_error.numel() > 0 else 0.0
            changed = int(torch.count_nonzero(q_delta != 0).item())
            saturated = int(torch.count_nonzero((q_new == self.qmin) | (q_new == self.qmax)).item())
            numel = int(param.data.numel())
            scale_stats = self._scale_stats(scale)
            ros_stats = self._residual_over_scale_stats(residual_after_f, scale, q_new=q_new)

            stats = {
                "skipped": 0.0,
                "numel": float(numel),
                "acc_sq": acc_sq,
                "intended_sq": intended_sq,
                "actual_sq": actual_sq,
                "residual_sq": residual_sq,
                "residual_before_sq": residual_before_sq,
                "residual_after_sq": residual_after_sq,
                "dot": dot,
                "acc_actual_dot": acc_actual_dot,
                "acc_actual_cos": acc_actual_cos,
                "actual_over_acc_norm_ratio": actual_over_acc_norm_ratio,
                "ef_error_sq": ef_error_sq,
                "ef_error_norm": ef_error_norm,
                "ef_error_max": ef_error_max,
                "residual_decay": float(self.residual_decay),
                "ef_exact_conservation_applicable": 1.0 if abs(float(self.residual_decay) - 1.0) <= 1e-12 else 0.0,
                "residual_commit_threshold": float(self.commit_threshold),
                "residual_commit_select": self.commit_select,
                "residual_target_active_frac": float(self.target_active_frac),
                "residual_actual_norm_ratio_cap": float(self.actual_norm_ratio_cap),
                "residual_budget_reference": self.budget_reference,
                "active_count": float(changed),
                "saturation_count": float(saturated),
                "active_frac": (float(changed) / float(numel)) if numel > 0 else 0.0,
                "saturation_frac": (float(saturated) / float(numel)) if numel > 0 else 0.0,
                "grid_error_sq": grid_error_sq,
                "grid_error_norm": grid_error_norm,
                "grid_error_max": grid_error_max,
            }
            stats.update(snap_stats)
            stats.update(scale_stats)
            stats.update(ros_stats)
            stats.update(commit_stats)
            stats["max_abs_residual_over_scale"] = stats["residual_over_scale_max"]
            return stats
