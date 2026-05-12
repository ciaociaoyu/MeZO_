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

    def _commit_codes(self, acc: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
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
        return torch.nan_to_num(k, nan=0.0, posinf=float(self.max_code_step or self.qmax), neginf=-float(self.max_code_step or self.qmax))

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

            acc = torch.nan_to_num(residual + delta.to(dtype=self.residual_dtype), nan=0.0, posinf=0.0, neginf=0.0)
            k = self._commit_codes(acc, scale)
            q_current = self.quantize_to_code(param.data, scale)
            q_delta_requested = k
            q_new = torch.clamp(q_current + q_delta_requested, self.qmin, self.qmax)
            q_delta = q_new - q_current
            actual_delta = self.dequantize_from_code(q_delta, scale)
            new_param = self.dequantize_from_code(q_new, scale)
            param.data.copy_(torch.nan_to_num(new_param, nan=0.0, posinf=0.0, neginf=0.0).to(dtype=param.data.dtype))

            new_residual = torch.nan_to_num(acc.float() - actual_delta.float(), nan=0.0, posinf=0.0, neginf=0.0)
            residual.copy_(new_residual.to(dtype=self.residual_dtype))

            intended_f = delta.float()
            actual_f = actual_delta.float()
            residual_f = residual.detach().float()
            grid_error_sq, grid_error_norm, grid_error_max = self._grid_error_stats(param.data, scale)
            intended_sq = float(torch.sum(intended_f * intended_f).item())
            actual_sq = float(torch.sum(actual_f * actual_f).item())
            residual_sq = float(torch.sum(residual_f * residual_f).item())
            dot = float(torch.sum(intended_f * actual_f).item())
            changed = int(torch.count_nonzero(q_delta != 0).item())
            saturated = int(torch.count_nonzero((q_new == self.qmin) | (q_new == self.qmax)).item())
            numel = int(param.data.numel())
            scale_stats = self._scale_stats(scale)
            ros_stats = self._residual_over_scale_stats(residual_f, scale, q_new=q_new)

            stats = {
                "skipped": 0.0,
                "numel": float(numel),
                "intended_sq": intended_sq,
                "actual_sq": actual_sq,
                "residual_sq": residual_sq,
                "dot": dot,
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
            stats["max_abs_residual_over_scale"] = stats["residual_over_scale_max"]
            return stats
