from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch

from .quant import GroupwiseQuantizedWeight


@dataclass
class UpdateResult:
    rule: str
    w_old: torch.Tensor
    w_new: torch.Tensor
    intended_update: torch.Tensor
    q_old: Optional[torch.Tensor]
    q_new: Optional[torch.Tensor]
    lattice: Optional[GroupwiseQuantizedWeight]
    extra: Dict[str, float]


def compute_lr_for_relative_update(
    weight: torch.Tensor,
    grad: torch.Tensor,
    relative_update_norm: float,
    eps: float = 1e-12,
) -> float:
    w_norm = torch.linalg.vector_norm(weight.detach().float())
    g_norm = torch.linalg.vector_norm(grad.detach().float())
    return float(relative_update_norm * w_norm.item() / (g_norm.item() + eps))


def _rand_like(shape, *, generator: torch.Generator, device: torch.device) -> torch.Tensor:
    return torch.rand(shape, device=device, generator=generator)


def _topk_mask(score: torch.Tensor, k_frac: float) -> torch.Tensor:
    if k_frac <= 0:
        return torch.zeros_like(score, dtype=torch.bool)
    flat = score.reshape(-1)
    k = max(1, int(torch.ceil(torch.tensor(float(flat.numel()) * k_frac)).item()))
    k = min(k, flat.numel())
    idx = torch.topk(flat, k=k, largest=True, sorted=False).indices
    mask = torch.zeros(flat.numel(), device=score.device, dtype=torch.bool)
    mask[idx] = True
    return mask.reshape_as(score)


def apply_update_rule(
    lattice: GroupwiseQuantizedWeight,
    grad: torch.Tensor,
    *,
    lr: float,
    rule: str,
    k_frac: float = 0.0,
    p_tail_max: float = 0.01,
    p_max: float = 0.01,
    generator: Optional[torch.Generator] = None,
) -> UpdateResult:
    grad_f = grad.detach().float()
    old = lattice.to(grad_f.device)
    w_old = old.dequantize(torch.float32)
    intended = -float(lr) * grad_f
    q_old = old.codes.to(grad_f.device)
    scale = old.expanded_scales().to(grad_f.device)
    w_intended = w_old + intended

    if rule == "fp_sgd_upper_bound":
        return UpdateResult(rule, w_old, w_intended, intended, None, None, None, {})

    if rule == "nearest_requant_fixed_grid":
        q_new = torch.round(w_intended / scale).clamp(old.qmin, old.qmax).to(torch.int16)
        new_lattice = old.with_codes(q_new)
        return UpdateResult(rule, w_old, new_lattice.dequantize(), intended, q_old, q_new, new_lattice, {})

    if generator is None:
        generator = torch.Generator(device=grad_f.device)
        generator.manual_seed(0)

    if rule == "stochastic_round_fixed_grid":
        x = w_intended / scale
        q_floor = torch.floor(x)
        prob = (x - q_floor).clamp(0.0, 1.0)
        q_new = torch.where(_rand_like(prob.shape, generator=generator, device=grad_f.device) < prob, q_floor + 1.0, q_floor)
        q_new = q_new.clamp(old.qmin, old.qmax).to(torch.int16)
        new_lattice = old.with_codes(q_new)
        return UpdateResult(rule, w_old, new_lattice.dequantize(), intended, q_old, q_new, new_lattice, {})

    delta_code_float = intended / scale
    sign = torch.sign(delta_code_float).to(torch.int16)
    sign = torch.where(sign == 0, torch.zeros_like(sign), sign)
    score = delta_code_float.abs()

    if rule == "topk_code_flip":
        mask = _topk_mask(score, k_frac)
        q_new = q_old.clone()
        q_new[mask] = (q_new[mask] + sign[mask]).clamp(old.qmin, old.qmax)
        new_lattice = old.with_codes(q_new)
        return UpdateResult(rule, w_old, new_lattice.dequantize(), intended, q_old, q_new, new_lattice, {"k_frac": float(k_frac)})

    if rule == "topk_code_flip_plus_stochastic_tail":
        top_mask = _topk_mask(score, k_frac)
        tail_prob = torch.minimum(score, torch.tensor(float(p_tail_max), device=score.device))
        tail_mask = (~top_mask) & (_rand_like(score.shape, generator=generator, device=score.device) < tail_prob)
        mask = top_mask | tail_mask
        q_new = q_old.clone()
        q_new[mask] = (q_new[mask] + sign[mask]).clamp(old.qmin, old.qmax)
        new_lattice = old.with_codes(q_new)
        return UpdateResult(
            rule,
            w_old,
            new_lattice.dequantize(),
            intended,
            q_old,
            q_new,
            new_lattice,
            {"k_frac": float(k_frac), "p_tail_max": float(p_tail_max)},
        )

    if rule == "dense_stochastic_code_flip":
        prob = torch.minimum(score, torch.tensor(float(p_max), device=score.device))
        mask = _rand_like(score.shape, generator=generator, device=score.device) < prob
        q_new = q_old.clone()
        q_new[mask] = (q_new[mask] + sign[mask]).clamp(old.qmin, old.qmax)
        new_lattice = old.with_codes(q_new)
        return UpdateResult(rule, w_old, new_lattice.dequantize(), intended, q_old, q_new, new_lattice, {"p_max": float(p_max)})

    raise ValueError(f"unknown update rule {rule}")
