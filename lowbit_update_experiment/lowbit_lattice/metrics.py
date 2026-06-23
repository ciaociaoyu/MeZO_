from __future__ import annotations

import math
from typing import Dict, Optional

import torch


def safe_cosine(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-12) -> float:
    af = a.detach().float().reshape(-1)
    bf = b.detach().float().reshape(-1)
    denom = torch.linalg.vector_norm(af) * torch.linalg.vector_norm(bf) + eps
    return float(torch.dot(af, bf).item() / denom.item())


def update_geometry_metrics(
    *,
    grad: torch.Tensor,
    intended_update: torch.Tensor,
    actual_update: torch.Tensor,
    q_old: Optional[torch.Tensor],
    q_new: Optional[torch.Tensor],
    qmin: Optional[int],
    qmax: Optional[int],
    train_loss_before: float,
    train_loss_after: float,
    heldout_loss_before: float,
    heldout_loss_after: float,
) -> Dict[str, object]:
    intended = intended_update.detach().float()
    actual = actual_update.detach().float()
    grad_f = grad.detach().float()
    intended_norm = torch.linalg.vector_norm(intended).item()
    actual_norm = torch.linalg.vector_norm(actual).item()
    first_order = torch.sum(grad_f * actual).item()
    delta_train = float(train_loss_after - train_loss_before)
    delta_heldout = float(heldout_loss_after - heldout_loss_before)
    out: Dict[str, object] = {
        "train_loss_before": float(train_loss_before),
        "train_loss_after": float(train_loss_after),
        "heldout_loss_before": float(heldout_loss_before),
        "heldout_loss_after": float(heldout_loss_after),
        "delta_train_loss": delta_train,
        "delta_heldout_loss": delta_heldout,
        "intended_update_norm": float(intended_norm),
        "actual_update_norm": float(actual_norm),
        "norm_ratio": float(actual_norm / (intended_norm + 1e-12)),
        "cosine_intended_actual": safe_cosine(intended, actual),
        "first_order_predicted_change": float(first_order),
        "loss_change_matches_first_order": bool(math.copysign(1.0, delta_train) == math.copysign(1.0, first_order))
        if delta_train != 0 and first_order != 0
        else False,
    }
    if q_old is not None and q_new is not None:
        changed = q_old != q_new
        out["active_fraction"] = float(changed.float().mean().item())
        out["num_codes_changed"] = int(changed.sum().item())
        if qmin is not None and qmax is not None:
            sat_before = (q_old == qmin) | (q_old == qmax)
            sat_after = (q_new == qmin) | (q_new == qmax)
            out["saturation_fraction_before"] = float(sat_before.float().mean().item())
            out["saturation_fraction_after"] = float(sat_after.float().mean().item())
    else:
        out["active_fraction"] = None
        out["num_codes_changed"] = None
        out["saturation_fraction_before"] = None
        out["saturation_fraction_after"] = None

    out["effective_lowbit_update"] = bool(
        (out["active_fraction"] or 0.0) > 0.0
        and actual_norm > 0.0
        and out["cosine_intended_actual"] > 0.05
        and first_order < 0.0
        and delta_train < 0.0
    )
    out["heldout_effective_lowbit_update"] = bool(out["effective_lowbit_update"] and delta_heldout < 0.0)
    return out
