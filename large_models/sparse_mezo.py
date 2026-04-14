import math
from typing import Dict, Iterable, Tuple

import torch


# Sparse MeZO keeps approximately sparse_ratio of coordinates active within each
# trainable tensor. ratio=1.0 disables masking and recovers dense MeZO.
SPARSE_MASK_STRATEGY_ALIASES = {
    "percentile_per_layer": "percentile_per_layer",
    "per_layer_percentile": "percentile_per_layer",
}

SPARSE_SCOPE_ALIASES = {
    "trainable_only": "trainable_only",
    "trainable": "trainable_only",
}


def validate_sparse_ratio(ratio: float) -> float:
    value = float(ratio)
    if (not math.isfinite(value)) or value <= 0.0 or value > 1.0:
        raise ValueError(f"Invalid sparse_ratio={ratio}. Expected a finite float in (0, 1].")
    return value


def normalize_sparse_mask_strategy(strategy: str) -> str:
    key = str(strategy or "percentile_per_layer").strip().lower()
    if key in SPARSE_MASK_STRATEGY_ALIASES:
        return SPARSE_MASK_STRATEGY_ALIASES[key]
    supported = ", ".join(sorted(SPARSE_MASK_STRATEGY_ALIASES))
    raise ValueError(f"Unsupported sparse_mask_strategy={strategy!r}. Supported: {supported}")


def normalize_sparse_scope(scope: str) -> str:
    key = str(scope or "trainable_only").strip().lower()
    if key in SPARSE_SCOPE_ALIASES:
        return SPARSE_SCOPE_ALIASES[key]
    supported = ", ".join(sorted(SPARSE_SCOPE_ALIASES))
    raise ValueError(f"Unsupported sparse_scope={scope!r}. Supported: {supported}")


def sparse_mezo_enabled(ratio: float) -> bool:
    return validate_sparse_ratio(ratio) < 1.0


def build_sparse_masks(
    named_parameters: Iterable[Tuple[str, torch.nn.Parameter]],
    *,
    ratio: float,
    mask_strategy: str,
    scope: str,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, float]]:
    ratio = validate_sparse_ratio(ratio)
    mask_strategy = normalize_sparse_mask_strategy(mask_strategy)
    scope = normalize_sparse_scope(scope)
    if scope != "trainable_only":
        raise ValueError(f"Only sparse_scope=trainable_only is supported right now, got {scope!r}")
    if mask_strategy != "percentile_per_layer":
        raise ValueError(f"Unsupported sparse_mask_strategy={mask_strategy!r}")

    masks: Dict[str, torch.Tensor] = {}
    total_params = 0
    active_params = 0

    for name, param in named_parameters:
        data = param.data.detach()
        numel = int(data.numel())
        total_params += numel
        if numel == 0:
            continue

        if ratio >= 1.0:
            active_params += numel
            continue

        k = max(int(math.floor(ratio * numel)), 1)
        if k >= numel:
            mask = torch.ones_like(data, dtype=torch.bool)
        else:
            # percentile_per_layer follows Sparse MeZO: keep the lowest-|param|
            # fraction active in each trainable tensor.
            flat_abs = data.abs().reshape(-1)
            threshold = torch.kthvalue(flat_abs, k).values
            mask = (flat_abs <= threshold).reshape_as(data)
        masks[name] = mask
        active_params += int(mask.sum().item())

    active_fraction = 1.0 if total_params == 0 else float(active_params) / float(total_params)
    stats = {
        "configured_ratio": float(ratio),
        "active_params": int(active_params),
        "total_trainable_params": int(total_params),
        "active_fraction": float(active_fraction),
        "mask_strategy": mask_strategy,
        "scope": scope,
    }
    return masks, stats


def apply_sparse_mask(direction: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if mask is None:
        return direction
    return direction * mask.to(device=direction.device, dtype=direction.dtype)
