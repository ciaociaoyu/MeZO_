import math
from typing import Dict, Iterable, Optional, Tuple

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


def build_sparse_thresholds(
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

    thresholds: Dict[str, torch.Tensor] = {}
    total_params = 0
    thresholded_tensors = 0

    for name, param in named_parameters:
        data = param.data.detach()
        numel = int(data.numel())
        total_params += numel
        if numel == 0 or ratio >= 1.0:
            continue

        k = max(int(math.floor(ratio * numel)), 1)
        flat_abs = data.abs().reshape(-1)
        if k >= numel:
            threshold = torch.max(flat_abs)
        else:
            # percentile_per_layer follows Sparse MeZO: keep the lowest-|param|
            # fraction active in each trainable tensor.
            threshold = torch.kthvalue(flat_abs, k).values
        thresholds[name] = threshold.detach()
        thresholded_tensors += 1

    stats = {
        "configured_ratio": float(ratio),
        "total_trainable_params": int(total_params),
        "thresholded_tensors": int(thresholded_tensors),
        "mask_strategy": mask_strategy,
        "scope": scope,
    }
    return thresholds, stats


def build_sparse_masks_from_thresholds(
    named_parameters: Iterable[Tuple[str, torch.nn.Parameter]],
    *,
    thresholds: Dict[str, torch.Tensor],
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

        threshold = thresholds.get(name)
        if threshold is None:
            raise KeyError(f"Missing sparse threshold for parameter {name!r}")
        mask = data.abs() <= threshold.to(device=data.device, dtype=data.dtype)
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


def build_sparse_masks(
    named_parameters: Iterable[Tuple[str, torch.nn.Parameter]],
    *,
    ratio: float,
    mask_strategy: str,
    scope: str,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, float]]:
    thresholds, _ = build_sparse_thresholds(
        named_parameters,
        ratio=ratio,
        mask_strategy=mask_strategy,
        scope=scope,
    )
    return build_sparse_masks_from_thresholds(
        named_parameters,
        thresholds=thresholds,
        ratio=ratio,
        mask_strategy=mask_strategy,
        scope=scope,
    )


def apply_sparse_mask(direction: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if mask is None:
        return direction
    return direction * mask.to(device=direction.device, dtype=direction.dtype)


def sample_masked_normal_like(
    tensor: torch.Tensor,
    *,
    mask: Optional[torch.Tensor] = None,
    seed: Optional[int] = None,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    out_dtype = dtype if dtype is not None else tensor.dtype
    generator = None
    if seed is not None:
        generator = torch.Generator(device=tensor.device.type)
        generator.manual_seed(int(seed))

    if mask is None:
        return torch.empty_like(tensor, dtype=out_dtype).normal_(mean=0.0, std=1.0, generator=generator)

    active_mask = mask.to(device=tensor.device, dtype=torch.bool)
    out = torch.zeros_like(tensor, dtype=out_dtype)
    active_count = int(active_mask.sum().item())
    if active_count <= 0:
        return out

    samples = torch.empty((active_count,), device=tensor.device, dtype=out_dtype).normal_(
        mean=0.0,
        std=1.0,
        generator=generator,
    )
    out[active_mask] = samples
    return out
