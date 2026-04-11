import hashlib
import math
from typing import Dict, Optional

import torch
from torch import nn


SUPPORTED_QUZO_BITS = {16, 8, 4}


def validate_quzo_bits(bits: int) -> int:
    bits = int(bits)
    if bits not in {32, 16, 8, 4}:
        raise ValueError(f"Unsupported QuZO bits={bits}. Allowed values: 32, 16, 8, 4")
    return bits


def quzo_enabled(bits: int) -> bool:
    return int(bits) in SUPPORTED_QUZO_BITS


def _seed_from_parts(*parts: object) -> int:
    text = "::".join(str(part) for part in parts)
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return int(digest[:16], 16) % 2147483647


def _rand_like_with_seed(tensor: torch.Tensor, seed: int) -> torch.Tensor:
    devices = [tensor.device] if tensor.is_cuda else []
    with torch.random.fork_rng(devices=devices, enabled=True):
        torch.manual_seed(int(seed))
        if tensor.is_cuda:
            torch.cuda.manual_seed_all(int(seed))
        return torch.rand_like(tensor, dtype=torch.float32)


def _normal_like_with_seed(tensor: torch.Tensor, seed: int, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
    devices = [tensor.device] if tensor.is_cuda else []
    out_dtype = dtype if dtype is not None else tensor.dtype
    with torch.random.fork_rng(devices=devices, enabled=True):
        torch.manual_seed(int(seed))
        if tensor.is_cuda:
            torch.cuda.manual_seed_all(int(seed))
        return torch.normal(
            mean=0.0,
            std=1.0,
            size=tensor.size(),
            device=tensor.device,
            dtype=out_dtype,
        )


def quantize_tensor(
    tensor: torch.Tensor,
    bits: int,
    *,
    seed: Optional[int] = None,
    target_dtype: Optional[torch.dtype] = None,
    stochastic: bool = True,
) -> torch.Tensor:
    bits = validate_quzo_bits(bits)
    if target_dtype is None:
        target_dtype = tensor.dtype

    if bits == 32:
        return tensor.detach().to(dtype=target_dtype)

    x = tensor.detach().float()
    if bits == 16:
        return x.to(dtype=torch.float16).to(dtype=target_dtype)

    qmax = (1 << (bits - 1)) - 1
    max_abs = float(torch.max(torch.abs(x)).item()) if x.numel() > 0 else 0.0
    if (not math.isfinite(max_abs)) or max_abs <= 0.0:
        return torch.zeros_like(x, dtype=target_dtype)

    scale = max_abs / float(qmax)
    y = torch.clamp(x / scale, -float(qmax), float(qmax))

    if stochastic:
        lower = torch.floor(y)
        prob = torch.clamp(y - lower, 0.0, 1.0)
        rnd = _rand_like_with_seed(y, int(seed)) if seed is not None else torch.rand_like(y, dtype=torch.float32)
        q = lower + (rnd < prob).to(dtype=y.dtype)
    else:
        q = torch.round(y)

    q = torch.clamp(q, -float(qmax), float(qmax))
    return (q * scale).to(dtype=target_dtype)


def make_quzo_direction_pair(
    tensor: torch.Tensor,
    *,
    bits: int,
    key: str,
    step_seed: int,
    target_dtype: Optional[torch.dtype] = None,
) -> Dict[str, torch.Tensor]:
    bits = validate_quzo_bits(bits)
    if target_dtype is None:
        target_dtype = tensor.dtype

    direction_dtype = torch.float32 if bits in {8, 4} else target_dtype
    gaussian_seed = _seed_from_parts(step_seed, key, "gaussian")
    perturb_seed = _seed_from_parts(step_seed, key, "perturb")
    update_seed = _seed_from_parts(step_seed, key, "update")
    state_seed = _seed_from_parts(step_seed, key, "state")

    u = _normal_like_with_seed(tensor, gaussian_seed, dtype=direction_dtype)
    u1 = quantize_tensor(u, bits, seed=perturb_seed, target_dtype=target_dtype)
    u2 = quantize_tensor(u, bits, seed=update_seed, target_dtype=target_dtype)
    return {
        "u1": u1,
        "u2": u2,
        "seed": torch.tensor(int(step_seed), device=tensor.device, dtype=torch.int64),
        "state_seed": torch.tensor(int(state_seed), device=tensor.device, dtype=torch.int64),
    }


def quantize_model_in_place(
    model: nn.Module,
    bits: int,
    *,
    include_frozen: bool = True,
    seed: int = 0,
) -> None:
    bits = validate_quzo_bits(bits)
    if bits == 32:
        return

    with torch.no_grad():
        for name, param in model.named_parameters():
            if (not include_frozen) and (not param.requires_grad):
                continue
            q_seed = _seed_from_parts(seed, name, "model_init")
            param.data.copy_(quantize_tensor(param.data, bits, seed=q_seed, target_dtype=param.data.dtype))
