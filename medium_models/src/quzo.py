import hashlib
import math
from typing import Any, Dict, Optional, Tuple

import torch
from torch import nn

from src.sparse_mezo import sample_masked_normal_like


SUPPORTED_QUZO_BITS = {16, 8, 4}
QUZO_BIT_ALIASES = {
    32: 32,
    16: 16,
    8: 8,
    4: 4,
    "32": 32,
    "fp32": 32,
    "none": 32,
    "off": 32,
    "16": 16,
    "fp16": 16,
    "half": 16,
    "8": 8,
    "int8": 8,
    "quzo8": 8,
    "4": 4,
    "int4": 4,
    "quzo4": 4,
}

QUANTIZATION_ALGORITHM_ALIASES = {
    "": "per_tensor_symmetric",
    "none": "per_tensor_symmetric",
    "uniform": "per_tensor_symmetric",
    "tensor": "per_tensor_symmetric",
    "per_tensor": "per_tensor_symmetric",
    "per_tensor_symmetric": "per_tensor_symmetric",
    "symmetric": "per_tensor_symmetric",
    "block": "groupwise_symmetric",
    "blockwise": "groupwise_symmetric",
    "group": "groupwise_symmetric",
    "groupwise": "groupwise_symmetric",
    "groupwise_symmetric": "groupwise_symmetric",
    "groupwise_int8_block256": "groupwise_symmetric",
    "groupwise_int4_block256": "groupwise_symmetric",
}


def validate_quzo_bits(bits: int) -> int:
    if isinstance(bits, str):
        key = bits.strip().lower()
        if key in QUZO_BIT_ALIASES:
            bits = QUZO_BIT_ALIASES[key]
        else:
            bits = int(bits)
    else:
        bits = int(bits)
    if bits not in {32, 16, 8, 4}:
        raise ValueError(f"Unsupported QuZO bits={bits}. Allowed values: 32, 16, 8, 4")
    return bits


def quzo_enabled(bits: int) -> bool:
    return int(bits) in SUPPORTED_QUZO_BITS


def normalize_quantization_algorithm(value: Optional[str]) -> str:
    key = str(value or "per_tensor_symmetric").strip().lower().replace("-", "_")
    if key in QUANTIZATION_ALGORITHM_ALIASES:
        return QUANTIZATION_ALGORITHM_ALIASES[key]
    raise ValueError(
        f"Unsupported quantization_algorithm={value!r}. "
        "Supported local algorithms: per_tensor_symmetric, groupwise_int8_block256."
    )


def quantization_algorithm_label(algorithm: Optional[str], *, bits: int, group_size: int = 0) -> str:
    algo = normalize_quantization_algorithm(algorithm)
    if algo == "groupwise_symmetric":
        size = int(group_size or 0)
        if size > 0 and int(bits) in {8, 4}:
            return f"groupwise_int{int(bits)}_block{size}"
        return "groupwise_symmetric"
    return "per_tensor_symmetric"


def exact_gptq_available() -> bool:
    return False


def _seed_from_parts(*parts: object) -> int:
    text = "::".join(str(part) for part in parts)
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return int(digest[:16], 16) % 2147483647


def _rand_like_with_seed(tensor: torch.Tensor, seed: int) -> torch.Tensor:
    generator = torch.Generator(device=tensor.device.type)
    generator.manual_seed(int(seed))
    return torch.rand(tensor.size(), device=tensor.device, dtype=torch.float32, generator=generator)


def _normal_like_with_seed(
    tensor: torch.Tensor,
    seed: int,
    dtype: Optional[torch.dtype] = None,
    *,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    return sample_masked_normal_like(tensor, mask=mask, seed=int(seed), dtype=dtype)


def _quant_scale_metadata(
    *,
    scale: torch.Tensor,
    q: torch.Tensor,
    bits: int,
    algorithm: str,
    group_size: int,
) -> Dict[str, Any]:
    scale_f = torch.nan_to_num(scale.detach().float().reshape(-1), nan=0.0, posinf=0.0, neginf=0.0)
    qmax = float((1 << (bits - 1)) - 1)
    qmin = -qmax
    if scale_f.numel() == 0:
        scale_min = scale_median = scale_max = 0.0
    else:
        scale_min = float(torch.min(scale_f).item())
        scale_median = float(torch.median(scale_f).item())
        scale_max = float(torch.max(scale_f).item())
    q_f = q.detach().float()
    return {
        "quantization_algorithm": quantization_algorithm_label(algorithm, bits=bits, group_size=group_size),
        "quantization_algorithm_impl": normalize_quantization_algorithm(algorithm),
        "bits": int(bits),
        "group_size": int(group_size or 0),
        "block_size": int(group_size or 0),
        "scale_shape": list(scale.shape),
        "scale_min": scale_min,
        "scale_median": scale_median,
        "scale_max": scale_max,
        "zero_point_mode": "none_symmetric",
        "num_groups": int(scale_f.numel()),
        "saturation_count": int(torch.count_nonzero((q_f <= qmin) | (q_f >= qmax)).item()) if q_f.numel() > 0 else 0,
        "saturation_frac": (
            float(torch.count_nonzero((q_f <= qmin) | (q_f >= qmax)).item()) / float(q_f.numel())
            if q_f.numel() > 0 else 0.0
        ),
    }


def _round_quant_values(
    y: torch.Tensor,
    *,
    qmax: float,
    stochastic: bool,
    seed: Optional[int],
) -> torch.Tensor:
    if stochastic:
        lower = torch.floor(y)
        prob = torch.clamp(y - lower, 0.0, 1.0)
        rnd = _rand_like_with_seed(y, int(seed)) if seed is not None else torch.rand_like(y, dtype=torch.float32)
        q = lower + (rnd < prob).to(dtype=y.dtype)
    else:
        q = torch.round(y)
    return torch.clamp(q, -float(qmax), float(qmax))


def _quantize_per_tensor_symmetric(
    tensor: torch.Tensor,
    bits: int,
    *,
    seed: Optional[int],
    target_dtype: torch.dtype,
    stochastic: bool,
    return_metadata: bool,
) -> Any:
    x = tensor.detach()
    qmax = (1 << (bits - 1)) - 1
    max_abs = float(torch.max(torch.abs(x)).item()) if x.numel() > 0 else 0.0
    if (not math.isfinite(max_abs)) or max_abs <= 0.0:
        out = torch.zeros_like(tensor, dtype=target_dtype)
        if return_metadata:
            q = torch.zeros_like(tensor, dtype=torch.float32)
            scale = torch.tensor(1.0 / float(qmax), device=tensor.device, dtype=torch.float32)
            return out, _quant_scale_metadata(
                scale=scale,
                q=q,
                bits=bits,
                algorithm="per_tensor_symmetric",
                group_size=0,
            )
        return out

    scale = max_abs / float(qmax)
    y = torch.clamp(x / scale, -float(qmax), float(qmax))
    q = _round_quant_values(y, qmax=float(qmax), stochastic=stochastic, seed=seed)
    out = (q * scale).to(dtype=target_dtype)
    if return_metadata:
        return out, _quant_scale_metadata(
            scale=torch.tensor(scale, device=tensor.device, dtype=torch.float32),
            q=q,
            bits=bits,
            algorithm="per_tensor_symmetric",
            group_size=0,
        )
    return out


def _quantize_groupwise_symmetric(
    tensor: torch.Tensor,
    bits: int,
    *,
    group_size: int,
    seed: Optional[int],
    target_dtype: torch.dtype,
    stochastic: bool,
    return_metadata: bool,
) -> Any:
    if group_size <= 0:
        raise ValueError("groupwise quantization requires group_size/block_size > 0")
    x = tensor.detach().float()
    qmax = (1 << (bits - 1)) - 1
    if x.numel() == 0:
        out = torch.zeros_like(tensor, dtype=target_dtype)
        if return_metadata:
            scale = torch.empty(0, device=tensor.device, dtype=torch.float32)
            q = torch.empty_like(x)
            return out, _quant_scale_metadata(
                scale=scale,
                q=q,
                bits=bits,
                algorithm="groupwise_symmetric",
                group_size=group_size,
            )
        return out

    flat = x.reshape(-1)
    numel = int(flat.numel())
    num_groups = int(math.ceil(float(numel) / float(group_size)))
    padded_numel = num_groups * int(group_size)
    if padded_numel > numel:
        pad = torch.zeros(padded_numel - numel, device=flat.device, dtype=flat.dtype)
        flat_padded = torch.cat([flat, pad], dim=0)
    else:
        flat_padded = flat
    grouped = flat_padded.view(num_groups, int(group_size))
    finite = torch.isfinite(grouped)
    grouped = torch.where(finite, grouped, torch.zeros_like(grouped))
    max_abs = torch.amax(torch.abs(grouped), dim=1, keepdim=True)
    default_scale = torch.full_like(max_abs, 1.0 / float(qmax))
    scale = torch.where(max_abs > 0.0, max_abs / float(qmax), default_scale)
    y = torch.clamp(grouped / scale, -float(qmax), float(qmax))
    q = _round_quant_values(y, qmax=float(qmax), stochastic=stochastic, seed=seed)
    out_flat = (q * scale).reshape(-1)[:numel]
    out = out_flat.view_as(tensor).to(dtype=target_dtype)
    if return_metadata:
        return out, _quant_scale_metadata(
            scale=scale.reshape(-1),
            q=q.reshape(-1)[:numel],
            bits=bits,
            algorithm="groupwise_symmetric",
            group_size=group_size,
        )
    return out


def quantize_tensor(
    tensor: torch.Tensor,
    bits: int,
    *,
    seed: Optional[int] = None,
    target_dtype: Optional[torch.dtype] = None,
    stochastic: bool = True,
    algorithm: Optional[str] = "per_tensor_symmetric",
    group_size: int = 0,
    block_size: int = 0,
    return_metadata: bool = False,
) -> Any:
    bits = validate_quzo_bits(bits)
    if target_dtype is None:
        target_dtype = tensor.dtype
    algorithm = normalize_quantization_algorithm(algorithm)
    effective_group_size = int(group_size or block_size or 0)

    if bits == 32:
        out = tensor.detach().to(dtype=target_dtype)
        if return_metadata:
            return out, {
                "quantization_algorithm": "none",
                "quantization_algorithm_impl": "none",
                "bits": 32,
                "group_size": 0,
                "block_size": 0,
                "scale_shape": [],
                "scale_min": None,
                "scale_median": None,
                "scale_max": None,
                "zero_point_mode": "none",
                "num_groups": 0,
                "saturation_count": 0,
                "saturation_frac": 0.0,
            }
        return out
    if bits == 16:
        out = tensor.detach().to(dtype=torch.float16)
        if return_metadata:
            return out, {
                "quantization_algorithm": "fp16",
                "quantization_algorithm_impl": "fp16",
                "bits": 16,
                "group_size": 0,
                "block_size": 0,
                "scale_shape": [],
                "scale_min": None,
                "scale_median": None,
                "scale_max": None,
                "zero_point_mode": "none",
                "num_groups": 0,
                "saturation_count": 0,
                "saturation_frac": 0.0,
            }
        return out

    if algorithm == "groupwise_symmetric":
        return _quantize_groupwise_symmetric(
            tensor,
            bits,
            group_size=effective_group_size,
            seed=seed,
            target_dtype=target_dtype,
            stochastic=stochastic,
            return_metadata=return_metadata,
        )
    return _quantize_per_tensor_symmetric(
        tensor,
        bits,
        seed=seed,
        target_dtype=target_dtype,
        stochastic=stochastic,
        return_metadata=return_metadata,
    )


def make_quzo_direction_pair(
    tensor: torch.Tensor,
    *,
    bits: int,
    key: str,
    step_seed: int,
    mask: Optional[torch.Tensor] = None,
    target_dtype: Optional[torch.dtype] = None,
    algorithm: Optional[str] = "per_tensor_symmetric",
    group_size: int = 0,
    block_size: int = 0,
) -> Dict[str, torch.Tensor]:
    bits = validate_quzo_bits(bits)
    if target_dtype is None:
        target_dtype = tensor.dtype

    if bits == 16:
        z_dtype = torch.float16
        z = _normal_like_with_seed(tensor, int(step_seed), dtype=z_dtype, mask=mask)
        return {
            "z": z,
            "seed": torch.tensor(int(step_seed), device=tensor.device, dtype=torch.int64),
        }

    direction_dtype = torch.float32 if bits in {8, 4} else target_dtype
    gaussian_seed = _seed_from_parts(step_seed, key, "gaussian")
    perturb_seed = _seed_from_parts(step_seed, key, "perturb")
    update_seed = _seed_from_parts(step_seed, key, "update")
    state_seed = _seed_from_parts(step_seed, key, "state")

    u = _normal_like_with_seed(tensor, gaussian_seed, dtype=direction_dtype, mask=mask)
    u1 = quantize_tensor(
        u,
        bits,
        seed=perturb_seed,
        target_dtype=target_dtype,
        algorithm=algorithm,
        group_size=group_size,
        block_size=block_size,
    )
    u2 = quantize_tensor(
        u,
        bits,
        seed=update_seed,
        target_dtype=target_dtype,
        algorithm=algorithm,
        group_size=group_size,
        block_size=block_size,
    )
    return {
        "u1": u1,
        "u2": u2,
        "seed": torch.tensor(int(step_seed), device=tensor.device, dtype=torch.int64),
        "gaussian_seed": torch.tensor(int(gaussian_seed), device=tensor.device, dtype=torch.int64),
        "perturb_seed": torch.tensor(int(perturb_seed), device=tensor.device, dtype=torch.int64),
        "update_seed": torch.tensor(int(update_seed), device=tensor.device, dtype=torch.int64),
        "state_seed": torch.tensor(int(state_seed), device=tensor.device, dtype=torch.int64),
    }


def quantize_model_in_place(
    model: nn.Module,
    bits: int,
    *,
    include_frozen: bool = True,
    seed: int = 0,
    algorithm: Optional[str] = "per_tensor_symmetric",
    group_size: int = 0,
    block_size: int = 0,
) -> None:
    bits = validate_quzo_bits(bits)
    if bits == 32:
        return
    if bits == 16:
        model.half()
        return
    with torch.no_grad():
        for name, param in model.named_parameters():
            if (not include_frozen) and (not param.requires_grad):
                continue
            q_seed = _seed_from_parts(seed, name, "model_init")
            param.data.copy_(
                quantize_tensor(
                    param.data,
                    bits,
                    seed=q_seed,
                    target_dtype=param.data.dtype,
                    algorithm=algorithm,
                    group_size=group_size,
                    block_size=block_size,
                )
            )
