from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import torch
import torch.nn.functional as F


def signed_qrange(bits: int) -> Tuple[int, int]:
    if bits < 2:
        raise ValueError(f"bits must be >= 2 for signed symmetric lattice, got {bits}")
    return -(2 ** (bits - 1)), 2 ** (bits - 1) - 1


@dataclass
class GroupwiseQuantizedWeight:
    """Explicit groupwise signed integer lattice for a 2D Linear weight.

    This is a surrogate explicit lattice initialized from GPTQ-dequantized
    weights. It is not a mutation of native packed GPTQ storage.
    """

    name: str
    codes: torch.Tensor
    scales: torch.Tensor
    original_shape: Tuple[int, int]
    bits: int
    group_size: int
    qmin: int
    qmax: int
    scale_policy: str = "fixed"

    @classmethod
    def from_weight(
        cls,
        name: str,
        weight: torch.Tensor,
        *,
        bits: int,
        group_size: int,
        scale_policy: str = "fixed",
        eps: float = 1e-12,
    ) -> "GroupwiseQuantizedWeight":
        if weight.ndim != 2:
            raise ValueError(f"expected 2D Linear weight for {name}, got shape {tuple(weight.shape)}")
        if group_size <= 0:
            raise ValueError(f"group_size must be positive, got {group_size}")
        if scale_policy not in {"fixed", "recompute"}:
            raise ValueError(f"unsupported scale_policy {scale_policy}")

        qmin, qmax = signed_qrange(bits)
        w = weight.detach().float()
        out_features, in_features = w.shape
        pad = (-in_features) % group_size
        if pad:
            w_pad = F.pad(w, (0, pad), value=0.0)
        else:
            w_pad = w
        grouped = w_pad.reshape(out_features, -1, group_size)
        denom = float(max(abs(qmin), abs(qmax)))
        max_abs = grouped.abs().amax(dim=-1, keepdim=True)
        scales = (max_abs / denom).clamp_min(eps)
        q_pad = torch.round(grouped / scales).clamp(qmin, qmax).to(torch.int16)
        codes = q_pad.reshape(out_features, -1)[:, :in_features].contiguous()
        return cls(
            name=name,
            codes=codes,
            scales=scales.contiguous(),
            original_shape=(out_features, in_features),
            bits=bits,
            group_size=group_size,
            qmin=qmin,
            qmax=qmax,
            scale_policy=scale_policy,
        )

    @property
    def device(self) -> torch.device:
        return self.codes.device

    @property
    def numel(self) -> int:
        return int(self.codes.numel())

    def clone(self) -> "GroupwiseQuantizedWeight":
        return GroupwiseQuantizedWeight(
            name=self.name,
            codes=self.codes.clone(),
            scales=self.scales.clone(),
            original_shape=self.original_shape,
            bits=self.bits,
            group_size=self.group_size,
            qmin=self.qmin,
            qmax=self.qmax,
            scale_policy=self.scale_policy,
        )

    def to(self, device: torch.device | str) -> "GroupwiseQuantizedWeight":
        return GroupwiseQuantizedWeight(
            name=self.name,
            codes=self.codes.to(device),
            scales=self.scales.to(device),
            original_shape=self.original_shape,
            bits=self.bits,
            group_size=self.group_size,
            qmin=self.qmin,
            qmax=self.qmax,
            scale_policy=self.scale_policy,
        )

    def expanded_scales(self, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        out_features, in_features = self.original_shape
        expanded = self.scales.to(dtype=dtype).repeat_interleave(self.group_size, dim=2)
        return expanded.reshape(out_features, -1)[:, :in_features].contiguous()

    def dequantize(self, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        return (self.codes.to(dtype=torch.float32) * self.expanded_scales()).to(dtype=dtype)

    def with_codes(self, codes: torch.Tensor) -> "GroupwiseQuantizedWeight":
        if tuple(codes.shape) != self.original_shape:
            raise ValueError(f"codes shape {tuple(codes.shape)} != {self.original_shape}")
        return GroupwiseQuantizedWeight(
            name=self.name,
            codes=codes.clamp(self.qmin, self.qmax).to(torch.int16).contiguous(),
            scales=self.scales.clone(),
            original_shape=self.original_shape,
            bits=self.bits,
            group_size=self.group_size,
            qmin=self.qmin,
            qmax=self.qmax,
            scale_policy=self.scale_policy,
        )

    def requantize(self, weight: torch.Tensor) -> "GroupwiseQuantizedWeight":
        if self.scale_policy == "recompute":
            return GroupwiseQuantizedWeight.from_weight(
                self.name,
                weight,
                bits=self.bits,
                group_size=self.group_size,
                scale_policy=self.scale_policy,
            )
        scales = self.expanded_scales(device_dtype(weight))
        q = torch.round(weight.detach().float() / scales.float()).clamp(self.qmin, self.qmax)
        return self.with_codes(q)

    def saturation_fraction(self) -> float:
        q = self.codes
        return float(((q == self.qmin) | (q == self.qmax)).float().mean().item())

    def state_dict(self) -> Dict[str, object]:
        return {
            "name": self.name,
            "codes": self.codes.cpu(),
            "scales": self.scales.cpu(),
            "original_shape": self.original_shape,
            "bits": self.bits,
            "group_size": self.group_size,
            "qmin": self.qmin,
            "qmax": self.qmax,
            "scale_policy": self.scale_policy,
        }

    @classmethod
    def from_state_dict(cls, state: Dict[str, object]) -> "GroupwiseQuantizedWeight":
        return cls(
            name=str(state["name"]),
            codes=state["codes"],
            scales=state["scales"],
            original_shape=tuple(state["original_shape"]),
            bits=int(state["bits"]),
            group_size=int(state["group_size"]),
            qmin=int(state["qmin"]),
            qmax=int(state["qmax"]),
            scale_policy=str(state.get("scale_policy", "fixed")),
        )


def device_dtype(tensor: torch.Tensor) -> torch.dtype:
    return torch.float32 if tensor.dtype not in (torch.float16, torch.bfloat16, torch.float32, torch.float64) else tensor.dtype
