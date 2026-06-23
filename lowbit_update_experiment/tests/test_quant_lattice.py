import torch

from lowbit_lattice.quant import GroupwiseQuantizedWeight
from lowbit_lattice.update_rules import apply_update_rule


def test_groupwise_quantize_dequantize_non_divisible_shape():
    w = torch.randn(7, 130) * 0.1
    q = GroupwiseQuantizedWeight.from_weight("linear.weight", w, bits=4, group_size=128)
    assert q.codes.shape == w.shape
    assert q.scales.shape == (7, 2, 1)
    assert q.codes.min() >= q.qmin
    assert q.codes.max() <= q.qmax
    wq = q.dequantize()
    assert wq.shape == w.shape
    q2 = q.requantize(wq)
    assert torch.equal(q.codes, q2.codes)


def test_nearest_rounding_drops_subgrid_updates():
    w = torch.ones(2, 8)
    q = GroupwiseQuantizedWeight.from_weight("w", w, bits=4, group_size=4)
    scale = q.expanded_scales()
    grad = torch.ones_like(w)
    lr = float(scale.min().item() * 0.1)
    res = apply_update_rule(q, grad, lr=lr, rule="nearest_requant_fixed_grid")
    assert torch.equal(res.q_old, res.q_new)
    assert torch.count_nonzero(res.w_new - res.w_old).item() == 0
