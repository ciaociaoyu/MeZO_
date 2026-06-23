import torch

from lowbit_lattice.quant import GroupwiseQuantizedWeight
from lowbit_lattice.update_rules import apply_update_rule


def test_topk_code_flip_changes_exact_k_when_unsaturated():
    w = torch.zeros(10, 10)
    q = GroupwiseQuantizedWeight.from_weight("w", w, bits=4, group_size=5)
    q = q.with_codes(torch.zeros_like(q.codes))
    grad = -torch.arange(100, dtype=torch.float32).reshape(10, 10) - 1.0
    res = apply_update_rule(q, grad, lr=1.0, rule="topk_code_flip", k_frac=0.1)
    changed = torch.count_nonzero(res.q_new != res.q_old).item()
    assert changed == 10


def test_dense_stochastic_code_flip_changes_some_codes():
    w = torch.zeros(20, 20)
    q = GroupwiseQuantizedWeight.from_weight("w", w, bits=4, group_size=10)
    q = q.with_codes(torch.zeros_like(q.codes))
    grad = -torch.ones_like(w)
    gen = torch.Generator().manual_seed(123)
    res = apply_update_rule(q, grad, lr=1.0, rule="dense_stochastic_code_flip", p_max=0.1, generator=gen)
    changed = torch.count_nonzero(res.q_new != res.q_old).item()
    assert 10 < changed < 80


def test_stochastic_rounding_expectation_is_close():
    w = torch.zeros(64, 64)
    q = GroupwiseQuantizedWeight.from_weight("w", w, bits=4, group_size=16)
    q.scales.fill_(1.0)
    q = q.with_codes(torch.zeros_like(q.codes))
    grad = -torch.ones_like(w)
    means = []
    for seed in range(80):
        gen = torch.Generator().manual_seed(seed)
        res = apply_update_rule(q, grad, lr=0.25, rule="stochastic_round_fixed_grid", generator=gen)
        means.append((res.w_new - res.w_old).mean().item())
    assert abs(sum(means) / len(means) - 0.25) < 0.04
