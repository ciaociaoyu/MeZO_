import importlib.util
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "tools" / "rtnclip_int8_mse_reprobe.py"
SPEC = importlib.util.spec_from_file_location("rtnclip_int8_mse_reprobe", SCRIPT_PATH)
probe = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(probe)


def test_identity_quantizer_visibility_nmse_zero():
    delta_ideal = torch.tensor([0.1, -0.2, 0.3], dtype=torch.float32)
    stats = probe.visibility_stats(delta_ideal.clone(), delta_ideal)
    assert stats["delta_visibility_nmse"] < 1e-12
    assert abs(stats["alignment"] - 1.0) < 1e-6
    assert abs(stats["norm_ratio"] - 1.0) < 1e-6


def test_central_difference_quadratic_matches_true_directional_derivative():
    w = torch.tensor([0.4, -0.7, 1.1], dtype=torch.float64)
    u = torch.tensor([0.2, -0.3, 0.5], dtype=torch.float64)
    a = torch.diag(torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64))

    def f(x):
        return 0.5 * x @ a @ x

    true = float((a @ w).dot(u))
    for h in (1e-5, 1e-3, 1e-1):
        fd = float((f(w + h * u) - f(w - h * u)) / (2.0 * h))
        assert abs(fd - true) < 1e-10


def test_cubic_central_difference_error_grows_with_h():
    w = torch.tensor([0.4, -0.7, 1.1], dtype=torch.float64)
    u = torch.tensor([0.2, -0.3, 0.5], dtype=torch.float64)

    def f(x):
        return torch.sum(x**3)

    true = float((3.0 * w.square()).dot(u))
    small_h = 1e-4
    large_h = 1e-1
    fd_small = float((f(w + small_h * u) - f(w - small_h * u)) / (2.0 * small_h))
    fd_large = float((f(w + large_h * u) - f(w - large_h * u)) / (2.0 * large_h))
    assert abs(fd_large - true) > abs(fd_small - true)


def test_toy_quantizer_separates_visibility_from_locality():
    direction = torch.ones(16, dtype=torch.float32)

    def fake_round(x):
        return torch.round(x)

    small_h = 0.1
    large_h = 0.6
    small_delta = fake_round(small_h * direction) - fake_round(-small_h * direction)
    large_delta = fake_round(large_h * direction) - fake_round(-large_h * direction)
    small_stats = probe.visibility_stats(small_delta, 2.0 * small_h * direction)
    large_stats = probe.visibility_stats(large_delta, 2.0 * large_h * direction)
    assert small_stats["delta_visibility_nmse"] > large_stats["delta_visibility_nmse"]


def test_rtnclip_shared_state_fresh_rounding_no_bypass():
    qrw = probe.qrw
    weight = torch.zeros((1, 4), dtype=torch.float16)
    state, _stats = qrw.compute_quantizer_state(
        "toy.weight",
        weight,
        quantizer="rtnclip",
        bitwidth=8,
        group_size=4,
        activation_rms=None,
    )
    direction = torch.ones_like(weight).float()
    h = 0.6
    scale_ptr = state.scales.data_ptr()
    q_plus = qrw.quantize_with_state(weight.float().add(direction, alpha=h), state)
    q_minus = qrw.quantize_with_state(weight.float().add(direction, alpha=-h), state)
    assert state.scales.data_ptr() == scale_ptr
    assert torch.count_nonzero(q_plus - q_minus).item() == weight.numel()
    assert not torch.allclose(q_plus.float(), weight.float().add(direction, alpha=h))
