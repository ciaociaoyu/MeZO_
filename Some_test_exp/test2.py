"""
Modified contrast experiment (as you asked):

We want TWO regimes that visibly separate:
A) theta change SMALL, but gradient ENERGY (mean(G^2)) changes BIG over time
B) theta change BIG, but gradient ENERGY changes SMALL (≈ constant)

Key fixes vs your previous run:
- Old A (pure 1-cos(omega*theta)) has almost constant gradient energy when theta barely moves.
  -> New A introduces a time-varying amplitude schedule A_t (or omega_t), so grad energy changes
     even if theta barely moves.
- Old B had grad = a/d, so mean(G^2) ≈ 1/d ~ 2e-4 (tiny) and constant.
  -> New B uses grad = a (no /d), so mean(G^2) ~ O(1) and constant, while theta drifts big.

We keep h_probe=1e-4, store=float16 by default to show dead-zone => MSE(D-G) ≈ mean(G^2).
"""

import math
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import matplotlib.pyplot as plt


# ----------------------------
# Config
@dataclass
class Cfg:
    seed: int = 0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    d: int = 4096# ----------------------------

    steps: int = 300
    probe_every: int = 5

    # hprobe
    h_probe: float = 1e-4
    n_probe: int = 64
    reuse_probe_dirs: bool = True

    # storage / quantization
    store_mode: str = "float16"  # "float16" | "bfloat16" | "kbit"
    kbits: int = 4
    clip: float = 10.0  # for kbit

    # -------- Regime A (theta small change, grad big change) --------
    # Base oscillatory loss with *time-varying amplitude* At:
    #   L_A(t,theta) = mean( At * (1 - cos(omega * theta)) )
    #   grad_A = At * (omega) * sin(omega*theta) / d
    omega_A: float = 2000.0
    theta0_A_scale: float = 1.0
    lr_A: float = 1e-7  # keep tiny -> theta barely moves

    # amplitude schedule: At varies over time (big change in grad energy)
    A_min: float = 0.05
    A_max: float = 30.0
    A_schedule: str = "exp"  # "exp" | "cos" | "linear"

    # -------- Regime B (theta big change, grad small change) --------
    # We will CONTROL theta magnitude directly: theta_t = alpha_t * theta_dir
    # so that |theta| grows from small to large.
    theta_dir_scale_B: float = 1.0  # scale for the base direction vector theta_dir
    alpha_B_min: float = 1e-3       # start small
    alpha_B_max: float = 1e2        # grow large
    alpha_B_schedule: str = "exp"  # "exp" | "linear"

    a_scale_B: float = 1.0  # scale of a -> controls constant grad magnitude

    verbose: bool = True


# ----------------------------
# k-bit uniform quantizer
# ----------------------------
def uniform_quantize_kbit(x: torch.Tensor, bits: int, clip: float) -> Tuple[torch.Tensor, torch.Tensor, float]:
    assert bits >= 2
    qmax = (2 ** (bits - 1)) - 1
    scale = float(clip) / float(qmax)
    xc = torch.clamp(x, -clip, clip)
    q = torch.round(xc / scale).to(torch.int32)
    q = torch.clamp(q, -qmax, qmax)
    xq = q.to(torch.float32) * scale
    return xq, q, scale


# ----------------------------
# mean ULP for float16/bfloat16
# ----------------------------
def mean_ulp(x_lowp: torch.Tensor) -> float:
    if x_lowp.dtype not in (torch.float16, torch.bfloat16):
        return float("nan")
    try:
        inf = torch.full_like(x_lowp, float("inf"))
        y = torch.nextafter(x_lowp, inf)
        return float((y - x_lowp).abs().to(torch.float32).mean().item())
    except Exception:
        x = x_lowp.to(torch.float32).abs()
        if x_lowp.dtype == torch.float16:
            mant_bits, min_norm, sub_ulp = 10, 2.0 ** (-14), 2.0 ** (-24)
        else:
            mant_bits, min_norm, sub_ulp = 7, 2.0 ** (-126), 2.0 ** (-133)
        _, exp = torch.frexp(torch.where(x == 0, torch.ones_like(x), x))
        ulp_norm = torch.pow(torch.tensor(2.0, device=x.device), (exp - (mant_bits + 1)).to(torch.float32))
        ulp = torch.where(x >= min_norm, ulp_norm, torch.full_like(ulp_norm, sub_ulp))
        return float(ulp.mean().item())


# ----------------------------
# Storage wrapper (float16/bf16/kbit)
# ----------------------------
class Storage:
    def __init__(self, cfg: Cfg):
        self.cfg = cfg
        self.mode = cfg.store_mode.lower().strip()
        if self.mode == "float16":
            self.store_dtype = torch.float16
        elif self.mode == "bfloat16":
            self.store_dtype = torch.bfloat16
        elif self.mode == "kbit":
            self.store_dtype = torch.float32
        else:
            raise ValueError(f"Unknown store_mode={cfg.store_mode}")
        self.kbit_scale: Optional[float] = None

    def quantize(self, theta_fp32: torch.Tensor) -> Dict[str, torch.Tensor]:
        if self.mode in ("float16", "bfloat16"):
            return {"theta_store": theta_fp32.to(self.store_dtype)}
        xq, code, scale = uniform_quantize_kbit(theta_fp32, bits=self.cfg.kbits, clip=self.cfg.clip)
        self.kbit_scale = scale
        return {"theta_store": xq, "theta_code": code}

    def to_float(self, store: Dict[str, torch.Tensor]) -> torch.Tensor:
        return store["theta_store"].to(torch.float32)

    def sample_dir(self, shape: torch.Size, device: str, seed: int) -> Dict[str, torch.Tensor]:
        torch.manual_seed(int(seed))
        if self.mode in ("float16", "bfloat16"):
            z = torch.normal(0.0, 1.0, size=shape, device=device, dtype=self.store_dtype)
        else:
            z = torch.normal(0.0, 1.0, size=shape, device=device, dtype=torch.float32)
        return {"z_store": z}

    def add_perturb(self, base: Dict[str, torch.Tensor], h: float, z: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        th = base["theta_store"]
        zz = z["z_store"]
        if self.mode in ("float16", "bfloat16"):
            h_t = torch.tensor(float(h), device=th.device, dtype=self.store_dtype)
            out = (th + h_t * zz).to(self.store_dtype)
            return {"theta_store": out}
        else:
            out_fp = th + float(h) * zz
            xq, code, _ = uniform_quantize_kbit(out_fp, bits=self.cfg.kbits, clip=self.cfg.clip)
            return {"theta_store": xq, "theta_code": code}

    def changed_ratio(self, base: Dict[str, torch.Tensor], pert: Dict[str, torch.Tensor]) -> float:
        if self.mode == "kbit":
            return float((base["theta_code"] != pert["theta_code"]).float().mean().item())
        return float((base["theta_store"] != pert["theta_store"]).float().mean().item())

    def h_eff(self, base: Dict[str, torch.Tensor], pert: Dict[str, torch.Tensor], z: Dict[str, torch.Tensor]) -> float:
        diff = (pert["theta_store"].to(torch.float32) - base["theta_store"].to(torch.float32)).reshape(-1)
        zz = z["z_store"].to(torch.float32).reshape(-1)
        denom = torch.norm(zz, p=2).item()
        if denom <= 0:
            return float("nan")
        return float(torch.norm(diff, p=2).item() / denom)

    def grid_step(self, base: Dict[str, torch.Tensor]) -> float:
        if self.mode == "kbit":
            return float(self.kbit_scale if self.kbit_scale is not None else float("nan"))
        return mean_ulp(base["theta_store"])


# ----------------------------
# Regime A: time-varying amplitude schedule
# ----------------------------
def amplitude_schedule(cfg: Cfg, t: int) -> float:
    T = max(1, cfg.steps)
    if cfg.A_schedule == "exp":
        # geometric progression from A_min -> A_max
        r = t / T
        return cfg.A_min * ((cfg.A_max / cfg.A_min) ** r)
    if cfg.A_schedule == "linear":
        r = t / T
        return cfg.A_min + r * (cfg.A_max - cfg.A_min)
    if cfg.A_schedule == "cos":
        # oscillate between A_min and A_max
        r = 0.5 * (1.0 - math.cos(2.0 * math.pi * t / T))
        return cfg.A_min + r * (cfg.A_max - cfg.A_min)
    raise ValueError(f"Unknown A_schedule={cfg.A_schedule}")


@torch.no_grad()
def loss_grad_A(theta_f32: torch.Tensor, omega: float, At: float) -> Tuple[torch.Tensor, torch.Tensor]:
    d = theta_f32.numel()
    x = theta_f32 * float(omega)
    loss = float(At) * (1.0 - torch.cos(x)).mean()
    grad = (float(At) * float(omega) / float(d)) * torch.sin(x)
    return loss, grad


# ----------------------------
# Regime B: linear with constant gradient (NO /d)
# L_B = sum a_i theta_i  (we can still compute a scalar loss)
# grad = a (constant)
# ----------------------------
@torch.no_grad()
def loss_grad_B(theta_f32: torch.Tensor, a: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    loss = torch.sum(a * theta_f32)  # scalar
    grad = a  # constant
    return loss, grad


# ----------------------------
# directional FD with quantized perturbations
# ----------------------------
@torch.no_grad()
def directional_fd(storage: Storage, base: Dict[str, torch.Tensor], z: Dict[str, torch.Tensor],
                   h: float, loss_grad_fn) -> float:
    tp = storage.add_perturb(base, +h, z)
    tm = storage.add_perturb(base, -h, z)
    fp, _ = loss_grad_fn(storage.to_float(tp))
    fm, _ = loss_grad_fn(storage.to_float(tm))
    return float(((fp - fm) / (2.0 * float(h))).item())


# ----------------------------
# hprobe stats at current theta
# ----------------------------
@torch.no_grad()
def probe(cfg: Cfg, storage: Storage, base: Dict[str, torch.Tensor], probe_seeds: List[int],
          loss_grad_fn) -> Dict[str, float]:
    theta_q = storage.to_float(base)
    loss, grad = loss_grad_fn(theta_q)

    errs, g2s = [], []
    crs, heffs = [], []
    dead = 0

    for s in probe_seeds:
        z = storage.sample_dir(base["theta_store"].shape, cfg.device, int(s))
        D = directional_fd(storage, base, z, cfg.h_probe, loss_grad_fn)
        zf = z["z_store"].to(torch.float32).reshape(-1)
        G = float(torch.dot(grad.reshape(-1), zf).item())

        e = D - G
        errs.append(e)
        g2s.append(G * G)
        if D == 0.0:
            dead += 1

        tp = storage.add_perturb(base, +cfg.h_probe, z)
        crs.append(storage.changed_ratio(base, tp))
        heffs.append(storage.h_eff(base, tp, z))

    err = np.asarray(errs, dtype=np.float64)
    return {
        "loss": float(loss.item()),
        "theta_norm": float(torch.norm(theta_q).item()),
        "grad_norm": float(torch.norm(grad).item()),
        "mse": float(np.mean(err ** 2)),
        "mean_g2": float(np.mean(np.asarray(g2s, dtype=np.float64))),
        "dead_frac": float(dead / max(1, len(probe_seeds))),
        "changed_ratio": float(np.mean(np.asarray(crs, dtype=np.float64))),
        "h_eff": float(np.mean(np.asarray(heffs, dtype=np.float64))),
        "grid_step": float(storage.grid_step(base)),
        "mean_abs_theta": float(theta_q.abs().mean().item()),
    }


# ----------------------------
# run regime A
# ----------------------------
def run_regime_A(cfg: Cfg):
    storage = Storage(cfg)
    g = torch.Generator(device=cfg.device)
    g.manual_seed(cfg.seed + 7)
    theta0 = cfg.theta0_A_scale * torch.randn(cfg.d, device=cfg.device, dtype=torch.float32, generator=g)
    base = storage.quantize(theta0)

    rs = np.random.RandomState(cfg.seed + 999)
    probe_seeds = rs.randint(0, 1_000_000_000, size=cfg.n_probe, dtype=np.int64).tolist()

    def lg(th, t):
        At = amplitude_schedule(cfg, t)
        return loss_grad_A(th, omega=cfg.omega_A, At=At)

    logs = {"t": [], "loss": [], "theta_norm": [], "grad_norm": [], "mse": [], "mean_g2": [],
            "dtheta": [], "dgrad": [], "dead_frac": [], "changed_ratio": [], "grid_step": [], "At": []}

    prev_theta = storage.to_float(base).clone()
    _, prev_grad = lg(prev_theta, 0)

    for t in range(cfg.steps + 1):
        if t % cfg.probe_every == 0:
            def lg_fixed(th):
                return lg(th, t)

            st = probe(cfg, storage, base, probe_seeds, lg_fixed)
            cur_theta = storage.to_float(base)
            _, cur_grad = lg(cur_theta, t)

            logs["t"].append(t)
            logs["loss"].append(st["loss"])
            logs["theta_norm"].append(st["theta_norm"])
            logs["grad_norm"].append(st["grad_norm"])
            logs["mse"].append(st["mse"])
            logs["mean_g2"].append(st["mean_g2"])
            logs["dead_frac"].append(st["dead_frac"])
            logs["changed_ratio"].append(st["changed_ratio"])
            logs["grid_step"].append(st["grid_step"])
            logs["At"].append(amplitude_schedule(cfg, t))

            logs["dtheta"].append(float(torch.norm(cur_theta - prev_theta).item()))
            logs["dgrad"].append(float(torch.norm(cur_grad - prev_grad).item()))
            prev_theta, prev_grad = cur_theta.clone(), cur_grad.clone()

            if cfg.verbose:
                print(f"[A t={t:4d}] At={logs['At'][-1]:.3e}  ||Δθ||={logs['dtheta'][-1]:.3e}  "
                      f"||Δg||={logs['dgrad'][-1]:.3e}  ||θ||={st['theta_norm']:.3e}  ||g||={st['grad_norm']:.3e}  "
                      f"MSE={st['mse']:.3e}  mean(G^2)={st['mean_g2']:.3e}")

        if t == cfg.steps:
            break

        th = storage.to_float(base)
        _, grad = lg(th, t)
        th_new = th - float(cfg.lr_A) * grad
        base = storage.quantize(th_new)

    return logs


# ----------------------------
# run regime B
# ----------------------------
def alpha_schedule_B(cfg: Cfg, t: int) -> float:
    """Schedule for Regime B: alpha grows from alpha_B_min to alpha_B_max."""
    T = max(1, cfg.steps)
    r = t / T
    if cfg.alpha_B_schedule == "exp":
        return cfg.alpha_B_min * ((cfg.alpha_B_max / cfg.alpha_B_min) ** r)
    if cfg.alpha_B_schedule == "linear":
        return cfg.alpha_B_min + r * (cfg.alpha_B_max - cfg.alpha_B_min)
    raise ValueError(f"Unknown alpha_B_schedule={cfg.alpha_B_schedule}")


def run_regime_B(cfg: Cfg):
    storage = Storage(cfg)
    g = torch.Generator(device=cfg.device)
    g.manual_seed(cfg.seed + 17)
    theta_dir = cfg.theta_dir_scale_B * torch.randn(cfg.d, device=cfg.device, dtype=torch.float32, generator=g)
    theta_fp32 = float(alpha_schedule_B(cfg, 0)) * theta_dir

    # constant "a" => constant gradient in float32
    g2 = torch.Generator(device=cfg.device)
    g2.manual_seed(cfg.seed + 12345)
    a = cfg.a_scale_B * torch.randn(cfg.d, device=cfg.device, dtype=torch.float32, generator=g2)

    rs = np.random.RandomState(cfg.seed + 999)
    probe_seeds = rs.randint(0, 1_000_000_000, size=cfg.n_probe, dtype=np.int64).tolist()

    def lg(th):
        return loss_grad_B(th, a=a)

    logs = {"t": [], "alpha": [], "loss": [], "theta_norm": [], "grad_norm": [], "mse": [], "mean_g2": [],
            "dtheta": [], "dgrad": [], "dead_frac": [], "changed_ratio": [], "grid_step": []}

    prev_theta = theta_fp32.clone()
    _, prev_grad = lg(prev_theta)

    for t in range(cfg.steps + 1):
        # Controlled growth of |theta|: theta_t = alpha_t * theta_dir
        theta_fp32 = float(alpha_schedule_B(cfg, t)) * theta_dir
        if t % cfg.probe_every == 0:
            # Quantize a temporary copy ONLY for probing / dead-zone measurements
            base = storage.quantize(theta_fp32)
            st = probe(cfg, storage, base, probe_seeds, lg)
            cur_theta = theta_fp32
            _, cur_grad = lg(cur_theta)

            logs["t"].append(t)
            logs["alpha"].append(float(alpha_schedule_B(cfg, t)))
            logs["loss"].append(st["loss"])
            logs["theta_norm"].append(st["theta_norm"])
            logs["grad_norm"].append(st["grad_norm"])
            logs["mse"].append(st["mse"])
            logs["mean_g2"].append(st["mean_g2"])
            logs["dead_frac"].append(st["dead_frac"])
            logs["changed_ratio"].append(st["changed_ratio"])
            logs["grid_step"].append(st["grid_step"])

            logs["dtheta"].append(float(torch.norm(cur_theta - prev_theta).item()))
            logs["dgrad"].append(float(torch.norm(cur_grad - prev_grad).item()))
            prev_theta, prev_grad = cur_theta.clone(), cur_grad.clone()

            if cfg.verbose:
                print(f"[B t={t:4d}] ||Δθ||={logs['dtheta'][-1]:.3e}  ||Δg||={logs['dgrad'][-1]:.3e}  "
                      f"||θ||={st['theta_norm']:.3e}  ||g||={st['grad_norm']:.3e}  "
                      f"MSE={st['mse']:.3e}  mean(G^2)={st['mean_g2']:.3e}")

        if t == cfg.steps:
            break

    return logs


# ----------------------------
# plot compare
# ----------------------------
def plot_compare(logA, logB, cfg: Cfg):
    """Plot only two summary figures (A/B) and include changed_ratio."""
    tA = np.asarray(logA["t"])
    tB = np.asarray(logB["t"])
    cr_eps = 1e-12

    # Regime A summary
    crA = np.asarray(logA["changed_ratio"], dtype=np.float64)
    crA_plot = np.maximum(crA, cr_eps)
    plt.figure()
    plt.semilogy(tA, np.asarray(logA["mse"]), marker="o", label="MSE(D-G)")
    plt.semilogy(tA, np.asarray(logA["grad_norm"]), marker="x", label="||g||")
    plt.semilogy(tA, np.asarray(logA["theta_norm"]), marker="s", label="||theta||")
    plt.semilogy(tA, crA_plot, marker="^", label=f"changed_ratio (floored at {cr_eps:g})")
    plt.xlabel("iteration")
    plt.ylabel("value (log)")
    plt.title(f"Regime A: MSE, ||g||, ||theta||, changed_ratio  (h_probe={cfg.h_probe}, store={cfg.store_mode})")
    plt.grid(True, which="both", ls="--", alpha=0.3)
    plt.legend()

    # Regime B summary
    crB = np.asarray(logB["changed_ratio"], dtype=np.float64)
    crB_plot = np.maximum(crB, cr_eps)
    plt.figure()
    plt.semilogy(tB, np.asarray(logB["mse"]), marker="o", label="MSE(D-G)")
    plt.semilogy(tB, np.asarray(logB["grad_norm"]), marker="x", label="||g||")
    plt.semilogy(tB, np.asarray(logB["theta_norm"]), marker="s", label="||theta||")
    plt.semilogy(tB, crB_plot, marker="^", label=f"changed_ratio (floored at {cr_eps:g})")
    plt.xlabel("iteration")
    plt.ylabel("value (log)")
    plt.title(f"Regime B: MSE, ||g||, ||theta||, changed_ratio  (h_probe={cfg.h_probe}, store={cfg.store_mode})")
    plt.grid(True, which="both", ls="--", alpha=0.3)
    plt.legend()

    plt.show()


def main():
    cfg = Cfg()
    print("===== Config =====")
    print(cfg)
    print("==================")

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    logA = run_regime_A(cfg)
    logB = run_regime_B(cfg)

    plot_compare(logA, logB, cfg)


if __name__ == "__main__":
    main()
