# Quantization dead-zone 验证实验（hprobe风格）
# ------------------------------------------------------------
# 目标：
# 1) 定义一个可控的一维/高维 toy loss（可让一阶导变化更剧烈：通过 omega 控制频率/导数幅度）
# 2) 固定 seed，设置“很低精度”的参数存储（默认用 float16；也支持 k-bit 均匀量化）
# 3) 对每个 h（步长），采样多条随机方向 z：
#       D(h) = [f(theta + h z) - f(theta - h z)] / (2h)          (finite-diff 方向导)
#       G    = <∇f(theta), z>                                     (true 方向导)
#    观察 MSE(h) = mean_z (D(h) - G)^2 的形状，并同时记录：
#       - param_changed_ratio(h): perturb 后有多少参数真的发生了“量化后”的改变（死区会接近 0）
#       - h_eff(h): ||theta_pert - theta|| / ||z||（有效步长，会在死区塌到 0）
#
# 运行方式：
#   python this_file.py
# 你可以改 Config 里的 store_mode / omega_list / theta_scale / h_min,h_max 等。
# ------------------------------------------------------------

import math
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import matplotlib.pyplot as plt


# ----------------------------
# 配置
# ----------------------------
@dataclass
class Config:
    seed: int = 0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # 参数维度（越大越接近期望，越小跑得越快）
    d: int = 4096

    # base 参数尺度：越大，float16 的 ULP 越大 => 死区更明显（更早出现）
    theta_scale: float = 0.01

    # 方向数（hprobe 的 ndir）
    ndir: int = 64

    # h 列表（hprobe 默认：每个 decade 取 {1,3}）
    h_min: float = 1e-8
    h_max: float = 1e-2

    # 量化/低精度模式：
    #   "float16": 用 float16 存储参数，扰动后再 cast 回 float16（最贴近你们 hprobe 的“低精度死区”）
    #   "bfloat16": 同理
    #   "kbit": 均匀 k-bit 量化（更“低精度”，死区更粗）
    store_mode: str = "float16"  # "float16" | "bfloat16" | "kbit"

    # 仅在 store_mode="kbit" 时使用
    kbits: int = 4
    clip: float = 2.0  # 量化截断范围 [-clip, clip]

    # 让一阶导变化更剧烈：omega 越大，导数幅度与局部振荡越强
    # 你可以只放一个 omega；也可以多放几个对比曲线
    omega_list: List[float] = None  # e.g. [5.0, 20.0, 80.0]

    # 打印每个 h 的日志
    verbose: bool = True


# ----------------------------
# 工具：h 列表（模仿 trainer.py 的 _hprobe_h_list）
# ----------------------------
def make_h_list(h_min: float, h_max: float) -> List[float]:
    if not (math.isfinite(h_min) and math.isfinite(h_max) and h_min > 0 and h_max > 0):
        h_min, h_max = 1e-8, 1e-2
    if h_min > h_max:
        h_min, h_max = h_max, h_min

    emin = int(math.floor(math.log10(h_min)))
    emax = int(math.ceil(math.log10(h_max)))
    hs: List[float] = []
    for e in range(emin, emax + 1):
        for m in (1.0, 3.0):
            h = m * (10.0 ** e)
            if h_min <= h <= h_max:
                hs.append(float(h))
    hs = sorted(set(hs))
    if len(hs) == 0:
        hs = [float(h_min), float(h_max)]
    return hs


# ----------------------------
# k-bit 均匀量化（对称 signed）
# 返回：量化后的 float32 张量 + int codes（方便精确比较是否 changed）
# ----------------------------
def uniform_quantize_kbit(x: torch.Tensor, bits: int, clip: float) -> Tuple[torch.Tensor, torch.Tensor, float]:
    assert bits >= 2, "bits too small"
    qmax = (2 ** (bits - 1)) - 1  # e.g. bits=4 => qmax=7
    scale = float(clip) / float(qmax)

    x_clipped = torch.clamp(x, -clip, clip)
    q = torch.round(x_clipped / scale).to(torch.int32)
    q = torch.clamp(q, -qmax, qmax)

    xq = q.to(torch.float32) * scale
    return xq, q, scale


# ----------------------------
# 构造可控 toy loss（让一阶导变化“剧烈”）
# 这里用 sin/cos（数值稳定），通过 omega 控制频率与导数幅度
#
# loss(theta) = mean( (w * sin(omega*theta + phi))^2 )
# 梯度幅度 ~ O(omega)
# ----------------------------
class ToyLoss:
    def __init__(self, d: int, device: str, seed: int):
        g = torch.Generator(device=device)
        g.manual_seed(seed + 12345)
        self.w = torch.randn(d, device=device, dtype=torch.float32, generator=g)
        self.phi = (2.0 * math.pi) * torch.rand(d, device=device, dtype=torch.float32, generator=g)

    def __call__(self, theta_f32: torch.Tensor, omega: float) -> torch.Tensor:
        # theta_f32: float32
        y = torch.sin(theta_f32 * float(omega) + self.phi)
        loss = torch.mean((self.w * y) ** 2)
        return loss


# ----------------------------
# 低精度/量化“存储”与“加扰动”的封装
#   base_store: 用 store 模式表示的 theta（可能是 float16/bfloat16/kbit 量化后 float32）
#   add_perturb(base_store, h, z): 返回 perturb 后的 store 表示
# 同时提供 changed_ratio / h_eff 等诊断
# ----------------------------
class Storage:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        mode = cfg.store_mode.lower().strip()
        self.mode = mode
        if mode == "float16":
            self.store_dtype = torch.float16
        elif mode == "bfloat16":
            self.store_dtype = torch.bfloat16
        elif mode == "kbit":
            self.store_dtype = torch.float32
        else:
            raise ValueError(f"Unknown store_mode={cfg.store_mode}")

        # kbit scale（运行时根据 clip/bits 决定）
        self.kbit_scale: Optional[float] = None

    def quantize_base(self, theta_fp32: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        返回一个 dict，里面至少有:
          - "theta_store": 存储态 theta（float16/bf16 或 kbit 量化后 float32）
        kbit 模式额外返回:
          - "theta_code": int32 code
        """
        if self.mode in ("float16", "bfloat16"):
            theta_store = theta_fp32.to(self.store_dtype)
            return {"theta_store": theta_store}
        else:
            xq, code, scale = uniform_quantize_kbit(theta_fp32, bits=self.cfg.kbits, clip=self.cfg.clip)
            self.kbit_scale = scale
            return {"theta_store": xq, "theta_code": code}

    def sample_direction(self, shape: torch.Size, device: str, seed: int) -> Dict[str, torch.Tensor]:
        """
        模仿 hprobe：z 的 dtype 跟参数存储 dtype 一致（float16/bf16），kbit 则用 float32。
        """
        torch.manual_seed(seed)
        if self.mode in ("float16", "bfloat16"):
            z = torch.normal(mean=0.0, std=1.0, size=shape, device=device, dtype=self.store_dtype)
            return {"z_store": z}
        else:
            z = torch.normal(mean=0.0, std=1.0, size=shape, device=device, dtype=torch.float32)
            return {"z_store": z}

    def add_perturb(self, base: Dict[str, torch.Tensor], h: float, z: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        返回 perturb 后的“存储态”theta。
        """
        theta_store = base["theta_store"]
        z_store = z["z_store"]

        if self.mode in ("float16", "bfloat16"):
            # 关键：在低精度 dtype 里做加法（会出现“加了也没变”的死区）
            # （这基本等价于：theta' = cast_to_lowp(theta + h*z)）
            theta_p = (theta_store + (float(h) * z_store)).to(self.store_dtype)
            return {"theta_store": theta_p}

        else:
            # k-bit：先加，再量化
            theta_fp = theta_store + (float(h) * z_store)
            xq, code, _ = uniform_quantize_kbit(theta_fp, bits=self.cfg.kbits, clip=self.cfg.clip)
            return {"theta_store": xq, "theta_code": code}

    def to_float_for_loss(self, store: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        loss 计算统一用 float32（避免把“算子低精度误差”混进来；只看参数量化/存储误差）
        """
        return store["theta_store"].to(torch.float32)

    def changed_ratio(self, base: Dict[str, torch.Tensor], pert: Dict[str, torch.Tensor]) -> float:
        if self.mode == "kbit":
            base_code = base["theta_code"]
            pert_code = pert["theta_code"]
            return float((base_code != pert_code).to(torch.float32).mean().item())
        else:
            base_t = base["theta_store"]
            pert_t = pert["theta_store"]
            return float((base_t != pert_t).to(torch.float32).mean().item())

    def h_eff(self, base: Dict[str, torch.Tensor], pert: Dict[str, torch.Tensor], z: Dict[str, torch.Tensor]) -> float:
        # h_eff = ||theta_pert - theta|| / ||z||，都用 float32
        diff = (pert["theta_store"].to(torch.float32) - base["theta_store"].to(torch.float32)).reshape(-1)
        zz = z["z_store"].to(torch.float32).reshape(-1)
        denom = torch.norm(zz, p=2).item()
        if denom <= 0:
            return float("nan")
        return float(torch.norm(diff, p=2).item() / denom)


# ----------------------------
# 核心实验（hprobe风格）
# ----------------------------
@torch.no_grad()
def finite_diff_directional(
    loss_fn: ToyLoss,
    storage: Storage,
    base_store: Dict[str, torch.Tensor],
    z_store: Dict[str, torch.Tensor],
    h: float,
    omega: float,
) -> Tuple[float, float, float]:
    """
    返回 (fp, fm, d_fd)
      fp = f(theta + h z)
      fm = f(theta - h z)
      d_fd = (fp - fm) / (2h)
    """
    tp = storage.add_perturb(base_store, +h, z_store)
    tm = storage.add_perturb(base_store, -h, z_store)
    fp = float(loss_fn(storage.to_float_for_loss(tp), omega=omega).item())
    fm = float(loss_fn(storage.to_float_for_loss(tm), omega=omega).item())
    d_fd = (fp - fm) / (2.0 * float(h))
    return fp, fm, float(d_fd)


def true_directional_grad(
    loss_fn: ToyLoss,
    storage: Storage,
    base_store: Dict[str, torch.Tensor],
    z_store: Dict[str, torch.Tensor],
    omega: float,
) -> Tuple[float, torch.Tensor]:
    """
    计算 G = <∇f(theta), z>
    其中 theta 取“存储态 base_store 对应的 float32 值”（即量化后落点处的真实梯度）
    """
    theta = storage.to_float_for_loss(base_store).detach().clone().requires_grad_(True)
    loss = loss_fn(theta, omega=omega)
    (g,) = torch.autograd.grad(loss, theta, retain_graph=False, create_graph=False)
    # dot in float32
    zf = z_store["z_store"].to(torch.float32)
    d_true = float(torch.sum(g * zf).item())
    return d_true, g.detach()


def run_one_setting(cfg: Config, omega: float) -> Dict[str, List[float]]:
    # 固定随机性
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    device = cfg.device
    d = cfg.d

    # toy loss
    loss_fn = ToyLoss(d=d, device=device, seed=cfg.seed)

    # base theta (float32)，再进入“低精度存储”
    g = torch.Generator(device=device)
    g.manual_seed(cfg.seed + 7)
    theta_fp32 = (cfg.theta_scale * torch.randn(d, device=device, dtype=torch.float32, generator=g))

    storage = Storage(cfg)
    base_store = storage.quantize_base(theta_fp32)

    # 方向 seeds（固定）
    rs = np.random.RandomState(cfg.seed + 999)
    dir_seeds = rs.randint(0, 1_000_000_000, size=cfg.ndir, dtype=np.int64).tolist()

    # h 列表（hprobe风格）
    hs = make_h_list(cfg.h_min, cfg.h_max)

    # 结果容器
    out = {
        "h": [],
        "mse": [],
        "mae": [],
        "d_true_mean": [],
        "d_fd_mean": [],
        "changed_ratio_mean": [],
        "h_eff_mean": [],
        "dead_frac": [],  # D_fd==0 的比例（一个简单的“死区”指示）
    }

    # 为了做“理论对照”：死区极限下 d_fd≈0 => MSE≈E[d_true^2]
    # 我们在每个 h 里都会采样 d_true_list，因此也可以直接看到 plateau。

    for h in hs:
        d_true_list: List[float] = []
        d_fd_list: List[float] = []
        err_list: List[float] = []
        cr_list: List[float] = []
        heff_list: List[float] = []
        dead_cnt = 0

        for s in dir_seeds:
            z_store = storage.sample_direction(shape=base_store["theta_store"].shape, device=device, seed=int(s))

            # true directional derivative
            d_true, _ = true_directional_grad(loss_fn, storage, base_store, z_store, omega=omega)

            # finite diff directional derivative
            fp, fm, d_fd = finite_diff_directional(loss_fn, storage, base_store, z_store, h=float(h), omega=omega)

            # 诊断：这条方向下 +h 是否真的改变了参数（量化/低精度后）
            tp = storage.add_perturb(base_store, +h, z_store)
            cr = storage.changed_ratio(base_store, tp)
            heff = storage.h_eff(base_store, tp, z_store)

            d_true_list.append(d_true)
            d_fd_list.append(d_fd)
            err = d_fd - d_true
            err_list.append(err)
            cr_list.append(cr)
            heff_list.append(heff)
            if d_fd == 0.0:
                dead_cnt += 1

        err_arr = np.asarray(err_list, dtype=np.float64)
        mse = float(np.mean(err_arr ** 2))
        mae = float(np.mean(np.abs(err_arr)))
        dtrue_mean = float(np.mean(np.asarray(d_true_list, dtype=np.float64)))
        dfd_mean = float(np.mean(np.asarray(d_fd_list, dtype=np.float64)))
        cr_mean = float(np.mean(np.asarray(cr_list, dtype=np.float64)))
        heff_mean = float(np.mean(np.asarray(heff_list, dtype=np.float64)))
        dead_frac = float(dead_cnt / max(1, len(dir_seeds)))

        out["h"].append(float(h))
        out["mse"].append(mse)
        out["mae"].append(mae)
        out["d_true_mean"].append(dtrue_mean)
        out["d_fd_mean"].append(dfd_mean)
        out["changed_ratio_mean"].append(cr_mean)
        out["h_eff_mean"].append(heff_mean)
        out["dead_frac"].append(dead_frac)

        if cfg.verbose:
            print(
                f"[omega={omega:6.2f}] h={h:.3e}  "
                f"MSE={mse:.3e}  MAE={mae:.3e}  "
                f"changed_ratio={cr_mean:.3e}  h_eff={heff_mean:.3e}  dead_frac(D==0)={dead_frac:.2f}"
            )

    return out


def plot_results(all_results: Dict[float, Dict[str, List[float]]], cfg: Config):
    # 1) MSE vs h
    plt.figure()
    for omega, res in all_results.items():
        plt.loglog(res["h"], res["mse"], marker="o", label=f"omega={omega:g}")
    plt.xlabel("h")
    plt.ylabel("MSE[(D-G)^2] across directions")
    plt.title(f"MSE vs h   (store_mode={cfg.store_mode}, d={cfg.d}, ndir={cfg.ndir})")
    plt.legend()
    plt.grid(True, which="both", ls="--", alpha=0.3)

    # 2) changed_ratio vs h（死区诊断）
    plt.figure()
    for omega, res in all_results.items():
        plt.semilogx(res["h"], res["changed_ratio_mean"], marker="o", label=f"omega={omega:g}")
    plt.xlabel("h")
    plt.ylabel("mean param_changed_ratio")
    plt.title("param_changed_ratio vs h (dead-zone indicator)")
    plt.legend()
    plt.grid(True, which="both", ls="--", alpha=0.3)

    # 3) h_eff vs h（有效步长）
    plt.figure()
    for omega, res in all_results.items():
        plt.loglog(res["h"], res["h_eff_mean"], marker="o", label=f"omega={omega:g}")
    plt.xlabel("h")
    plt.ylabel("mean h_eff = ||theta_pert-theta|| / ||z||")
    plt.title("h_eff vs h")
    plt.legend()
    plt.grid(True, which="both", ls="--", alpha=0.3)

    # 4) dead_frac vs h（D_fd==0 的比例）
    plt.figure()
    for omega, res in all_results.items():
        plt.semilogx(res["h"], res["dead_frac"], marker="o", label=f"omega={omega:g}")
    plt.xlabel("h")
    plt.ylabel("fraction of directions with D_fd==0")
    plt.title("dead_frac vs h")
    plt.legend()
    plt.grid(True, which="both", ls="--", alpha=0.3)

    plt.show()


def main():
    cfg = Config()

    if cfg.omega_list is None:
        # 你可以只留一个 omega；或者加多个 omega 对比“一阶导更剧烈”时 MSE 曲线如何变化
        cfg.omega_list = [5.0, 20.0, 80.0]

    print("===== Config =====")
    print(cfg)
    print("==================")

    all_results: Dict[float, Dict[str, List[float]]] = {}
    for omega in cfg.omega_list:
        res = run_one_setting(cfg, omega=float(omega))
        all_results[float(omega)] = res

    plot_results(all_results, cfg)


if __name__ == "__main__":
    main()