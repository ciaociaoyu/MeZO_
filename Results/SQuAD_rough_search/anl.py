import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 汇总不同 eps 下的最优 train/eval loss，用于画 eps-vs-best-loss 总结图
TRAIN_POINTS = []  # 列表元素形如 (eps_float, best_train_loss)
EVAL_POINTS = []   # 列表元素形如 (eps_float, best_eval_loss)

# ==========================
# 配置区域（按需修改）
# ==========================
# CSV 文件所在目录，默认当前目录
# 可以改成绝对路径，例如：
# DATA_DIR = "/home/jy03364/MeZO/Results/SQuAD"
DATA_DIR = "./"

# 匹配文件名示例：
# metrics_SQuAD-opt-1.3b-sampleeval-1-ndev-1-eps0.1-mezo-ft-20000-16-1e-7-1e-1-0.csv
# 这里把 eps 的匹配改得更通用，支持 0.1 / 1e-1 / 1E-1 等格式
# 注意 eps 捕获不包含后面的连字符，保证可以被 float() 正确解析
FILENAME_PATTERN = re.compile(r"metrics_([^-\s]+)-.*eps([0-9.eE+]+)", re.IGNORECASE)


def parse_task_and_eps(filename: str):
    """
    从文件名中解析任务名和 eps 值。
    例如：metrics_SQuAD-opt-1.3b-...-eps0.1-...csv
    -> task = "SQuAD", eps = "0.1"
    """
    base = os.path.basename(filename)
    m = FILENAME_PATTERN.search(base)
    if not m:
        # 没匹配到就给一个兜底
        task = "TASK"
        eps = "NA"
    else:
        task = m.group(1)
        eps = m.group(2)
    return task, eps


def find_loss_columns(df: pd.DataFrame):
    """
    自动在列名里寻找 train loss / eval loss 列。
    优先使用常见命名 train_loss / eval_loss，
    否则模糊匹配：列名同时包含 train 和 loss / eval 和 loss。
    仅用于“列式”记录（每列一个 loss）。
    """
    train_col = None
    eval_col = None

    # 先直接尝试常见名称
    if "train_loss" in df.columns:
        train_col = "train_loss"
    if "eval_loss" in df.columns:
        eval_col = "eval_loss"

    # 再尝试一些常见变体
    if train_col is None:
        for c in df.columns:
            lc = c.lower()
            if lc in ["loss_train", "trainloss"]:
                train_col = c
                break

    if eval_col is None:
        for c in df.columns:
            lc = c.lower()
            if lc in ["loss_eval", "loss_dev", "evalloss", "dev_loss"]:
                eval_col = c
                break

    # 如果还没找到，再做模糊匹配
    if train_col is None:
        for c in df.columns:
            lc = c.lower()
            if "train" in lc and "loss" in lc:
                train_col = c
                break

    if eval_col is None:
        for c in df.columns:
            lc = c.lower()
            if ("eval" in lc or "dev" in lc) and "loss" in lc:
                eval_col = c
                break

    return train_col, eval_col


def plot_from_split_format(df: pd.DataFrame, task: str, eps: str):
    """
    处理“行式”记录的情况：
    兼容两种常见结构：
    1）epoch, split/train/eval, loss
    2）epoch, phase(train/eval), metric(loss/acc/...), value

    会自动拆成 train / eval 两条曲线。
    """
    # 使用全局列表记录每个 eps 的最优 loss
    global TRAIN_POINTS, EVAL_POINTS

    # 统一列名小写，防止大小写混乱
    cols_lower = {c.lower(): c for c in df.columns}

    # 哪一列表示 train / eval / dev
    split_col_name = cols_lower.get("split") or cols_lower.get("type") or cols_lower.get("phase")
    # 哪一列表示“这一行是什么指标”（如 loss / acc）
    metric_col_name = cols_lower.get("metric")
    # 哪一列是真正的数值（loss 值）
    loss_value_col_name = cols_lower.get("loss") or cols_lower.get("value")

    if split_col_name is None or loss_value_col_name is None:
        # 保险 return，理论上只有在检查过才会调用到这里
        print("  ❌ split/phase 或 loss/value 列缺失，无法按行式记录绘图。")
        print(f"     当前列名: {list(df.columns)}")
        return

    df_use = df

    # 如果有 metric 列，就只保留 metric 为 loss 的行
    if metric_col_name is not None:
        df_use = df_use[df_use[metric_col_name].astype(str).str.lower().str.contains("loss")]

    # epoch 列
    if "epoch" in df_use.columns:
        epoch_col_name = "epoch"
    elif "step" in df_use.columns:
        epoch_col_name = "step"
    else:
        # 没有 epoch/step 就用行号
        df_use = df_use.copy()
        df_use["__idx__"] = range(1, len(df_use) + 1)
        epoch_col_name = "__idx__"

    # 按 split/phase 分两组
    split_series = df_use[split_col_name].astype(str).str.lower()
    df_train = df_use[split_series.str.contains("train")]
    df_eval = df_use[split_series.str.contains("eval") | split_series.str.contains("dev")]

    # 可以按 epoch 取均值，避免一个 epoch 多条记录
    if not df_train.empty:
        train_group = df_train.groupby(epoch_col_name)[loss_value_col_name].mean().reset_index()
    else:
        train_group = None

    if not df_eval.empty:
        eval_group = df_eval.groupby(epoch_col_name)[loss_value_col_name].mean().reset_index()
    else:
        eval_group = None

    # 记录该 eps 下的最优 train/eval loss（按 epoch 分组后的最小值）
    try:
        eps_float = float(eps)
    except Exception:
        eps_float = None

    if eps_float is not None:
        if train_group is not None and not train_group.empty:
            best_train = train_group[loss_value_col_name].min()
            TRAIN_POINTS.append((eps_float, best_train))
        if eval_group is not None and not eval_group.empty:
            best_eval = eval_group[loss_value_col_name].min()
            EVAL_POINTS.append((eps_float, best_eval))

    title_base = f"{task}+{eps}"
    safe_eps = str(eps).replace(".", "_").replace("+", "").replace("-", "m")

    # 画 train loss
    if train_group is not None and not train_group.empty:
        plt.figure()
        plt.plot(train_group[epoch_col_name], train_group[loss_value_col_name])
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(f"{title_base} - Train Loss")
        plt.grid(True)

        out_name = f"{task}_eps{safe_eps}_train_loss.png"
        out_path = os.path.join(DATA_DIR, out_name)
        plt.savefig(out_path, bbox_inches="tight")
        plt.close()
        print(f"  ✅ 保存 train loss 图(行式): {out_path}")
    else:
        print("  ⚠️ 行式记录中未找到 train 行，跳过 train 图。")

    # 画 eval loss
    if eval_group is not None and not eval_group.empty:
        plt.figure()
        plt.plot(eval_group[epoch_col_name], eval_group[loss_value_col_name])
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(f"{title_base} - Eval Loss")
        plt.grid(True)

        out_name = f"{task}_eps{safe_eps}_eval_loss.png"
        out_path = os.path.join(DATA_DIR, out_name)
        plt.savefig(out_path, bbox_inches="tight")
        plt.close()
        print(f"  ✅ 保存 eval loss 图(行式): {out_path}")
    else:
        print("  ⚠️ 行式记录中未找到 eval/dev 行，跳过 eval 图。")


def plot_single_file(csv_path: str):
    """
    读取单个 csv 文件，画两张图：
    1. train loss vs epoch
    2. eval loss vs epoch
    并保存为 png 图片。

    同时兼容两种常见格式：
    A. 列式：epoch, train_loss, eval_loss, ...
    B. 行式：epoch, split(train/eval), loss, ...
    """
    print(f"处理文件: {csv_path}")
    df = pd.read_csv(csv_path)

    # 解析任务名和 eps
    task, eps = parse_task_and_eps(csv_path)

    # 先判断是否为“行式”记录（包含 phase/split/type + loss/value 这种纵向记录）
    cols_lower = {c.lower(): c for c in df.columns}
    split_col_name = cols_lower.get("split") or cols_lower.get("type") or cols_lower.get("phase")
    loss_value_col_name = cols_lower.get("loss") or cols_lower.get("value")
    metric_col_name = cols_lower.get("metric")

    has_row_format = (split_col_name is not None) and (loss_value_col_name is not None)

    # 若满足条件，则按行式处理（例如列名为: time, step, epoch, phase, metric, value, ...）
    if has_row_format:
        plot_from_split_format(df, task, eps)
        return

    # ===== 下面是“列式”记录处理逻辑 =====

    # 获取 epoch
    if "epoch" in df.columns:
        epochs = df["epoch"]
    elif "step" in df.columns:
        epochs = df["step"]
    else:
        # 如果没有 epoch/step 列，就用行号代替，从 1 开始
        epochs = range(1, len(df) + 1)

    # 自动找 loss 列
    train_col, eval_col = find_loss_columns(df)

    # 列式记录下，直接用整列的最小值作为该 eps 的最优 loss
    global TRAIN_POINTS, EVAL_POINTS
    try:
        eps_float = float(eps)
    except Exception:
        eps_float = None

    if eps_float is not None:
        if train_col is not None:
            best_train = df[train_col].min()
            TRAIN_POINTS.append((eps_float, best_train))
        if eval_col is not None:
            best_eval = df[eval_col].min()
            EVAL_POINTS.append((eps_float, best_eval))

    # 如果找不到对应的列，就给出提示
    if train_col is None and eval_col is None:
        print(f"  ❌ 找不到 train loss / eval loss 列，跳过该文件: {csv_path}")
        print(f"     当前列名: {list(df.columns)}")
        return

    title_base = f"{task}+{eps}"
    safe_eps = str(eps).replace(".", "_").replace("+", "").replace("-", "m")

    # 画 train loss
    if train_col is not None:
        plt.figure()
        plt.plot(epochs, df[train_col])
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(f"{title_base} - Train Loss")
        plt.grid(True)

        out_name = f"{task}_eps{safe_eps}_train_loss.png"
        out_path = os.path.join(DATA_DIR, out_name)
        plt.savefig(out_path, bbox_inches="tight")
        plt.close()
        print(f"  ✅ 保存 train loss 图(列式): {out_path}")
    else:
        print(f"  ⚠️ 未找到 train loss 列，跳过 train 图: {csv_path}")

    # 画 eval loss
    if eval_col is not None:
        plt.figure()
        plt.plot(epochs, df[eval_col])
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(f"{title_base} - Eval Loss")
        plt.grid(True)

        out_name = f"{task}_eps{safe_eps}_eval_loss.png"
        out_path = os.path.join(DATA_DIR, out_name)
        plt.savefig(out_path, bbox_inches="tight")
        plt.close()
        print(f"  ✅ 保存 eval loss 图(列式): {out_path}")
    else:
        print(f"  ⚠️ 未找到 eval loss 列，跳过 eval 图: {csv_path}")


def main():
    # 遍历目录下所有 csv 文件
    files = [
        f for f in os.listdir(DATA_DIR)
        if f.endswith(".csv") and f.startswith("metrics_")
    ]

    if not files:
        print("当前目录下没有找到 metrics_*.csv 文件。")
        return

    for f in files:
        csv_path = os.path.join(DATA_DIR, f)
        plot_single_file(csv_path)

    # 所有文件处理完后，画 eps vs 最优 loss 的两张汇总图
    plot_best_vs_eps()


# 新增函数：画 eps vs 最优 loss 汇总图
def plot_best_vs_eps():
    """
    汇总不同 eps 下的最优 train/eval loss，分别画两张图：
    1) 横坐标为 eps，纵坐标为 best train loss
    2) 横坐标为 eps，纵坐标为 best eval loss
    """
    # 去重 & 按 eps 排序（如果同一个 eps 多次出现，取最小值）
    if TRAIN_POINTS:
        # 聚合同一 eps 的最小 train loss
        best_train_map = {}
        for e, l in TRAIN_POINTS:
            if e not in best_train_map:
                best_train_map[e] = l
            else:
                best_train_map[e] = min(best_train_map[e], l)
        eps_list_train = sorted(best_train_map.keys())
        loss_list_train = [best_train_map[e] for e in eps_list_train]

        plt.figure()
        plt.plot(eps_list_train, loss_list_train, marker="o")
        plt.xlabel("eps")
        plt.ylabel("Best Train Loss")
        plt.title("Best Train Loss vs eps")
        plt.grid(True)
        plt.xscale("log", base=10)
        ax = plt.gca()
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"$10^{{{int(round(np.log10(v)))}}}$" if v > 0 else ""))
        out_path = os.path.join(DATA_DIR, "best_train_loss_vs_eps.png")
        plt.savefig(out_path, bbox_inches="tight")
        plt.close()
        print(f"  ✅ 保存汇总图(Train): {out_path}")

    if EVAL_POINTS:
        # 聚合同一 eps 的最小 eval loss
        best_eval_map = {}
        for e, l in EVAL_POINTS:
            if e not in best_eval_map:
                best_eval_map[e] = l
            else:
                best_eval_map[e] = min(best_eval_map[e], l)
        eps_list_eval = sorted(best_eval_map.keys())
        loss_list_eval = [best_eval_map[e] for e in eps_list_eval]

        plt.figure()
        plt.plot(eps_list_eval, loss_list_eval, marker="o")
        plt.xlabel("eps")
        plt.ylabel("Best Eval Loss")
        plt.title("Best Eval Loss vs eps")
        plt.grid(True)
        plt.xscale("log", base=10)
        ax = plt.gca()
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"$10^{{{int(round(np.log10(v)))}}}$" if v > 0 else ""))
        out_path = os.path.join(DATA_DIR, "best_eval_loss_vs_eps.png")
        plt.savefig(out_path, bbox_inches="tight")
        plt.close()
        print(f"  ✅ 保存汇总图(Eval): {out_path}")


if __name__ == "__main__":
    main()