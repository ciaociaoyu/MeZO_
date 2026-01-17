

"""
自动批量生成作业 txt 文件的小脚本。

功能概述：
1. 读取当前目录下的模板作业文件 `h1e-3.txt`；
2. 在 1e-2 到 1e-6 之间，按照「每个相邻指数区间（例如 1e-2 到 1e-3）采样 10 个 log 均匀点」的规则生成 EPS 值；
3. 去掉采样过程中产生的重复点（例如两个区间的公共端点 1e-3、1e-4、1e-5 等）；
4. 将所有去重后的 EPS 值按顺序每 9 个分成一组，每一组生成一个作业 txt 文件；
   - 作业文件名使用该组的第 1 个和第 9 个 EPS 值进行标识；
   - 同时在 SBATCH 的 job-name 中也用这两个值进行标识；
5. 将模板中的
   - `#SBATCH --job-name=sst-5_fixH1e-3_1118` 中的 `fixH1e-3` 替换为该组对应的英文标识；
   - `#SBATCH --time=15:00:00` 修改为 `#SBATCH --time=48:00:00`；
   - 将原本唯一一行的 TASK 命令复制 9 次，并把其中的 `EPS=1e-3` 分别替换为这一组的 9 个 EPS 值。

注意：
- 由于 1e-2 到 1e-6 按照 4 个区间（1e-2~1e-3、1e-3~1e-4、1e-4~1e-5、1e-5~1e-6）各采样 10 个点，一共得到 40 个点，去掉重复点后通常不是 9 的整倍数。
- 为了保证「每个作业文件对应 9 个 EPS」这一要求，本脚本只使用最前面的 (len(eps_list) // 9) * 9 个 EPS，最后多出来的 1 个值会被丢弃。
"""

import numpy as np
from pathlib import Path
from typing import List


def generate_eps_values(num_per_interval: int = 10, tol: float = 1e-12) -> list:
    """
    在 1e-2 到 1e-6 的范围内生成 EPS 取值，并且在 log 空间上均匀。

    具体规则：
    - 指数从 -2 到 -6，每次间隔 1 个指数：
      [-2, -3], [-3, -4], [-4, -5], [-5, -6] 共 4 个区间；
    - 对每个区间使用 np.logspace 采样 num_per_interval 个点（包含两端点）；
    - 为避免在相邻区间端点处重复（比如 1e-3 同时出现在前一个区间的末尾和下一个区间的开头），
      使用一个简单的“近似相等”判断（|a-b| < tol）进行去重。

    参数：
        num_per_interval: 每个指数区间采样的点数，题目要求为 10。
        tol: 判断两个浮点数是否相等的容差。

    返回：
        eps_list: 一个按从大到小（从 1e-2 到 1e-6）排序的 EPS 浮点数列表，且不含重复值。
    """
    eps_list: List[float] = []

    # 指数从 -2 递减到 -5（因为每次使用 [start, start-1] 这个区间）
    for exp_start in range(-2, -6, -1):
        exp_end = exp_start - 1
        # 在 [exp_start, exp_end] 区间内均匀取 num_per_interval 个指数点
        exponents = np.linspace(exp_start, exp_end, num_per_interval)
        # 将指数转换为真实的 10**exponent 值
        values = 10.0 ** exponents

        for v in values:
            # 判断 v 是否与已有的某个 eps 非常接近，如果接近则认为是重复点
            if not any(abs(v - old_v) < tol for old_v in eps_list):
                eps_list.append(float(v))

    return eps_list


def format_eps_for_name(eps: float) -> str:
    """
    将 EPS 浮点数格式化成适合放进文件名和 job-name 中的字符串。

    这里采用科学计数法，保留类似 1e-02、7e-03 这样的形式：
        - 使用 `"{:.0e}"`，例如：0.01 -> '1e-02'。
        - 去掉可能出现的 `+` 号，以免影响文件名可读性。
    """
    return f"{eps:.0e}".replace("+", "")


def format_eps_for_cmd(eps: float) -> str:
    """
    将 EPS 浮点数格式化成适合写进命令行参数的字符串。

    这里采用 "{:.6e}"，与示例中类似：
    例如：0.01 -> '1.000000e-02'。
    """
    return f"{eps:.6e}"


def load_template(template_path: Path) -> str:
    """
    读取模板 txt 作业文件的内容。

    参数：
        template_path: 模板文件的路径，例如 Path("h1e-3.txt")。

    返回：
        模板文件的完整文本内容。
    """
    if not template_path.exists():
        raise FileNotFoundError(f"模板文件不存在：{template_path}")

    return template_path.read_text(encoding="utf-8")


def build_job_text(template_text: str, eps_group: List[float]) -> str:
    """
    根据一组 EPS 值，构造一个新的作业 txt 文件内容。

    主要修改三部分：
    1. 修改 job-name：
       - 原行中包含 `sst-5_fixH1e-3_1118`，我们只将其中的 `fixH1e-3` 替换成形如
         `eps1e-02_to_1e-03` 的标识。
    2. 修改运行时长：
       - 将 `#SBATCH --time=15:00:00` 替换为 `#SBATCH --time=48:00:00`。
    3. 修改 TASK 命令：
       - 找到模板中唯一一行以 `TASK=sst-5` 开头的命令；
       - 复制 9 份（len(eps_group) 份），把每一行中的 `EPS=1e-3` 替换为对应的 EPS 值。

    参数：
        template_text: 模板 txt 的原始文本。
        eps_group: 一组 EPS 值（长度为 9）。

    返回：
        替换完毕后的新作业文本。
    """
    if len(eps_group) == 0:
        raise ValueError("eps_group 为空，无法构造作业文本。")

    # 1. 构造 job-name 中使用的标识，使用该组中的第 1 个和第 9 个值
    first_eps_name = format_eps_for_name(eps_group[0])
    last_eps_name = format_eps_for_name(eps_group[-1])
    job_tag = f"eps{first_eps_name}_to_{last_eps_name}"

    # 将 job-name 中的 fixH1e-3 替换为新的 job_tag
    new_text = template_text.replace("fixH1e-3", job_tag)

    # 2. 修改时间为 48 小时
    new_text = new_text.replace("#SBATCH --time=15:00:00", "#SBATCH --time=48:00:00")

    # 3. 处理 TASK 命令行
    lines = new_text.splitlines()
    task_line = None
    for line in lines:
        # 去掉行首空白再判断是否以 TASK=sst-5 开头
        if line.strip().startswith("TASK=sst-5"):
            task_line = line
            break

    if task_line is None:
        raise ValueError("在模板中没有找到以 'TASK=sst-5' 开头的命令行，请检查模板文件。")

    # 构造新的多行 TASK 命令，每一行替换不同的 EPS 值
    task_lines = []
    for eps in eps_group:
        eps_str = format_eps_for_cmd(eps)
        # 假设模板行中原本包含 `EPS=1e-3`，这里进行替换
        new_task = task_line.replace("EPS=1e-3", f"EPS={eps_str}")
        task_lines.append(new_task)

    task_block = "\n".join(task_lines)

    # 用新的多行 TASK 命令替换掉原来的单行命令
    new_text = new_text.replace(task_line, task_block)

    return new_text


def main():
    """
    主函数：
    1. 生成 EPS 取值列表；
    2. 按每 9 个一组进行分组；
    3. 对每一组调用 build_job_text 生成新的作业文本；
    4. 将结果写入到当前目录下的 `jobs_eps` 子目录中。
    """
    # 当前脚本所在目录
    base_dir = Path(__file__).resolve().parent

    # 模板文件路径，假定与本脚本在同一目录下，并且文件名为 h1e-3.txt
    template_path = base_dir / "h1e-3.txt"

    # 输出目录：用于存放自动生成的作业 txt 文件
    output_dir = base_dir / "jobs_eps"
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. 生成所有 EPS 取值
    eps_list = generate_eps_values(num_per_interval=10)

    # 为了方便调试，可以打印一下总数量：
    print(f"总共生成的去重后 EPS 数量: {len(eps_list)}")

    # 2. 按每 9 个一组进行切分
    group_size = 9
    total_full_groups = len(eps_list) // group_size

    if total_full_groups == 0:
        raise RuntimeError("有效的 EPS 组数为 0，请检查采样参数。")

    # 如果不能整除，最后多出来的一部分会被丢弃，这里给出提示
    leftover = len(eps_list) - total_full_groups * group_size
    if leftover > 0:
        print(f"注意：共有 {len(eps_list)} 个 EPS 值，按每 {group_size} 个一组只会使用前 "
              f"{total_full_groups * group_size} 个，剩余 {leftover} 个将被丢弃。")

    # 3. 读取模板文本
    template_text = load_template(template_path)

    # 4. 遍历每一组 EPS，生成对应的作业文件
    for group_idx in range(total_full_groups):
        start = group_idx * group_size
        end = start + group_size
        eps_group = eps_list[start:end]

        # 生成该组对应的作业文本
        job_text = build_job_text(template_text, eps_group)

        # 构造该组对应的名字：包含第一个与第九个 EPS 值
        first_eps_name = format_eps_for_name(eps_group[0])
        last_eps_name = format_eps_for_name(eps_group[-1])
        job_tag = f"eps{first_eps_name}_to_{last_eps_name}"

        # 输出文件名完全使用英文与数字，例如：job_eps1e-02_to_1e-03.txt
        job_filename = f"job_{job_tag}.txt"
        job_path = output_dir / job_filename

        # 将文本写入文件
        job_path.write_text(job_text, encoding="utf-8")

        print(f"生成作业文件: {job_path}")


if __name__ == "__main__":
    main()