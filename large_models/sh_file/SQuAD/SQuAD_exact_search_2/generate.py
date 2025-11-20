"""
自动批量生成 SQuAD + OPT 作业 txt 文件的小脚本。

新版功能概述：
1. 读取当前目录下的模板作业文件（例如 `1.58e-3AND1e-3.txt`）；
2. 在 1e-2.5 到 1e-4 之间，在对数空间（指数上）均匀采样若干个 EPS 值；
3. 将所有 EPS 值按顺序每 2 个分成一组，每一组生成一个作业 txt 文件；
   - 每个作业文件内包含 2 行运行命令（2 个不同的 EPS）；
4. 参考模板中的 job-name：
   `#SBATCH --job-name=SQuAD_optH1.58e-3AND1e-3_1120`
   自动将其中的 `optH1.58e-3AND1e-3` 替换为本组对应的
   `optH{eps1}AND{eps2}`，并将日期后缀更新为当天（MMDD，例如 1120）；
5. 模板中的 `EPS=1.58e-3` 会被替换为对应组内的 EPS 值。

注意：
- EPS 的采样区间为 [1e-2.5, 1e-4]（指数从 -2.5 到 -4），在 log 空间上均匀；
- 每个作业文件只放 2 个 EPS，对应 2 行运行命令。
"""

import numpy as np
from pathlib import Path
from typing import List
from datetime import datetime


def generate_eps_values(num_per_interval: int = 14, tol: float = 1e-12) -> list:
    """
    在 1e-2.5 到 1e-4 的范围内生成 EPS 取值，并且在 log 空间上均匀。

    具体规则：
    - 指数从 -2.5 到 -4.0，使用 np.linspace 在该区间上均匀取 num_per_interval 个点；
    - 然后将指数转换为 10**exponent 得到真实的 EPS 值。

    参数：
        num_per_interval: 在整个 [-2.5, -4.0] 指数区间采样的点数。
        tol: 保留参数（目前未使用，兼容旧接口）。

    返回：
        eps_list: 一个按从大到小（从 1e-2.5 到 1e-4）排序的 EPS 浮点数列表。
    """
    exponents = np.linspace(-2.5, -4.0, num_per_interval)
    values = 10.0 ** exponents
    eps_list: List[float] = [float(v) for v in values]
    return eps_list


def format_eps_for_name(eps: float) -> str:
    """
    将 EPS 浮点数格式化成适合放进文件名和 job-name 中的字符串。

    这里采用三位有效数字的科学计数法，比如：
        0.00158 -> '1.58e-03'
    """
    return f"{eps:.3g}"


def format_eps_for_cmd(eps: float) -> str:
    """
    将 EPS 浮点数格式化成适合写进命令行参数的字符串。

    这里也采用三位有效数字的科学计数法，例如：
        0.00158 -> '1.58e-03'
    """
    return f"{eps:.3g}"


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
       - 模板中包含形如
         `#SBATCH --job-name=SQuAD_optH1.58e-3AND1e-3_1120`
         的一行；
       - 我们将其中的 `SQuAD_optH1.58e-3AND1e-3_1120` 替换为
         `SQuAD_optH{eps1}AND{eps2}_{MMDD}`，其中 {MMDD} 为当天日期。
    2. 保持运行时长不变（仍然是 48:00:00，如模板所示）；
    3. 修改运行命令：
       - 找到模板中第一行包含 `TASK=SQuAD` 且包含 `EPS=` 的命令行；
       - 删除所有包含 `TASK=SQuAD` 且包含 `EPS=` 的旧命令行；
       - 用该模板行复制 len(eps_group) 份（这里为 2），并把其中的
         `EPS=1.58e-3` 替换为对应的 EPS 值。

    参数：
        template_text: 模板 txt 的原始文本。
        eps_group: 一组 EPS 值（长度为 2）。

    返回：
        替换完毕后的新作业文本。
    """
    if len(eps_group) == 0:
        raise ValueError("eps_group 为空，无法构造作业文本。")

    if len(eps_group) != 2:
        print(f"警告：期望每组有 2 个 EPS，但当前组大小为 {len(eps_group)}。")

    # 构造 job-name 使用的标识
    first_eps_name = format_eps_for_name(eps_group[0])
    second_eps_name = format_eps_for_name(eps_group[1])
    date_str = datetime.now().strftime("%m%d")
    new_job_name = f"SQuAD_optH{first_eps_name}AND{second_eps_name}_{date_str}"

    # 1. 修改 job-name：直接替换整段 SQuAD_* 名称
    # 模板中原行为：#SBATCH --job-name=SQuAD_optH1.58e-3AND1e-3_1120 ...
    # 为避免依赖日期，先找到以 "#SBATCH --job-name=SQuAD_" 开头的那一行。
    lines = template_text.splitlines()
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("#SBATCH --job-name=SQuAD_"):
            # 保留注释中可能存在的其他内容（例如 # Job name ...）
            # 只替换 job-name 的值部分。
            prefix, _, _ = stripped.partition("#SBATCH --job-name=")
            # 找到原始 job-name 的完整部分
            _, _, after = stripped.partition("#SBATCH --job-name=")
            # after 里可能还带有注释，用空格切出第一个 token 作为 job-name
            original_name = after.split()[0]
            # 用 new_job_name 替换 original_name
            new_line = line.replace(original_name, new_job_name)
            lines[i] = new_line
            break
    new_text = "\n".join(lines)

    # 2. 处理 TASK 行：删除原有的 SQuAD+EPS 行，然后用模板行生成新的多行
    lines = new_text.splitlines()
    base_lines = []
    task_template_line = None
    insert_index = None

    for idx, line in enumerate(lines):
        if "TASK=SQuAD" in line and "EPS=" in line:
            if task_template_line is None:
                task_template_line = line
                insert_index = len(base_lines)
            # 跳过所有旧的 SQuAD+EPS 行
            continue
        else:
            base_lines.append(line)

    if task_template_line is None:
        raise ValueError("在模板中没有找到包含 'TASK=SQuAD' 且包含 'EPS=' 的命令行，请检查模板文件。")

    # 构造新的多行命令，每一行替换不同的 EPS 值
    task_lines = []
    for eps in eps_group:
        eps_str = format_eps_for_cmd(eps)
        # 假设模板行中原本包含 `EPS=1.58e-3`，这里进行替换
        new_task = task_template_line.replace("EPS=1.58e-3", f"EPS={eps_str}")
        task_lines.append(new_task)

    # 将新的任务行插回原来的位置
    if insert_index is None:
        # 理论上不会发生，这里做一个保护
        insert_index = len(base_lines)
    new_lines = base_lines[:insert_index] + task_lines + base_lines[insert_index:]
    final_text = "\n".join(new_lines)

    return final_text


def main():
    """
    主函数：
    1. 生成 EPS 取值列表；
    2. 按每 2 个一组进行分组；
    3. 对每一组调用 build_job_text 生成新的作业文本；
    4. 将结果写入到当前目录下的 `jobs_eps` 子目录中。
    """
    # 当前脚本所在目录
    base_dir = Path(__file__).resolve().parent

    # 模板文件路径，假定与本脚本在同一目录下，并且文件名为 h1e-3.txt
    template_path = base_dir / "1.58e-3AND1e-3.txt"

    # 输出目录：用于存放自动生成的作业 txt 文件
    output_dir = base_dir / "jobs_eps"
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. 生成所有 EPS 取值
    eps_list = generate_eps_values(num_per_interval=10)

    # 为了方便调试，可以打印一下总数量：
    print(f"总共生成的去重后 EPS 数量: {len(eps_list)}")

    # 2. 按每 2 个一组进行切分
    group_size = 2
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

        # 构造该组对应的名字：包含两个 EPS 值，并带上当天日期
        first_eps_name = format_eps_for_name(eps_group[0])
        second_eps_name = format_eps_for_name(eps_group[1])
        date_str = datetime.now().strftime("%m%d")
        job_tag = f"SQuAD_optH{first_eps_name}AND{second_eps_name}_{date_str}"

        # 输出文件名，例如：SQuAD_optH1.58e-3AND1e-3_1120.txt
        job_filename = f"{job_tag}.txt"
        job_path = output_dir / job_filename

        # 将文本写入文件
        job_path.write_text(job_text, encoding="utf-8")

        print(f"生成作业文件: {job_path}")


if __name__ == "__main__":
    main()