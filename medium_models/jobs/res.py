import ast
import pandas as pd
import matplotlib.pyplot as plt


def parse_log_file(filepath):
    """
    从日志文件中解析出训练和评估记录。
    训练记录包含 'global_step' 和 'loss'，
    评估记录包含 'eval_loss' 和 'eval_acc'，
    评估记录会附加最近一次训练日志的 global_step。
    """
    training_records = []
    eval_records = []
    last_global_step = None

    with open(filepath, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            # 判断是否包含花括号
            if "{" in line and "}" in line:
                try:
                    # 提取花括号内的内容并转换为字典
                    dict_str = line[line.index("{"): line.rindex("}") + 1]
                    record = ast.literal_eval(dict_str)
                except Exception as e:
                    print(f"第 {i} 行转换失败: {line}")
                    continue

                # 训练记录：含有 global_step 和 loss
                if "global_step" in record and "loss" in record:
                    training_records.append(record)
                    last_global_step = record["global_step"]
                # 评估记录：含有 eval_loss 和 eval_acc
                elif "eval_loss" in record and "eval_acc" in record:
                    if last_global_step is None:
                        print(f"第 {i} 行评估日志，但尚未记录训练 global_step: {line}")
                        continue
                    record["global_step"] = last_global_step
                    eval_records.append(record)
                else:
                    print(f"第 {i} 行未识别的记录: {line}")
            else:
                print(f"第 {i} 行未包含字典内容: {line}")

    print(f"共解析出 {len(training_records)} 条训练记录和 {len(eval_records)} 条评估记录。")
    return training_records, eval_records


def create_dataframes(training_records, eval_records):
    df_train = pd.DataFrame(training_records)
    df_eval = pd.DataFrame(eval_records)
    return df_train, df_eval


def plot_results(df_train, df_eval, window_size=100, alpha=0.01):
    """
    绘制三个图：
    1. 训练 loss（包含原始数据、移动平均和 EMA 平滑后数据）随 global_step 的变化
    2. 评估 accuracy 随 global_step 的变化
    3. 评估 loss 随 global_step 的变化
    """
    # 对训练 loss 进行平滑处理
    df_train['loss_rolling'] = df_train['loss'].rolling(window=window_size).mean()
    df_train['loss_ema'] = df_train['loss'].ewm(alpha=alpha).mean()

    # 图1：训练 loss 随 global_step 的变化（显示原始及平滑后的曲线）
    plt.figure(figsize=(10, 6))
    plt.plot(df_train['global_step'], df_train['loss'], label='原始 Loss', alpha=0.3)
    plt.plot(df_train['global_step'], df_train['loss_rolling'], label=f'移动平均 (window={window_size})', linewidth=2)
    plt.plot(df_train['global_step'], df_train['loss_ema'], label=f'EMA (alpha={alpha})', linewidth=2)
    plt.xlabel('Global Step')
    plt.ylabel('Loss')
    plt.title('训练 Loss vs Global Step (平滑处理)')
    plt.legend()
    plt.grid(True)
    plt.savefig("loss_vs_global_step_smoothed.png")

    # 图2：评估 accuracy 随 global_step 的变化
    if not df_eval.empty:
        plt.figure(figsize=(10, 6))
        plt.plot(df_eval['global_step'], df_eval['eval_acc'], marker='o', color='green')
        plt.xlabel('Global Step')
        plt.ylabel('Eval Accuracy')
        plt.title('评估 Accuracy vs Global Step')
        plt.grid(True)
        plt.savefig("eval_acc_vs_global_step.png")

        # 图3：评估 loss 随 global_step 的变化
        plt.figure(figsize=(10, 6))
        plt.plot(df_eval['global_step'], df_eval['eval_loss'], marker='o', color='red')
        plt.xlabel('Global Step')
        plt.ylabel('Eval Loss')
        plt.title('评估 Loss vs Global Step')
        plt.grid(True)
        plt.savefig("eval_loss_vs_global_step.png")
    else:
        print("评估数据为空，跳过评估图表绘制。")

    plt.show()


if __name__ == "__main__":
    # 请确保文件路径正确
    filepath = "sst-2full_36129372_1.out"
    training_records, eval_records = parse_log_file(filepath)
    df_train, df_eval = create_dataframes(training_records, eval_records)

    print("训练数据预览：")
    print(df_train.head())
    print("\n评估数据预览：")
    print(df_eval.head())

    # window_size 和 alpha 参数可根据需要调整
    plot_results(df_train, df_eval, window_size=100, alpha=0.01)