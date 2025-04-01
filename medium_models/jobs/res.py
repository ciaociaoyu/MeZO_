import ast
import pandas as pd
import matplotlib.pyplot as plt


def parse_log_file(filepath):
    """
    从日志文件中解析出训练和评估记录。
    训练记录包含 'global_step' 和 'loss'，
    评估记录包含 'eval_loss' 和 'eval_acc'，
    评估记录会附加最近一次训练记录的 global_step。
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
                    # 提取花括号内的内容
                    dict_str = line[line.index("{"): line.rindex("}") + 1]
                    record = ast.literal_eval(dict_str)
                except Exception as e:
                    print(f"第 {i} 行转换失败: {line}")
                    continue

                # 如果是训练日志（含有 global_step 和 loss）
                if "global_step" in record and "loss" in record:
                    training_records.append(record)
                    last_global_step = record["global_step"]
                # 如果是评估日志（含有 eval_loss 和 eval_acc）
                elif "eval_loss" in record and "eval_acc" in record:
                    # 如果尚未获取训练 global_step，则尝试跳过或赋默认值
                    if last_global_step is None:
                        print(f"第 {i} 行评估日志，但尚未记录训练 global_step: {line}")
                        continue
                    # 为评估记录添加 global_step 信息
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


def plot_results(df_train, df_eval):
    # 检查数据是否为空
    if df_train.empty:
        print("训练数据为空，无法绘图。")
    else:
        # 图1：训练 loss 随 global_step 的变化
        plt.figure()
        plt.plot(df_train['global_step'], df_train['loss'], marker='o')
        plt.xlabel('Global Step')
        plt.ylabel('Loss')
        plt.title('Training Loss vs Global Step')
        plt.grid(True)
        plt.savefig("loss_vs_global_step.png")

    if df_eval.empty:
        print("评估数据为空，无法绘图评估指标。")
    else:
        # 图2：评估 accuracy 随 global_step 的变化
        plt.figure()
        plt.plot(df_eval['global_step'], df_eval['eval_acc'], marker='o', color='green')
        plt.xlabel('Global Step')
        plt.ylabel('Eval Accuracy')
        plt.title('Eval Accuracy vs Global Step')
        plt.grid(True)
        plt.savefig("eval_acc_vs_global_step.png")

        # 图3：评估 loss 随 global_step 的变化
        plt.figure()
        plt.plot(df_eval['global_step'], df_eval['eval_loss'], marker='o', color='red')
        plt.xlabel('Global Step')
        plt.ylabel('Eval Loss')
        plt.title('Eval Loss vs Global Step')
        plt.grid(True)
        plt.savefig("eval_loss_vs_global_step.png")

    # 展示所有图表
    plt.show()


if __name__ == "__main__":
    filepath = "sst-2full_test.out"  # 请确保文件路径正确
    training_records, eval_records = parse_log_file(filepath)
    df_train, df_eval = create_dataframes(training_records, eval_records)

    # 打印部分数据，方便确认读取是否正确
    print("训练数据预览：")
    print(df_train.head())
    print("\n评估数据预览：")
    print(df_eval.head())

    plot_results(df_train, df_eval)