import pandas as pd
import os
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, classification_report

# 直接指定结果CSV路径
result_csv_path = "results/gpt-4o-mini-2024-07-18_2025-07-14_14-07-57_answers.csv"

def analyze_results(result_csv_path):
    # 检查路径有效性
    if not os.path.exists(result_csv_path):
        print(f"❌ 文件不存在: {result_csv_path}")
        return

    # 读取CSV
    df = pd.read_csv(result_csv_path)

    # 去除空答案或无效答案
    df = df[df["model_answer"].isin(["A", "B", "C", "D"])]

    y_true = df["correct_answer"]
    y_pred = df["model_answer"]

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
    recall = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

    # 输出控制
    print("\n📊 评估结果:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision (macro): {precision:.4f}")
    print(f"Recall (macro): {recall:.4f}")
    print(f"F1-score (macro): {f1:.4f}")
    print("\n📄 详细分类报告:")
    print(classification_report(y_true, y_pred, zero_division=0))

    # 保存为新 CSV
    data_output_path = result_csv_path.replace(".csv", "_analysis.csv")
    summary_df = pd.DataFrame({
        "Metric": ["Accuracy", "Precision (macro)", "Recall (macro)", "F1-score (macro)"],
        "Score": [accuracy, precision, recall, f1]
    })

    summary_df.to_csv(data_output_path, index=False)
    print(f"\n✅ 评估结果已保存至: {data_output_path}")

# 直接执行
if __name__ == "__main__":
    analyze_results(result_csv_path)
