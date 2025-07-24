import pandas as pd
import os
from sklearn.metrics import precision_score, recall_score, f1_score

# 输入文件路径
input_file = "results/psychiatry_questions_qwen3-14b_2025-07-22_16-23-09_answers.csv"
df = pd.read_csv(input_file)

# 将 is_correct 列转换为布尔值
df["is_correct"] = df["is_correct"].astype(bool)

# 所有数据的总体指标
y_true_all = [True] * len(df)  # 视每个问题都有标准答案
y_pred_all = df["is_correct"].tolist()

overall_accuracy = sum(y_pred_all) / len(y_pred_all)
overall_precision = precision_score(y_true_all, y_pred_all, zero_division=0)
overall_recall = recall_score(y_true_all, y_pred_all, zero_division=0)
overall_f1 = f1_score(y_true_all, y_pred_all, zero_division=0)

results = []
results.append({
    "topic_category": "overall",
    "num_questions": len(df),
    "num_correct": sum(y_pred_all),
    "accuracy": overall_accuracy,
    "precision": overall_precision,
    "recall": overall_recall,
    "f1": overall_f1
})

# 每个 topic_category 的指标
for topic in df["topic_category"].unique():
    sub_df = df[df["topic_category"] == topic]
    y_true = [True] * len(sub_df)
    y_pred = sub_df["is_correct"].tolist()
    
    acc = sum(y_pred) / len(y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    results.append({
        "topic_category": topic,
        "num_questions": len(sub_df),
        "num_correct": sum(y_pred),
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1
    })

# 保存结果
analysis_df = pd.DataFrame(results)

# 输出路径
basename = os.path.basename(input_file)
output_file = f"results/{basename.replace('.csv', '_analysis.csv')}"
analysis_df.to_csv(output_file, index=False)

print(f"✅ 分析结果已保存到：{output_file}")
