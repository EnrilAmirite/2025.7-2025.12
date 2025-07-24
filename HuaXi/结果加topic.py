import pandas as pd
import os

# 设置文件路径
answer_file = "results/gpt-4o-mini-2024-07-18_2025-07-14_14-07-57_answers.csv"
question_file = "question/psychiatry_questions.csv"
output_dir = "results"

# 读取CSV文件
answers_df = pd.read_csv(answer_file)
questions_df = pd.read_csv(question_file)

# 创建 id 到 topic name 的映射，并进行分类
# 定义分类函数（不区分大小写）
def categorize_topic(topic_name):
    if pd.isna(topic_name):
        return "treatments"
    name_lower = topic_name.lower()
    if "disorders" in name_lower:
        return "disorders"
    elif "symptoms" in name_lower:
        return "symptoms"
    else:
        return "treatments"

# 添加新列：topic_category
topic_mapping = questions_df.set_index("id")["topic_name"].apply(categorize_topic)
answers_df["topic_category"] = answers_df["id"].map(topic_mapping)

# 构造输出文件名
output_file = f"psychiatry_questions_{os.path.basename(answer_file)}"

# 构造输出文件名（保存在 results/ 目录下）
output_filename = f"psychiatry_questions_{os.path.basename(answer_file)}"
output_path = os.path.join(output_dir, output_filename)

# 保存结果
answers_df.to_csv(output_path, index=False)
print(f"✅ 已保存新文件：{output_path}")