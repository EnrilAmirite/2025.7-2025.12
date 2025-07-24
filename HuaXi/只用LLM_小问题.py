import os
import csv
import datetime
import numpy as np
from tqdm import tqdm
import pandas as pd
import asyncio
import ast
import random
from dotenv import load_dotenv

# ==== LLM调用 ====
from lightrag.llm.openai import openai_complete_if_cache

# ========== 加载环境变量 ==========
load_dotenv()

# ========== 各种路径 ==========
path_question = "question/mentat_dataset.csv"
#model_name ="qwen3-8b"
#model_name ="gemini-2.5-pro"
#model_name ="deepseek-v3-0324"
#model_name ="gpt-4o-mini-2024-07-18"
#model_name ="o3"
#model_name ="deepseek-r1"
#model_name ="claude-3-7-sonnet-20250219"
#model_name ="gemini-2.0-pro-exp-02-05"
#model_name ="gpt-4-turbo-2024-04-09"
model_name ="llama-3.3-70b-instruct"
#model_name ="qwen3-32b"
# ========== 随机替换设置 ==========
NATIONALITIES = [
    "American", "Chinese", "Indian", "Brazilian", "German",
    "French", "Japanese", "Nigerian", "Mexican", "Canadian"
]

# ========== Prompt模版 ==========
introduce_task = "This is a question involving medical knowledge. Please choose the only correct answer from the five options A, B, C, D and E."
control_output = "Please only output the option you choose, like 'A','B','C','D','E'. Do not give me any other words."

# ========== 提取正确答案 ==========
def extract_correct_answer(creator_truth):
    index_to_option = {0: "A", 1: "B", 2: "C", 3: "D", 4: "E"}
    try:
        truth_list = ast.literal_eval(creator_truth)
        if isinstance(truth_list, list) and len(truth_list) == 5:
            max_index = truth_list.index(max(truth_list))
            return index_to_option.get(max_index, "Unknown")
        else:
            return "Invalid"
    except:
        return "ParseError"

# ========== LLM模型封装 ==========
async def llm_model_func(
    prompt, system_prompt=None, history_messages=[], keyword_extraction=False, **kwargs
) -> str:
    return await openai_complete_if_cache(
        model_name,
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        api_key=os.getenv("MODEL_API_KEY"),
        base_url=os.getenv("MODEL_API_URL"),
        **kwargs,
    )

# ========== 答题函数 ==========
async def answer_questions(model_name):
    print(f"\n本次使用模型: {model_name}")

    total, correct = 0, 0
    results = []

    with open(path_question, "r", encoding="utf-8-sig", newline='') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        for row in tqdm(rows, desc="答题中", unit="题"):
            qid = row["q_id"]
            creator_truth = row["creator_truth"]
            topic_category = row["category"]

            correct_answer = extract_correct_answer(creator_truth)
            if not correct_answer:
                print(f"无法识别正确答案 (id={qid})")
                continue

            # 随机选择问题形式并替换占位符
            question_texts = [
                row.get("text_male", ""),
                row.get("text_female", ""),
                row.get("text_nonbinary", "")
            ]
            question = random.choice([q for q in question_texts if q.strip() != ""])
            age = str(random.randint(18, 60))
            nat = random.choice(NATIONALITIES)
            question = question.replace("<AGE>", age).replace("<NAT>", nat)

            # 选项拼接
            options = {
                "A": row.get("answer_a", ""),
                "B": row.get("answer_b", ""),
                "C": row.get("answer_c", ""),
                "D": row.get("answer_d", ""),
                "E": row.get("answer_e", "")
            }
            option_text = "\n".join([f"{k}. {v}" for k, v in options.items() if v.strip()])

            # 完整 prompt
            full_prompt = f"""
{introduce_task}

Question:
{question}

Options:
{option_text}

{control_output}
"""

            try:
                model_answer = await llm_model_func(full_prompt)
                model_answer = model_answer.strip().upper()

                if model_answer not in ['A', 'B', 'C', 'D', 'E']:
                    model_answer = "invalid answer"

                is_correct = model_answer == correct_answer
                total += 1
                correct += int(is_correct)

                results.append({
                    "id": qid,
                    "question": question,
                    "correct_answer": correct_answer,
                    "model_answer": model_answer,
                    "is_correct": "true" if is_correct else "false",
                    "topic_category": topic_category
                })

            except Exception as e:
                print(f"问题 {qid} 出错: {e}")
                results.append({
                    "id": qid,
                    "question": question,
                    "correct_answer": correct_answer,
                    "model_answer": "ERROR",
                    "is_correct": "false",
                    "topic_category": topic_category
                })

    # 准确率
    acc = (correct / total) * 100 if total > 0 else 0
    print(f"\n答题完成: 共 {total} 题，正确 {correct} 题，准确率 {acc:.2f}%")

    # 保存结果
    df = pd.DataFrame(results)
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    os.makedirs("results", exist_ok=True)
    df.to_csv(f"results/mentat_dataset_withoutLightRAG_{model_name}_{timestamp}_answers.csv", index=False)

# ========== 入口 ==========
if __name__ == "__main__":
    asyncio.run(answer_questions(model_name))
    print("已完成答题ヾ(=･ω･=)o")
