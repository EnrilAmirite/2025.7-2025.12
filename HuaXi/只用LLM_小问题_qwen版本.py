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
import httpx  # 新增

# ========== 加载环境变量 ==========
load_dotenv()

# ========== 各种路径 ==========
path_question = "question/mentat_dataset_test_1.csv"
model_name = "qwen3-14b"
#model_name = "qwen3-32b"

# ========== 随机替换设置 ==========
NATIONALITIES = [
    "American", "Chinese", "Indian", "Brazilian", "German",
    "French", "Japanese", "Nigerian", "Mexican", "Canadian"
]

# ========== Prompt模板 ==========
introduce_task = "This is a question involving medical knowledge. Please choose the only correct answer from the five options A, B, C, D and E."
control_output = "Please only output the option you choose, like 'A','B','C','D','E'.Please do not give me any other words."

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

# ========== 直接用 httpx 异步请求 qwen3-32b ==========
async def llm_model_func(prompt: str) -> str:
    api_key = os.getenv("MODEL_API_KEY")
    base_url = os.getenv("MODEL_API_URL")  # 例：https://openapi.yourprovider.com/v1
    if not base_url:
        raise RuntimeError("请设置环境变量 MODEL_API_URL")

    url = f"{base_url}/chat/completions"

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    json_body = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0,
        "max_tokens": 20,
        "enable_thinking": False  # qwen模型必须加这个参数
    }

    async with httpx.AsyncClient(timeout=60) as client:
        response = await client.post(url, headers=headers, json=json_body)
        if response.status_code != 200:
            raise RuntimeError(f"LLM 调用失败: 状态码 {response.status_code}, 内容: {response.text}")
        data = response.json()
        return data["choices"][0]["message"]["content"].strip()

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

            question_texts = [
                row.get("text_male", ""),
                row.get("text_female", ""),
                row.get("text_nonbinary", "")
            ]
            question = random.choice([q for q in question_texts if q.strip() != ""])
            age = str(random.randint(18, 60))
            nat = random.choice(NATIONALITIES)
            question = question.replace("<AGE>", age).replace("<NAT>", nat)

            options = {
                "A": row.get("answer_a", ""),
                "B": row.get("answer_b", ""),
                "C": row.get("answer_c", ""),
                "D": row.get("answer_d", ""),
                "E": row.get("answer_e", "")
            }
            option_text = "\n".join([f"{k}. {v}" for k, v in options.items() if v.strip()])

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

                #if model_answer not in ['A', 'B', 'C', 'D', 'E']:
                #    model_answer = "invalid answer"

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

    acc = (correct / total) * 100 if total > 0 else 0
    print(f"\n答题完成: 共 {total} 题，正确 {correct} 题，准确率 {acc:.2f}%")

    df = pd.DataFrame(results)
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    os.makedirs("results", exist_ok=True)
    df.to_csv(f"results/mentat_dataset_withoutLightRAG_{model_name}_{timestamp}_answers.csv", index=False)

# ========== 入口 ==========
if __name__ == "__main__":
    asyncio.run(answer_questions(model_name))
    print("已完成答题ヾ(=･ω･=)o")
