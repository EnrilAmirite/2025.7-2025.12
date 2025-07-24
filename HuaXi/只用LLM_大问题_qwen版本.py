import os
import csv
import datetime
import numpy as np
from tqdm import tqdm
import pandas as pd
from dotenv import load_dotenv
import asyncio
import httpx

# ====================== 加载环境变量 ======================
load_dotenv()

# ===================== 各种路径 =====================
path_question ="question/psychiatry_questions_test.csv"
#model_name = "qwen3-8b"
model_name = "qwen3-14b"
#model_name = "qwen3-32b"
# ===================== Prompt =====================
introduce_task = "This is a question involving medical knowledge. Please choose the only correct answer from the four options A, B, C, and D."
control_output = "Please only output the option you choose, like 'A','B','C','D'. Please do not give me any other words."

# ========== 提取正确答案 ==========
def extract_correct_answer(cop):
    mapping = {
        "1": "A", 1: "A",
        "2": "B", 2: "B",
        "3": "C", 3: "C",
        "4": "D", 4: "D",
    }
    return mapping.get(cop, None)

#=====================转换topic===========
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

# ========== 调用 Qwen3-32b 接口 ==========
async def llm_model_func(prompt: str) -> str:
    api_key = os.getenv("MODEL_API_KEY")
    base_url = os.getenv("MODEL_API_URL", "https://openai.fakeapi.com/v1")  # 你实际的接口地址
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
        "enable_thinking": False  # 关键参数，禁止“思考模式”
    }

    async with httpx.AsyncClient(timeout=60) as client:
        response = await client.post(url, headers=headers, json=json_body)
        if response.status_code != 200:
            raise RuntimeError(f"LLM 调用失败: 状态码 {response.status_code}, 内容: {response.text}")
        data = response.json()
        # 返回内容路径，参考OpenAI标准接口格式
        return data["choices"][0]["message"]["content"].strip()

# ======================== 答题函数 =======================
async def answer_questions():
    print(f"\n本次使用模型: {model_name}")

    total, correct = 0, 0
    results = []

    with open(path_question, "r", encoding="utf-8-sig", newline='') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        for row in tqdm(rows, desc="答题中", unit="题"):
            qid = row["id"]
            question = row["question"]
            cop = row["cop"]
            topic_name = row["topic_name"]
            topic_category = categorize_topic(topic_name)

            correct_answer = extract_correct_answer(cop)

            if not correct_answer:
                print(f"无法识别正确答案 (id={qid})")
                continue

            options = {
                "A": row.get("opa", ""),
                "B": row.get("opb", ""),
                "C": row.get("opc", ""),
                "D": row.get("opd", ""),
            }

            option_text = "\n".join([f"{k}. {v}" for k, v in options.items() if v.strip()])
            full_prompt = f"""{introduce_task}

Question: {question}

Options:
{option_text}

{control_output}
"""

            try:
                model_answer = await llm_model_func(full_prompt)
                model_answer = model_answer.strip().upper()

                #if model_answer not in ["A", "B", "C", "D"]:
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
    df.to_csv(f"results/psychiatry_questions_withoutLightRAG_{model_name}_{timestamp}_answers.csv", index=False)

# ========== 入口 ==========
if __name__ == "__main__":
    asyncio.run(answer_questions())
    print("已完成答题ヾ(=･ω･=)o")
