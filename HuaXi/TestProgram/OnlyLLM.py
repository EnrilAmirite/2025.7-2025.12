import csv
import re
from openai import OpenAI
import os, re, pandas as pd, openai, asyncio, datetime
from tqdm.asyncio import tqdm as tqdm_asyncio

# =============== 设置模型 ==================
#用来回答问题的模型
MODEL_NAME = "deepseek-chat"
API_KEY = "sk-b0e0448fb7784847acb70e59f3cadacb"
API_URL = "https://api.deepseek.com"

client = OpenAI(api_key=API_KEY, base_url=API_URL)

# ========== 设置路径 ==========
path_question = "question/psychiatry_questions_test.csv"
path_answer = "answers.csv"

# ========== Prompt 模板 ==========
introduce_task = "This is a question involving medical knowledge. Please choose the only correct answer from the four options A, B, C, and D."
introduce_rag = "Here are some relevant information on the issue:\n(N/A for now, please answer based on general knowledge)"
control_output = "Please only output the option you choose, like 'A','B','C','D'. Do not give me any other words."

# ========== 提取正确答案 ==========
def extract_correct_answer(cop):
    mapping = {
        "1": "A", 1: "A",
        "2": "B", 2: "B",
        "3": "C", 3: "C",
        "4": "D", 4: "D",
    }
    return mapping.get(cop, None)

# ========== 主函数 ==========
def main():
    total, correct = 0, 0
    results = []

    with open(path_question, "r", encoding="utf-8-sig", newline='') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        for row in tqdm_asyncio(rows, desc="处理问题", unit="题"):
            qid = row["id"]
            question = row["question"]
            cop = row["cop"]
            correct_answer = extract_correct_answer(cop)

            if not correct_answer:
                print(f"跳过问题 {qid}，无法识别正确答案 cop={cop}")
                continue

            # 构造选项文本
            options = {
                "A": row.get("opa", ""),
                "B": row.get("opb", ""),
                "C": row.get("opc", ""),
                "D": row.get("opd", ""),
            }
            option_text = "\n".join([f"{k}. {v}" for k, v in options.items() if v.strip()])

            # 构造 prompt
            full_prompt = f"""{introduce_task}

Question:
{question}

Options:
{option_text}

{control_output}
"""

            # print(f"\n处理问题 {qid}...\nPrompt:\n{full_prompt}")

            try:
                response = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[
                        {"role": "system", "content": "You are a helpful medical assistant."},
                        {"role": "user", "content": full_prompt}
                    ]
                )
                model_answer = response.choices[0].message.content.strip().upper()
                #match = re.search(r"\b([A-D])\b", model_answer)
                #model_answer = match.group(1) if match else "?"

                is_correct = model_answer == correct_answer
                total += 1
                correct += int(is_correct)

                results.append({
                    "id": qid,
                    "question": question,
                    "correct_answer": correct_answer,
                    "model_answer": model_answer,
                    "is_correct": "true" if is_correct else "false"
                })

            except Exception as e:
                print(f"问题 {qid} 出错: {e}")
                results.append({
                    "id": qid,
                    "question": question,
                    "correct_answer": correct_answer,
                    "model_answer": "ERROR",
                    "is_correct": "false"
                })


    out = pd.DataFrame(results)
    ts = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out.to_csv(f"OnlyLLM_answer_{MODEL_NAME}_{ts}.csv", index=False)


    acc = (correct / total) * 100 if total > 0 else 0
    print(f"\n答题完成！总题数: {total}，正确: {correct}，正确率: {acc:.2f}%")


if __name__ == "__main__":
    main()
