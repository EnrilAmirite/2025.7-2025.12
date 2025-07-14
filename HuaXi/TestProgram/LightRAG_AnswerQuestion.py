import os
import csv
import re
import argparse
import datetime
import numpy as np
from tqdm import tqdm
import pandas as pd
from glob import glob
from lightrag import LightRAG, QueryParam
from lightrag.utils import EmbeddingFunc
import asyncio
from openai import OpenAI

#====================各种路径====================
WORKING_DIR = "./lightrag_workspace"
path_question = "question/psychiatry_questions_test.csv"
path_books = "books"

#================================在这里更换模型======================
MODEL_MAP = {
    "deepseek-chat": {
        "api_key": "sk-你的deepseek-key",
        "base_url": "https://api.deepseek.com",
        "model_name": "deepseek-chat"
    },
    "openai": {
        "api_key": "sk-你的openai-key",
        "base_url": "https://api.openai.com/v1",
        "model_name": "gpt-4o"
    },
    "moonshot": {
        "api_key": "sk-你的moonshot-key",
        "base_url": "https://api.moonshot.cn/v1",
        "model_name": "moonshot-v1-8k"
    }
}

#===========================prompt======================================
introduce_task = "This is a question involving medical knowledge. Please choose the only correct answer from the four options A, B, C, and D."
control_output = "Please only output the option you choose, like 'A','B','C','D'. Do not give me any other words."


# ========== 从cop中提取正确答案 ==========
def extract_correct_answer(cop):
    mapping = {
        "1": "A", 1: "A",
        "2": "B", 2: "B",
        "3": "C", 3: "C",
        "4": "D", 4: "D",
    }
    return mapping.get(cop, None)

#==================调用模型+使用RAG=====================
def build_llm_func(model_name):
    conf = MODEL_MAP[model_name]
    client = OpenAI(api_key=conf["api_key"], base_url=conf["base_url"])

    async def model_func(prompt, **kwargs):
        response = client.chat.completions.create(
            model=conf["model_name"],
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content

    return model_func

# ========== 构造 dummy embedding ==========
async def dummy_embedding(texts: list[str]) -> np.ndarray:
    return np.zeros((len(texts), 4096), dtype=np.float32)

# ========== 主答题函数 ==========
async def answer_questions(model_name):
    print(f"\n使用模型：{model_name}")
    model_func = build_llm_func(model_name)

    rag = LightRAG(
        working_dir=WORKING_DIR,
        embedding_func=EmbeddingFunc(
            embedding_dim=4096,
            max_token_size=8192,
            func=dummy_embedding
        ),
        llm_model_func=model_func
    )
    await rag.initialize_storages()

    total, correct = 0, 0
    results = []

    with open(path_question, "r", encoding="utf-8-sig", newline='') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        for row in tqdm(rows, desc="答题中", unit="题"):
            qid = row["id"]
            question = row["question"]
            cop = row["cop"]
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

Question:
{question}

Options:
{option_text}

{control_output}"""

            try:
                #导入prompt,设置查询方式(QueryParam),这里用的hybrid
                response = await rag.aquery(full_prompt, param=QueryParam(mode="hybrid"))
                model_answer = response.strip().upper()
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
                    "model_answer": model_answer,
                    "is_correct": "false"
                })

    acc = (correct / total) * 100 if total > 0 else 0
    print(f"\n答题完成：共 {total} 题，正确 {correct} 题，准确率 {acc:.2f}%")

    df = pd.DataFrame(results)
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    df.to_csv(f"{model_name}_{timestamp}_answers.csv", index=False)

    await rag.finalize_storages()


# ========== 入口 ==========
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODEL_MAP.keys(), required=True, help="选择使用的模型")
    args = parser.parse_args()

    asyncio.run(answer_questions(args.model))
