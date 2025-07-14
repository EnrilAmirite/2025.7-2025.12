import os
import csv
import re
import asyncio
from glob import glob
from lightrag import LightRAG, QueryParam
from lightrag.llm.openai import openai_embed
from lightrag.kg.shared_storage import initialize_pipeline_status
from lightrag.utils import EmbeddingFunc, logger, set_verbose_debug
import fitz  # PyMuPDF
import openai
from openai import OpenAI
import numpy as np
import os, re, pandas as pd, openai, asyncio, datetime
from tqdm.asyncio import tqdm as tqdm_asyncio

# =============== 设置模型 ==================
#用来回答问题的模型
MODEL_NAME = "deepseek-chat"
API_KEY = "sk-b0e0448fb7784847acb70e59f3cadacb"
API_URL = "https://api.deepseek.com"
#os.environ["OPENAI_API_KEY"] = "sk-你的openai key"
#嵌入模型
EBD_MODEL_NAME="text-embedding-3-small"
EBD_API_URL= "https://ai.nengyongai.cn/v1"
EBD_API_KEY = "sk-qz2oVXp2zhSaGYd6H0bkvDdIgHCRVTeZjQs4U07Fj2MiyFDO"

# ========== 设置 prompt ==========
introduce_task = "This is a question involving medical knowledge. Please choose the only correct answer from the four options A, B, C, and D."
introduce_rag = "Here are some relevant information on the issue:"
control_output = "Please only output the option you choose, like 'A','B','C','D'. Do not give me any other words."

# ========== 设置路径 ==========
path_books = "books"
path_question = "question/psychiatry_questions_test.csv"
path_answer = "answers.csv"
WORKING_DIR = "./lightrag_workspace"

# ==========回答问题的LLM 的接口==========
#这里用的deepseek
client = OpenAI(api_key=API_KEY, base_url=API_URL)
def deepseek_chat(messages):
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=messages
    )
    return response.choices[0].message.content

#==================封装到rag======================
async def LLM_Chat(prompt, system_prompt=None, history_messages=[], keyword_extraction=False, **kwargs):
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt},
    ]
    return deepseek_chat(messages)

# ========== 从cop中提取正确答案 ==========
def extract_correct_answer(cop):
    mapping = {
        "1": "A", 1: "A",
        "2": "B", 2: "B",
        "3": "C", 3: "C",
        "4": "D", 4: "D",
    }
    return mapping.get(cop, None)



# ==================embedding模型================
#用于构建文档索引和查询任务
async def embedding_func(texts: list[str]) -> np.ndarray:
    return await openai_embed(
        texts,
        model=EBD_MODEL_NAME,
        api_key=EBD_API_KEY,
        base_url=EBD_API_URL
    )

# ============== RAG初始化 ===============
async def initialize_rag():
    rag = LightRAG(
        working_dir=WORKING_DIR,
        embedding_func=EmbeddingFunc(
            embedding_dim=4096,
            max_token_size=8192,
            func=embedding_func
        ),  
        llm_model_func=LLM_Chat
    )
    await rag.initialize_storages()
    await initialize_pipeline_status()
    return rag

#======================================主函数===============================
async def main():
    if not API_KEY:
        print("请设置 API_KEY")
        return

    # 清空旧文件
    for f in glob(os.path.join(WORKING_DIR, "*")):
        os.remove(f)

    # 初始化 RAG 系统
    rag = await initialize_rag()

    # ==== 加载 TXT 文档 ====
    print("正在加载 TXT 文本...")
    txt_files = glob(os.path.join(path_books, "*.txt"))
    full_text = ""
    for file_path in txt_files:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read().strip()
            full_text += content + "\n\n"

    # 插入到 RAG 系统中
    await rag.ainsert(full_text)
    print("已将所有文本插入LightRAG")
    #做题...做题...
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

            print(f"{full_prompt}")

            try:
                #导入prompt,设置查询方式(QueryParam),这里用的hybrid
                print(f"正在回答问题{total}")
                response = await rag.aquery(full_prompt, param=QueryParam(mode="hybrid"))
                model_answer = response.strip().upper()

                match = re.search(r"\b([A-D])\b", model_answer)
                model_answer = match.group(1) if match else "?"

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

    # ==== 写入结果 ====
    out = pd.DataFrame(results)
    ts = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out.to_csv(f"LLMWithRAG_answers_{MODEL_NAME}_{ts}.csv", index=False)

    acc = (correct / total) * 100 if total > 0 else 0
    print(f"\n答题完成！总题数: {total}，正确: {correct}，正确率: {acc:.2f}%")

    await rag.finalize_storages()

if __name__ == "__main__":
    asyncio.run(main())