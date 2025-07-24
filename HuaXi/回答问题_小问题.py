import os
import csv
import datetime
import numpy as np
from tqdm import tqdm
import pandas as pd
from glob import glob
from lightrag import LightRAG, QueryParam
from lightrag.utils import EmbeddingFunc
import asyncio
from openai import OpenAI

import asyncio#这是处理同步异步的
import inspect
from lightrag import LightRAG, QueryParam
from lightrag.llm.openai import openai_complete_if_cache
#from lightrag.llm.ollama import ollama_embed
from lightrag.llm.openai import openai_embed
from lightrag.utils import EmbeddingFunc, logger, set_verbose_debug
from lightrag.kg.shared_storage import initialize_pipeline_status
import numpy as np
from glob import glob
from dotenv import load_dotenv
import ast
import random

#加载环境变量
load_dotenv() 
#每次答题完成之后记得清理kv_store_llm_response_cache.json这个文件...

#==========================题目中的随机替换=====================
NATIONALITIES = [
    "American", "Chinese", "Indian", "Brazilian", "German",
    "French", "Japanese", "Nigerian", "Mexican", "Canadian"
]
#============================================================


#====================各种路径====================
WORKING_DIR=os.getenv("WORKING_DIR")
path_question = os.getenv("PATH_QUESTION")
path_books = os.getenv("PATH_BOOKS")
model_name=os.getenv("MODEL_NAME")
#===============================================

#===========================prompt======================================
introduce_task = "This is a question involving medical knowledge. Please choose the only correct answer from the five options A, B, C, D and E."
introduce_context="Please give your answer based on the knowledge you found in the knowledge base."
control_output = "Please only output the option you choose, like 'A','B','C','D','E'. Do not give me any other words."
#======================================================================

# ========== 从creator_truth中提取正确答案 ==========
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
#===============================================

#===========================设置需要用的LLM======================
async def llm_model_func(
    prompt, system_prompt=None, history_messages=[], keyword_extraction=False, **kwargs
) -> str:
    return await openai_complete_if_cache(
        #如果没有设置就默认deepseek-chat
        #修改请在.env里修改
        os.getenv("MODEL_NAME", "deepseek-chat"),
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        api_key=os.getenv("MODEL_API_KEY"),
        base_url=os.getenv("MODEL_API_URL"),
        **kwargs,
    )
#=============================================================


#==============================初始化rag===========================
#主要是设置embedding模型,参见官方的文档
async def embedding_llm(texts: list[str]) -> np.ndarray:
    return await openai_embed(
        texts,
        model=os.getenv("EBD_MODEL_NAME"),
        api_key=os.getenv("EBD_MODEL_API_KEY"),
        base_url=os.getenv("EBD_MODEL_API_URL")
    )

async def initialize_rag():
    rag = LightRAG(
        working_dir=WORKING_DIR,
        llm_model_func=llm_model_func,
        embedding_func=EmbeddingFunc(
            embedding_dim=int(os.getenv("EMBEDDING_DIM", "1536")),
            max_token_size=int(os.getenv("MAX_EMBED_TOKENS", "8192")),
            func=embedding_llm
        ),
    )

    await rag.initialize_storages()
    await initialize_pipeline_status()

    return rag
#===============================================================

#==============================逐步打印流式输出====================
async def print_stream(stream):
    async for chunk in stream:
        if chunk:
            print(chunk, end="", flush=True)
#================================================================



#=======================答题函数===================
async def answer_questions(model_name):
    print(f"\n本次使用模型:{model_name}")

    #调用rag初始化
    rag = await initialize_rag()

    total, correct = 0, 0
    results = []

    #打开题目文件
    path_question=os.getenv("PATH_QUESTION")
    with open(path_question, "r", encoding="utf-8-sig", newline='') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        for row in tqdm(rows, desc="答题中", unit="题"):
            qid = row["q_id"]
            creator_truth = row["creator_truth"]
            topic_category=row["category"]
            correct_answer = extract_correct_answer(creator_truth)
            if not correct_answer:
                print(f"无法识别正确答案 (id={qid})")
                continue
            #这里是随机选择一个性别作为问题
            question_texts = [
                row.get("text_male", ""),
                row.get("text_female", ""),
                row.get("text_nonbinary", "")
            ]
            question = random.choice([q for q in question_texts if q.strip() != ""])
            #随机替换题目中的占位符
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
{introduce_context}
{control_output}
"""
            try:
                model_answer="ERROR"
                #设置不用来查询的prompt
                query_param = QueryParam(
                    mode="global",
                    user_prompt=f"{full_prompt}",
                )
                #导入prompt,设置查询方式(QueryParam),这里用的global
                response = await rag.aquery(f"{question}",param=query_param)
                model_answer = response.strip().upper()

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
                    "topic_category":topic_category
                })
                
            except Exception as e:
                print(f"问题 {qid} 出错: {e}")
                results.append({
                    "id": qid,
                    "question": question,
                    "correct_answer": correct_answer,
                    "model_answer": model_answer,
                    "is_correct": "false",
                    "topic_category":topic_category
                })

    acc = (correct / total) * 100 if total > 0 else 0
    print(f"\n答题完成:共 {total} 题，正确 {correct} 题，准确率 {acc:.2f}%")

    df = pd.DataFrame(results)
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    df.to_csv(f"results/mentat_dataset_{model_name}_{timestamp}_answers.csv", index=False)

    await rag.finalize_storages()


# ========== 入口 ==========
if __name__ == "__main__":
    path_question = os.getenv("PATH_QUESTION")
    asyncio.run(answer_questions(model_name))
    print("已完成答题ヾ(=･ω･=)o") 

