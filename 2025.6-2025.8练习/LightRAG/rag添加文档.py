import os
import asyncio#这是处理同步异步的
import inspect
import logging
import logging.config
from lightrag import LightRAG, QueryParam
from lightrag.llm.openai import openai_complete_if_cache
#from lightrag.llm.ollama import ollama_embed
from lightrag.llm.openai import openai_embed
from lightrag.utils import EmbeddingFunc, logger, set_verbose_debug
from lightrag.kg.shared_storage import initialize_pipeline_status
import numpy as np
from glob import glob
import nest_asyncio

from dotenv import load_dotenv
load_dotenv(dotenv_path=".env", override=False)
WORKING_DIR=os.getenv("WORKING_DIR")

#=========================设置日志==========================
def configure_logging():
    """Configure logging for the application"""

    #重置处理程序
    for logger_name in ["uvicorn", "uvicorn.access", "uvicorn.error", "lightrag"]:
        logger_instance = logging.getLogger(logger_name)
        logger_instance.handlers = []
        logger_instance.filters = []

    #日志目录
    log_dir = os.getenv("LOG_DIR", os.getcwd())
    log_file_path = os.path.abspath(
        os.path.join(log_dir, "lightrag_compatible_demo.log")
    )

    print(f"\nLightRAG 的日志已存储在{log_file_path}\n")
    os.makedirs(os.path.dirname(log_dir), exist_ok=True)

    # 从环境变量中 获取日志文件最大大小\备份计数
    log_max_bytes = int(os.getenv("LOG_MAX_BYTES", 10485760))  # Default 10MB
    log_backup_count = int(os.getenv("LOG_BACKUP_COUNT", 5))  # Default 5 backups

    logging.config.dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "default": {
                    "format": "%(levelname)s: %(message)s",
                },
                "detailed": {
                    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                },
            },
            "handlers": {
                "console": {
                    "formatter": "default",
                    "class": "logging.StreamHandler",
                    "stream": "ext://sys.stderr",
                },
                "file": {
                    "formatter": "detailed",
                    "class": "logging.handlers.RotatingFileHandler",
                    "filename": log_file_path,
                    "maxBytes": log_max_bytes,
                    "backupCount": log_backup_count,
                    "encoding": "utf-8",
                },
            },
            "loggers": {
                "lightrag": {
                    "handlers": ["console", "file"],
                    "level": "INFO",
                    "propagate": False,
                },
            },
        }
    )

    # Set the logger level to INFO
    logger.setLevel(logging.INFO)
    # Enable verbose debug if needed
    set_verbose_debug(os.getenv("VERBOSE_DEBUG", "false").lower() == "true")
#==============================================================================

#===========================设置需要用的LLM======================
async def llm_model_func(
    prompt, system_prompt=None, history_messages=[], keyword_extraction=False, **kwargs
) -> str:
    return await openai_complete_if_cache(
        #如果没有设置就默认deepseek-chat
        os.getenv("MODEL_NAME", "deepseek-chat"),
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        api_key=os.getenv("MODEL_API_KEY"),
        base_url=os.getenv("MODEL_API_URL", "https://api.deepseek.com"),
        **kwargs,
    )
#=============================================================


#==============================初始化rag===========================
#主要是设置embedding模型,参见官方的文档
async def embedding_llm(texts: list[str]) -> np.ndarray:
    return await openai_embed(
        texts,
        model=os.getenv("EBD_MODEL_NAME", "text-embedding-3-small"),
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


#=========================插入需要的文档=========================
#注意并发的问题
#其实最好用pipline..懒得写了
nest_asyncio.apply()
async def insert_books_txt(rag):
    files = glob("books/*.txt")
    texts = []

    for file in files:
        with open(file, "r", encoding="utf-8") as f:
            content = f.read().strip()
            if content:
                texts.append(content)

    if texts:
        await rag.insert(texts)
        print(f"已插入 {len(texts)} 条文本")
    else:
        print("没有有效文本可插入")

    pass
#===========================================================


#==============================逐步打印流式输出====================
async def print_stream(stream):
    async for chunk in stream:
        if chunk:
            print(chunk, end="", flush=True)
#================================================================


#===========================运行主函数=======================
async def main():
    try:
        #新插入就不需要删除旧文件了
        #调用rag初始化
        rag = await initialize_rag()

        #测试embedding模型是否可用
        test_text = ["This is a test string for embedding."]
        embedding = await rag.embedding_func(test_text)
        embedding_dim = embedding.shape[1]
        print("\n=======================")
        print("测试embedding.........")
        print("========================")
        print(f"Test dict: {test_text}")
        print(f"Detected embedding dimension: {embedding_dim}\n\n")

        #调用文档插入的函数
        try:
            loop = asyncio.get_event_loop()
            loop.run_until_complete(insert_books_txt(rag))
        except Exception as e:
            print(f"发生了错误!: {e}")

        #test问题
        print("\n=====================")
        print("Query mode: hybird")
        print("=====================")
        print("简单介绍一下新插入的文档")
        resp = await rag.aquery(
            "简单介绍一下新插入的文档",
            param=QueryParam(mode="hybird", stream=True),
        )
        if inspect.isasyncgen(resp):
            await print_stream(resp)
        else:
            print(resp)

    except Exception as e:
        print(f"发生了错误: {e}")
    finally:
        if rag:
            await rag.finalize_storages()


if __name__ == "__main__":
    # Configure logging before running the main function
    configure_logging()
    asyncio.run(main())
    print("\n文档插入完成啦ε==(づ′▽`)づ")