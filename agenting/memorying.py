from langchain_core.messages import (
    AnyMessage,
    AIMessage,
    HumanMessage,
    RemoveMessage,
    SystemMessage,
)
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from langchain_core.language_models import LanguageModelLike
from config import *
from os import environ
from powermem import AsyncMemory

# KEEP_MSGS = 4
# MSGS_THRESH = 10
KEEP_MSGS = 20
MSGS_THRESH = 50


def make_mem():
    config = {
        "vector_store": {
            "provider": "sqlite",
            "config": {
                "database_path": DATADIR + "/powermem/mem.db",
            },
        },
        "llm": {
            "provider": "openai",
            "config": {
                "api_key": environ.get("LLM_API_KEY"),
                "model": environ.get("LLM_MODEL"),
                "openai_base_url": environ.get("LLM_BASE_URL"),
            },
        },
        "embedder": {
            "provider": "openai",
            "config": {
                "api_key": environ.get("EMBED_API_KEY"),
                "model": environ.get("EMBED_MODEL"),
                "openai_base_url": environ.get("EMBED_BASE_URL"),
                "embedding_dims": 1536,
            },
        },
    }
    return AsyncMemory(config)


mem = make_mem()


async def fetch_memory(model: LanguageModelLike, msgs: list[AnyMessage]):
    if len(msgs) <= MSGS_THRESH:
        return None
    cut = 0
    for i in range(len(msgs) - KEEP_MSGS, -1, -1):
        msg = msgs[i]
        if isinstance(msg, AIMessage) and msg.tool_calls:
            continue
        else:
            cut = i
            break
    if cut <= 0:
        return None
    msgs_to_sum = msgs[:cut]
    msgs_to_keep = msgs[cut:]
    sum = await model.ainvoke(
        msgs_to_sum
        + [
            SystemMessage(SUMMARY_PROMPT),
            HumanMessage("输出提取的上下文摘要文本："),
        ]
    )
    return [
        RemoveMessage(REMOVE_ALL_MESSAGES),
        HumanMessage(content="之前的交互历史摘要：\n" + sum.text),  # type: ignore
    ] + msgs_to_keep
