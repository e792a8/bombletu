from dataclasses import dataclass
from langchain.tools import tool
from langgraph.prebuilt.tool_node import ToolNode
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import add_messages, MessagesState
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from langgraph.types import Command
from langchain_core.embeddings import Embeddings
from langchain_core.rate_limiters import InMemoryRateLimiter
from os import environ
from langchain.messages import (
    AnyMessage,
    SystemMessage,
    AIMessage,
    HumanMessage,
    RemoveMessage,
)
from typing_extensions import TypedDict, Annotated
from typing import Literal
from langgraph.graph import StateGraph, START, END
from langgraph.func import task, entrypoint
from langchain_core.runnables import RunnableConfig
from config import *
from cqface import CQFACE
from adapt import GiteeAIEmbeddings
from mem0 import Memory
from langchain.agents.middleware import SummarizationMiddleware

logger = get_log(__name__)


llm = ChatOpenAI(
    rate_limiter=InMemoryRateLimiter(requests_per_second=1),
    max_retries=30,
    temperature=0.6,
    model=environ["LLM_MODEL"],
    api_key=environ["LLM_API_KEY"],  # type:ignore
    base_url=environ["LLM_BASE_URL"],
)


embed = GiteeAIEmbeddings(
    dimensions=1024,
    model=environ["EMBED_MODEL"],
    base_url=environ["EMBED_BASE_URL"],
    api_key=environ["EMBED_API_KEY"],  # type: ignore
)


def make_chroma(col: str, persist_dir: str | None = None):
    # embed = OpenAIEmbeddings(
    return Chroma(col, embedding_function=embed, persist_directory=persist_dir)


def make_mem0() -> Memory:
    mem0_config = {
        "vector_store": {
            "provider": "langchain",
            "config": {"client": make_chroma("mem0", DATADIR + "/chroma")},
        },
        "llm": {"provider": "langchain", "config": {"model": llm}},
        "embedder": {"provider": "langchain", "config": {"model": embed}},
        "reranker": {
            "provider": "llm_reranker",
            "config": {
                "llm": {"provider": "langchain", "config": {"model": llm}},
            },
        },
    }
    mem = Memory.from_config(mem0_config)
    return mem
