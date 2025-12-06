import asyncio
from dataclasses import dataclass
from chromadb.api.types import Embeddings
from langchain.tools import tool
from langgraph.prebuilt.tool_node import ToolNode
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from chromadb import PersistentClient
from langchain_chroma import Chroma
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import add_messages, MessagesState
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from langgraph.types import Command
from langchain_core.embeddings import Embeddings
from langchain_core.rate_limiters import InMemoryRateLimiter
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
from mem0 import AsyncMemory
from langchain.agents.middleware import SummarizationMiddleware

logger = get_log(__name__)


llm = ChatOpenAI(
    rate_limiter=InMemoryRateLimiter(requests_per_second=1),
    max_retries=30,
    temperature=0.6,
    model=ENV["LLM_MODEL"],
    api_key=ENV["LLM_API_KEY"],  # type:ignore
    base_url=ENV["LLM_BASE_URL"],
)

llm2 = ChatOpenAI(
    rate_limiter=InMemoryRateLimiter(requests_per_second=1),
    max_retries=30,
    temperature=0.6,
    model=ENV["LLM2_MODEL"],
    api_key=ENV["LLM2_API_KEY"],  # type:ignore
    base_url=ENV["LLM2_BASE_URL"],
)


embed = GiteeAIEmbeddings(
    dimensions=int(ENV["EMBED_DIMENSIONS"]),
    model=ENV["EMBED_MODEL"],
    base_url=ENV["EMBED_BASE_URL"],
    api_key=ENV["EMBED_API_KEY"],  # type: ignore
)


def make_chroma(col: str, persist_dir: str | None = None):
    # embed = OpenAIEmbeddings(
    return Chroma(col, embedding_function=embed, persist_directory=persist_dir)
    # from chromadb import Documents, Embeddings, EmbeddingFunction
    # import numpy as np

    # class EmbedWrap(EmbeddingFunction):
    #     def __init__(self, *args, **kwargs):
    #         super().__init__(*args, **kwargs)

    #     @staticmethod
    #     def name():
    #         return "embed_wrap"

    #     def __call__(self, input: Documents) -> Embeddings:
    #         return [np.array(em) for em in embed.embed_documents(input)]

    # chroma = PersistentClient(persist_dir)
    # coll = chroma.get_or_create_collection(col, embedding_function=EmbedWrap())  # type: ignore
    # return coll


def make_mem0() -> AsyncMemory:
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
    mem = asyncio.run(AsyncMemory.from_config(mem0_config))
    return mem
