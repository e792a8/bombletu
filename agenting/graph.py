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
from typing import Literal, TYPE_CHECKING
from langgraph.graph import StateGraph, START, END
from langgraph.func import task, entrypoint
from langchain_core.runnables import RunnableConfig
from config import *
from cqface import CQFACE
from adapt import GiteeAIEmbeddings
from mem0 import Memory
from .types import BotContext, BotState, Idle
from langchain.agents.middleware import SummarizationMiddleware
from .summarization import summarize
from .prompts import SYSTEM_PROMPT, INITIAL_PROMPTS
from langgraph.runtime import Runtime
from components import llm, embed

if TYPE_CHECKING:
    from app import App

logger = get_log(__name__)

type Rt = Runtime[BotContext]

# model = ChatOllama(
#     model="qwen3:0.6b", base_url="http://192.168.66.1:11434", reasoning=True
# )
# model = init_chat_model("anthropic:claude-sonnet-4-5", temperature=0)

# @task
# async def make_memory(msgs: list[AnyMessage]):
#     PROMPT = """根据以上交互记录，提取需要长期记忆的重点内容，每行一条："""
#     msgs = (
#         [SystemMessage(SYSTEM_PROMPT), HumanMessage("[ignore this]")]
#         + msgs
#         + [SystemMessage(PROMPT)]
#     )
#     ret = await model_with_tools.ainvoke(msgs)
#     return ret.text.split("\n")


# @task
# async def fetch_memory(msgs: list[AnyMessage]):
#     PROMPT = """根据以上交互记录，提取与当前环境"""


async def state_guard(state: BotState, runtime: Rt) -> BotState:
    return {
        "messages": [HumanMessage(state.get("info_inject"))],
        "info_inject": None,
        "idle_until": None,
    }


async def llm_call(state: BotState, runtime: Rt):
    """LLM decides whether to call a tool or not"""

    llm_with_tools = runtime.context.app.llm_with_tools
    prompts = INITIAL_PROMPTS + state.get("messages", [])
    return {
        "messages": [llm_with_tools.invoke(prompts)],
    }


async def context_ng(state: BotState, runtime: Rt):
    llm_with_tools = runtime.context.app.llm_with_tools
    if (cur_msgs := state.get("messages", None)) is None:
        return None
    msgs = INITIAL_PROMPTS + cur_msgs
    sum = await summarize(llm_with_tools, msgs)
    if sum:
        return {"messages": sum}


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


def make_agent(
    app: "App",
    ckptr: BaseCheckpointSaver = InMemorySaver(),
    store_dir: str | None = None,
):
    # Build workflow
    builder = StateGraph(BotState, BotContext)

    # Add edges to connect nodes
    builder.add_sequence(
        [
            state_guard,
            llm_call,
            ("tool_node", ToolNode(app.model_tools)),
            context_ng,
        ]
    )
    builder.set_entry_point("state_guard")
    builder.set_finish_point("context_ng")

    # Compile the agent
    agent = builder.compile(checkpointer=ckptr)
    return agent


def make_agent_deep(
    app: "App",
    ckptr: BaseCheckpointSaver = InMemorySaver(),
    store_dir: str | None = None,
):
    from deepagents import create_deep_agent
    from deepagents.backends import FilesystemBackend
    from langchain.agents.structured_output import ToolStrategy

    agent = create_deep_agent(
        llm,
        app.model_tools,
        backend=FilesystemBackend(DATADIR + "/agentfs"),
        system_prompt=SYSTEM_PROMPT,
        context_schema=BotContext,
        response_format=ToolStrategy(Idle),
    )
    return agent


# agent = create_agent(
#     model=llm, tools=tools, system_prompt=SYSTEM_PROMPT, middleware=[after_model_do]
# )


def main():
    from ncatbot.utils.logger import setup_logging

    setup_logging()
    # cr = make_chroma("test")
    # cr.add_texts("我家在长春。")


if __name__ == "__main__":
    main()
