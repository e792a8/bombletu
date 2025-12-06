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
from .types import BotContext, BotState, Idle, GraphRt
from langchain.agents.middleware import SummarizationMiddleware
from .summarization import summarize
from .prompts import SYSTEM_PROMPT, INITIAL_PROMPTS
from langgraph.runtime import Runtime
from components import llm, embed
from .contexting import context_ng

if TYPE_CHECKING:
    from app import App

logger = get_log(__name__)


async def state_guard(state: BotState, runtime: GraphRt) -> BotState:
    return {
        "messages": [HumanMessage(state.get("info_inject"))],
        "info_inject": None,
        "idle_until": None,
    }


async def llm_call(state: BotState, runtime: GraphRt):
    """LLM decides whether to call a tool or not"""

    llm_with_tools = runtime.context.app.llm_with_tools
    prompts = INITIAL_PROMPTS + state.get("messages", [])
    return {
        "messages": [llm_with_tools.invoke(prompts)],
    }


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
