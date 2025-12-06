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
from typing import Literal
from langgraph.graph import StateGraph, START, END
from langgraph.func import task, entrypoint
from langchain_core.runnables import RunnableConfig
from config import *
from cqface import CQFACE
from adapt import GiteeAIEmbeddings
from mem0 import Memory
from .tools import ALL_TOOLS
from .types import BotContext, BotState, Idle, GraphRt
from langchain.agents.middleware import SummarizationMiddleware
from .summarization import summarize
from .prompts import SYSTEM_PROMPT, INITIAL_PROMPTS

logger = get_log(__name__)


async def context_ng(state: BotState, runtime: GraphRt):
    llm_with_tools = runtime.context.app.llm_with_tools
    if (cur_msgs := state.get("messages", None)) is None:
        return None
    msgs = INITIAL_PROMPTS + cur_msgs
    sum = await summarize(llm_with_tools, msgs)
    if sum:
        return {"messages": sum}
