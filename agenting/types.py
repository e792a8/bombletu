from dataclasses import dataclass
from typing import List, TypedDict
from langgraph.graph import add_messages
from langgraph.runtime import Runtime
from langchain_core.messages import AnyMessage
from config import *
from app import App
from typing import Annotated
from datetime import datetime
from langchain.tools import tool, ToolRuntime


class Idle(TypedDict):
    idle_minutes: int | None


class BotState(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], add_messages]
    info_inject: str | None
    idle_until: float | None
    notes: list[str]


@dataclass
class BotContext:
    app: App


GraphRt = Runtime[BotContext]

ToolRt = ToolRuntime[BotContext, BotState]
# NOTE langgraph 1.0.3, ToolRuntime inject only supports:
# argument name being exactly "runtime", or argument type being `ToolRuntime`
# without type arguments.
