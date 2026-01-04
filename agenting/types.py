from dataclasses import dataclass
from typing import TypedDict
from langgraph.graph import add_messages
from langgraph.runtime import Runtime
from langchain_core.messages import AnyMessage
from config import *
from typing import Annotated
from langchain.tools import ToolRuntime, BaseTool


class Idle(TypedDict):
    idle_minutes: int | None


class BotState(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], add_messages]
    summary: str | None
    info_inject: str | None
    idle_minutes: int | None
    idle_until: float | None
    tire_level: float


@dataclass
class BotContext:
    applet_instructions: str
    tools: list[BaseTool]


GraphRt = Runtime[BotContext]

ToolRt = ToolRuntime[BotContext, BotState]
# NOTE langgraph 1.0.3, ToolRuntime inject only supports:
# argument name being exactly "runtime", or argument type being `ToolRuntime`
# without type arguments.
