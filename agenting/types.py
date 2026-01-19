from dataclasses import dataclass, field
from typing import TypedDict
from langgraph.graph import add_messages
from langgraph.runtime import Runtime
from langchain_core.messages import AnyMessage
from pydantic_ai import AbstractToolset
from applet.base import BaseApplet
from config import *
from typing import Annotated
from langchain.tools import ToolRuntime, BaseTool
from pydantic_ai.messages import ModelMessage


@dataclass
class BotState:
    messages: list[ModelMessage] = field(default_factory=list)
    summary: str | None = field(default=None)
    info_inject: str | None = field(default=None)
    idle_minutes: int | None = field(default=None)
    idle_until: float | None = field(default=None)
    tire_level: float = field(default=0)


@dataclass
class BotDeps:
    toolset: AbstractToolset["BotDeps"]
    applet_instructions: str


GraphRt = Runtime[BotDeps]

ToolRt = ToolRuntime[BotDeps, BotState]
# NOTE langgraph 1.0.3, ToolRuntime inject only supports:
# argument name being exactly "runtime", or argument type being `ToolRuntime`
# without type arguments.
