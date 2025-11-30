from dataclasses import dataclass
from typing import List, TypedDict
from langgraph.graph import add_messages
from langchain_core.messages import AnyMessage
from config import *
from app import App
from typing import Annotated
from datetime import datetime


class Idle(TypedDict):
    idle_minutes: int | None


class BotState(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], add_messages]
    info_inject: str | None
    idle_until: float | None


@dataclass
class BotContext:
    app: App
