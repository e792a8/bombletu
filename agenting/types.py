from dataclasses import dataclass
from typing import List, TypedDict
from langgraph.graph import add_messages, MessagesState
from config import *
from app import App


class Idle(TypedDict):
    idle_minutes: int | None


class BotState(MessagesState):
    idle_minutes: int | None


@dataclass
class BotContext:
    app: App
