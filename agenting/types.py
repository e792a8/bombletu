from dataclasses import dataclass
from typing import List
from langgraph.graph import add_messages, MessagesState
from config import *
from app import App


class BotState(MessagesState):
    idle_minutes: int | None


@dataclass
class BotContext:
    app: App
