from dataclasses import dataclass
from langgraph.graph import add_messages, MessagesState
from config import *
from app import App


class BotState(MessagesState):
    pass


@dataclass
class BotContext:
    app: App
