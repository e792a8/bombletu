from dataclasses import dataclass
from typing import Callable, List
from ncatbot.core import BotClient, MessageArray
from ncatbot.core.event import GroupMessageEvent
from asyncio_channel import create_channel, create_sliding_buffer
import asyncio
from datetime import datetime
from pytz import timezone
from ncatbot.core.api import NapCatAPIError
import os, signal
from langchain.messages import AIMessage, ToolMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langchain_chroma import Chroma
from langgraph.graph.state import CompiledStateGraph
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.checkpoint.memory import InMemorySaver
from time import time
from msgfmt import msglfmt, parse_msg
import traceback
from config import *

logger = get_log(__name__)


class App:
    async def wait_intr(self, minutes: int):  # FIXME
        target_time = time() + minutes * 60
        while True:
            intr = await self.intrchan.take(timeout=5)
            if intr is not None:
                return intr
            if time() > target_time:
                return None

    async def collect_unread(self) -> int:
        collected = 0
        while ev := await self.newmsgchan.take(timeout=0):
            self.unread.append(ev)
            collected += 1
        return collected

    async def get_unread(self, limit: int) -> str:
        await self.collect_unread()
        l = []
        for _ in range(limit):
            if len(self.unread) < 1:
                break
            l.append(self.unread.pop(0))
        return "\n".join(
            [await msglfmt(l, False, self.qbot.api), f"[unread {len(self.unread)}]"]
        )

    def __init__(self, make_agent_loop: Callable, chroma: Chroma):
        async def group_message_handler(event: GroupMessageEvent):
            if event.group_id == GRP:
                await self.newmsgchan.put(event)
                if event.message.is_user_at(USR):
                    logger.info(
                        "group_message_handler: 提及我的消息 {event.raw_message} 送入intrchan"
                    )
                    await self.intrchan.put("提及我的消息")
            elif event.group_id == CON:
                if event.raw_message == "/kill":
                    logger.warning("panic called")
                    os.kill(os.getpid(), signal.SIGKILL)
                elif event.raw_message == "/term":
                    logger.info("exit called")
                    os.kill(os.getpid(), signal.SIGTERM)
                elif event.raw_message == "/int":
                    logger.info("exit called")
                    os.kill(os.getpid(), signal.SIGINT)

        qbot = BotClient()
        qbot.add_group_message_handler(group_message_handler)  # type: ignore
        qbot.add_startup_handler(make_agent_loop(self))  # type: ignore
        msgchan = create_channel(create_sliding_buffer(100))  # type: ignore
        intrchan = create_channel(create_sliding_buffer(1))  # type: ignore

        self.qbot = qbot
        self.qapi = qbot.api
        self.newmsgchan = msgchan
        self.intrchan = intrchan
        self.unread = []
        self.chroma = chroma

    def run(self):
        self.qbot.run_frontend()
