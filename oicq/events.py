from ncatbot.core.event import GroupMessageEvent, PrivateMessageEvent
from logging import getLogger
from .globl import qapi, qbot, real_friend_id_list
from config import *
from .msgfmt import format_msg_oneline
from time import time
from utils import get_date
from asyncio_channel import create_channel, create_sliding_buffer
from .status import get_chats_info
import asyncio

logger = getLogger(__name__)

eventchan = create_channel(create_sliding_buffer(100))  # type: ignore

eventflag = asyncio.Event()  # allows false positive


@qbot.on_group_message()  # type: ignore
async def group_message_handler(event: GroupMessageEvent):
    logger.info(f"群消息: {event}")
    eventflag.set()


@qbot.on_private_message()  # type: ignore
async def private_message_handler(event: PrivateMessageEvent):
    logger.info(f"私信消息: {event}")
    eventflag.set()


async def wait_events(until: float) -> str | None:
    # should omit false positive
    while True:
        timeout = min(5, max(1, until - time()))
        chats = await get_chats_info(important_only=True)
        if len(chats.strip()) > 0:
            return "重点消息通知:\n  " + chats.replace("\n", "\n  ")
        try:
            await asyncio.wait_for(eventflag.wait(), timeout)
            eventflag.clear()
        except asyncio.TimeoutError:
            if time() > until:
                return None
