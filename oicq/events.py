from ncatbot.core.event import GroupMessageEvent, PrivateMessageEvent
from logging import getLogger
from .globl import qapi, qbot, real_friend_id_list
from config import *
from .msgfmt import format_msg_oneline
from time import time
from utils import get_date
from asyncio_channel import create_channel, create_sliding_buffer
from .status import get_group_active

logger = getLogger(__name__)

eventchan = create_channel(create_sliding_buffer(100))  # type: ignore


@qbot.on_group_message()  # type: ignore
async def group_message_handler(event: GroupMessageEvent):
    if event.group_id != CON:
        if event.message.is_user_at(USR):
            logger.info(f"提及我的消息 {event.raw_message} 送入eventchan")
            await eventchan.put(event)
        elif time() < await get_group_active(event.group_id):
            logger.info(f"观望的消息 {event.raw_message} 送入eventchan")
            await eventchan.put(event)


@qbot.on_private_message()  # type: ignore
async def private_message_handler(event: PrivateMessageEvent):
    friends = await real_friend_id_list()
    if event.sender.user_id in friends:
        logger.info(f"好友消息 {event.raw_message} 送入eventchan")
        await eventchan.put(event)


async def format_events(events) -> str:
    lines = []
    for ev in events:
        if isinstance(ev, GroupMessageEvent):
            if ev.message.is_user_at(USR):
                lines.append(f"Notify: 提及你的群消息: {await format_msg_oneline(ev)}")
            else:
                lines.append(f"Event: 实时群消息: {await format_msg_oneline(ev)}")
        elif isinstance(ev, PrivateMessageEvent):
            lines.append(f"Notify: 好友私信消息: {await format_msg_oneline(ev)}")
        else:
            lines.append(f"Notify: {ev}")
    return "\n".join(lines)


async def wait_events(until: float) -> str | None:  # FIXME
    while True:
        timeout = min(5, max(1, until - time()))
        if await eventchan.item(timeout=timeout):
            break
        if time() > until:
            return None
    events = []
    while ev := await eventchan.take(timeout=0):
        events.append(ev)
    return await format_events(events)
