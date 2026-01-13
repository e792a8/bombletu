import asyncio
from time import time
from config import USR
from utils import get_date
from .globl import (
    get_messages_wrapped,
    qapi,
    real_friend_id_list,
    real_group_id_list,
    ChatTy,
)
from .msgfmt import format_user, format_group_name
from ncatbot.core.event.message import PrivateMessageEvent, GroupMessageEvent

read_status = {}
read_status_lock = asyncio.locks.Lock()

group_watch = {}
group_watch_lock = asyncio.locks.Lock()


async def set_group_watch(group: str, until: float):
    async with group_watch_lock:
        group_watch[group] = until


async def get_group_watch(group: str) -> float:
    async with group_watch_lock:
        return group_watch.get(group, 0)


async def clear_unread(cty: ChatTy, cid: str):
    async with read_status_lock:
        last = await get_messages_wrapped(cty, cid, 0, 1)
        if len(last) == 0:
            return
        read_status[(cty, cid)] = last[0].message_id


async def init_read_status():
    groups = await real_group_id_list()
    for g in groups:
        await clear_unread("group", g)  # type: ignore
    friends = await real_friend_id_list()
    for u in friends:
        await clear_unread("private", u)


def calc_unread(rd: str, msgs: list[PrivateMessageEvent] | list[GroupMessageEvent]):
    if rd == "0":
        return len(msgs)
    for i, m in enumerate(reversed(msgs)):
        if m.message_id == rd:
            return i
    return len(msgs)


async def get_unread(cty: ChatTy, cid: str) -> tuple[int, int]:  # (unread, mention)
    msgs = await get_messages_wrapped(cty, cid, 0, 110)
    unread = len(msgs)
    mention = 0
    if unread == 0:
        return 0, 0
    async with read_status_lock:
        rd = read_status.get((cty, cid), "0")
    for i, m in enumerate(reversed(msgs)):
        if m.message.is_user_at(USR):
            mention += 1
        if m.message_id == rd:
            unread = i
            break
    return unread, mention


async def collect_unread() -> tuple[int, int, int, int]:
    """(private, mention, watch, group)"""
    friends = await real_friend_id_list()
    private = 0
    for f in friends:
        private += (await get_unread("private", f))[0]
    groups = await real_group_id_list()
    mention = 0
    watch = 0
    group = 0
    now = time()
    for g in groups:
        u, m = await get_unread("group", g)
        if now < await get_group_watch(g):
            watch += u
        group += u
        mention += m
    return private, mention, watch, group


async def get_chats_info(important_only=False):
    now = time()

    friends = await real_friend_id_list()
    friends_sum = []
    friends_sum_inactive = []
    for f in friends:
        unread, _ = await get_unread("private", f)
        msg = await get_messages_wrapped("private", f, 0, 1)
        active = msg[0].time if len(msg) else None
        if active:
            friends_sum.append(("private", f, active, unread))
        else:
            friends_sum_inactive.append(("private", f, active, unread))

    groups = await real_group_id_list()
    groups_sum = []
    groups_sum_inactive = []
    for g in groups:
        unread, mention = await get_unread("group", g)
        msg = await get_messages_wrapped("group", g, 0, 1)
        active = msg[0].time if len(msg) else None
        if active:
            groups_sum.append(("group", g, active, unread, mention))
        else:
            groups_sum_inactive.append(("group", g, active, unread, mention))

    friends_sum.sort(key=lambda x: -x[2])
    groups_sum.sort(key=lambda x: -x[2])
    total = []
    for x in friends_sum + groups_sum + friends_sum_inactive + groups_sum_inactive:
        if x[0] == "private":
            _, uid, active, unread = x
            if important_only and unread <= 0:
                continue
            disp = f"[private {await format_user(uid)}][最近活跃时间: {get_date(active) if active else '无消息'}][未读: {'99+' if unread > 99 else unread}]"
            if unread > 0:
                disp += "[提醒: 好友私信消息]"
        else:
            _, gid, active, unread, mention = x
            watch = await get_group_watch(gid)
            if important_only and mention <= 0 and not (unread > 0 and now < watch):
                continue
            disp = f"[group {await format_group_name(gid)}][最近活跃时间: {get_date(active) if active else '无消息'}][未读: {'99+' if unread > 99 else unread}]"
            if mention > 0:
                disp += f"[提醒: 提及你的消息: {'99+' if mention > 99 else mention}]"
            if unread > 0 and now < watch:
                disp += f"[提醒: 关注中的群]"
        total.append(disp)

    return "\n".join(total)


async def get_status() -> str:
    private, mention, watch, group = await collect_unread()
    minor = group - mention - watch
    if minor > 0:
        return f"次要消息: {'99+' if minor > 99 else minor}"
    return ""
