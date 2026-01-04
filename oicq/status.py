import asyncio
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

group_active = {}
group_active_lock = asyncio.locks.Lock()


async def set_group_active(group: str, until: float):
    async with group_active_lock:
        group_active[group] = until


async def get_group_active(group: str) -> float:
    async with group_active_lock:
        return group_active.get(group, 0)


async def clear_unread(cty: ChatTy, cid: str):
    async with read_status_lock:
        last = await get_messages_wrapped(cty, cid, 0, 1)
        if len(last) == 0:
            return
        read_status[f"{cty} {cid}"] = last[0].message_id


async def init_read_status():
    groups = await qapi.get_group_list()
    for g in groups:
        await clear_unread("group", g)  # type: ignore
    friends = await qapi.get_friend_list()
    for u in friends:
        await clear_unread("private", u["user_id"])


def calc_unread(rd: str, msgs: list[PrivateMessageEvent] | list[GroupMessageEvent]):
    if rd == "0":
        return len(msgs)
    for i, m in enumerate(reversed(msgs)):
        if m.message_id == rd:
            return i
    return len(msgs)


async def get_chats_info():
    friends = await real_friend_id_list()
    friends_disp = [f"[private {await format_user(f)}]" for f in friends]
    friends_msgs = [await get_messages_wrapped("private", f, 0, 110) for f in friends]
    friends_sum = []
    friends_sum_inactive = []

    groups = await real_group_id_list()
    groups_disp = [f"[group {await format_group_name(g)}]" for g in groups]
    groups_msgs = [await get_messages_wrapped("group", g, 0, 110) for g in groups]
    groups_sum = []
    groups_sum_inactive = []
    async with read_status_lock:
        for f, fd, fm in zip(friends, friends_disp, friends_msgs):
            active = None
            unread = 0
            if len(fm) > 0:
                active = fm[-1].time
                rd = read_status.get(f"private {f}", "0")
                unread = calc_unread(rd, fm)
            if active:
                friends_sum.append(("private", f, fd, active, unread))
            else:
                friends_sum_inactive.append(("private", f, fd, active, unread))
        for g, gd, gm in zip(groups, groups_disp, groups_msgs):
            active = None
            unread = 0
            if len(gm) > 0:
                active = gm[-1].time
                rd = read_status.get(f"group {g}", "0")
                unread = calc_unread(rd, gm)
            if active:
                groups_sum.append(("group", g, gd, active, unread))
            else:
                groups_sum_inactive.append(("group", g, gd, active, unread))
    friends_sum.sort(key=lambda x: -x[3])
    groups_sum.sort(key=lambda x: -x[3])
    total = []
    nowdate, nowtime = get_date().split()
    for x in friends_sum + groups_sum + friends_sum_inactive + groups_sum_inactive:
        disp = x[2]
        if not x[3]:
            ac = ""
        else:
            adate, atime = get_date(x[3]).split()
            ac = f"[{atime}]" if nowdate == adate else f"[{adate} {atime}]"
        unr = f"[unread {'99+' if x[4] > 99 else x[4]}]"
        total.append(f"{disp}{ac}{unr}")
    return "\n".join(total)


async def get_status() -> str:
    # TODO idk what to do
    return ""
