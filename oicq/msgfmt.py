from datetime import datetime
from pytz import timezone
from ncatbot.core.api import NapCatAPIError
from ncatbot.core import MessageArray
from ncatbot.core.event import GroupMessageEvent, PrivateMessageEvent
from ncatbot.core.event.message_segment.message_segment import (
    PlainText,
    At,
    AtAll,
    Face,
    Reply,
    Image,
    Forward,
)
import re
from .cqface import CQFACE, RCQFACE
from .globl import ChatTy, qapi
from config import *

logger = get_log(__name__)

MSGP = re.compile(r"(\[\:.*?\])")


def format_face(id: str):
    if not id.isdigit():
        logger.warning(f"invalid face id {id}")
        return id
    iid = int(id)
    if iid not in CQFACE.keys():
        logger.warning(f"unknown face id {iid}")
        return id
    return CQFACE[iid]


async def format_group_member(group: str, uid: str) -> str:
    """格式化为`ID (昵称)`的形式"""
    name = ""
    try:
        u = await qapi.get_group_member_info(group, uid)
        name = u.card or u.nickname
    except NapCatAPIError as e:
        logger.error(f"napcat api: {e}")
    return f"{'ME ' if uid == USR else ''}{uid} ({name})"


async def format_group_name(group: str) -> str:
    """格式化为`ID (群备注或群名)`的形式"""
    name = ""
    try:
        g = await qapi.get_group_info(group)
        name = g.group_remark or g.group_name
    except NapCatAPIError as e:
        logger.error(f"napcat api: {e}")
    return f"{group} ({name})"


async def format_friend(uid: str) -> str:
    """格式化为`ID (昵称)`的形式"""
    u = None
    name = ""
    try:
        for u_ in await qapi.get_friend_list():
            if u_["user_id"] == uid:
                u = u_
                break
        if not u:
            u = await qapi.get_stranger_info(uid)
        name = u.get("remark")
        if not name:
            name = u.get("nickname", "")
    except NapCatAPIError as e:
        logger.error(f"napcat api: {e}")
    return f"{uid} ({name})"


def parse_face(name: str):
    if name.isdigit():
        return name
    if name not in RCQFACE.keys():
        logger.warning(f"unknown face name {name}")
        return None
    return RCQFACE[name]


def parse_msg(msg: str) -> MessageArray:
    mch = re.split(MSGP, msg)
    m = MessageArray()
    for seg in mch:
        if not (seg.startswith("[:") and seg.endswith("]")):
            m += PlainText(seg)
            continue
        sq = seg[2:-1]
        sp = sq.split()
        cmd = sp[0]
        if cmd == "at":
            if sp[1] == "ALL":
                m += AtAll()
            else:
                m += At(sp[1])
        elif cmd == "face":
            fid = parse_face(sp[1])
            if fid is not None:
                m += Face(fid)
        elif cmd == "refer":
            m += Reply(sp[1])
        else:
            logger.warning(f"unknown seg: {seg}")
            m += PlainText(seg[2:-1])
    return m


async def format_msg(msg: MessageArray, group_id: str | None = None) -> str:
    m = ""
    for seg in msg:
        if isinstance(seg, PlainText):
            m += seg.text
        elif isinstance(seg, AtAll):
            m += "[:at ALL]"
        elif isinstance(seg, At):
            if group_id:
                m += f"[:at {await format_group_member(group_id, seg.qq)}]"
            else:
                m += f"[:at {await format_friend(seg.qq)}]"
        elif isinstance(seg, Face):
            m += f"[:face {format_face(seg.id)}]"
        elif isinstance(seg, Reply):
            m += f"[:refer {seg.id}]"
        elif isinstance(seg, Image):
            m += f"[:image {seg.file}]"
        # elif isinstance(seg, Video):
        #     m += f"[:video {seg.file_id}]"
        elif isinstance(seg, Forward):
            m += f"[:forward {seg.id}]"
        else:
            m += "[:unsupported]"
    return m


async def format_msg_oneline(e: GroupMessageEvent | PrivateMessageEvent) -> str:
    if isinstance(e, GroupMessageEvent):
        return f"[group {await format_group_name(e.group_id)}][from {await format_group_member(e.group_id, e.sender.user_id)}]{await format_msg(e.message, e.group_id)}"
    elif isinstance(e, PrivateMessageEvent):
        return f"[friend {await format_friend(e.sender.user_id)}]{await format_msg(e.message)}"


async def format_from_str(
    cty: ChatTy, cid: str, e: GroupMessageEvent | PrivateMessageEvent
):
    if cty == "group":
        return await format_group_member(cid, e.sender.user_id)
    elif e.sender.user_id == cid:
        return "THEM"
    elif e.sender.user_id == USR:
        return "ME"
    return e.sender.user_id


async def msglfmt(
    cty: ChatTy,
    cid: str,
    events: list[GroupMessageEvent] | list[PrivateMessageEvent],
    with_id: bool | str,
):
    header = None
    if cty == "friend":
        header = f"[friend {await format_friend(cid)}]"
    elif cty == "group":
        header = f"[group {await format_group_name(cid)}]"
    d = None
    t = None
    fro = None
    l = []
    if header:
        l.append(header)
    for e in events:
        infoln = ""
        dt = datetime.fromtimestamp(e.time, tz=timezone(TZ))
        ed = dt.date().isoformat()
        et = dt.timetz().isoformat(timespec="minutes")[:5]
        if ed != d:
            infoln += f"[on {ed} {et}]"
        elif et != t:
            infoln += f"[on {et}]"
        d, t = ed, et
        if e.sender.user_id != fro:
            infoln += f"[from {await format_from_str(cty, cid, e)}]"
        fro = e.sender.user_id
        if len(infoln) > 0:
            l.append(infoln)
        msgln = ""
        if with_id == True:
            msgln += f"[id {e.message_id}]"
        elif with_id == e.message_id:
            msgln += "[this]"
        msgln += await format_msg(e.message, cid if cty == "group" else None)
        l.append(msgln)
    return "\n".join(l)


def main():
    msg = "[:refer 353734][:at 3266073720][:at ALL]haha[:face 11]"
    # msg2 = asyncio.run(format_msg(parse_msg(msg), None))
    print(msg)
    # print(msg2)


if __name__ == "__main__":
    main()
