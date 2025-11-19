import asyncio
from datetime import datetime
from pytz import timezone
from typing import List
from ncatbot.core.api import BotAPI, NapCatAPIError
from ncatbot.core import MessageArray
from ncatbot.core.event import GroupMessageEvent
from ncatbot.core.event.message_segment.message_segment import (
    PlainText,
    At,
    AtAll,
    Face,
    Reply,
    Image,
    Video,
    Forward,
)
import re
from cqface import CQFACE, RCQFACE
from config import *

logger = get_log(__name__)

MSGP = re.compile(r"(\[\:.*?\])")


def format_face(id: str):
    if not id.isdigit():
        logger.warning(f"invalid face id {id}")
        return id
    iid = int(id)
    if not iid in CQFACE.keys():
        logger.warning(f"unknown face id {iid}")
        return id
    return CQFACE[iid]


def parse_face(name: str):
    if name.isdigit():
        return name
    if not name in RCQFACE.keys():
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


async def format_msg(msg: MessageArray, api: BotAPI) -> str:
    m = ""
    for seg in msg:
        if isinstance(seg, PlainText):
            m += seg.text
        elif isinstance(seg, AtAll):
            m += "[:at ALL]"
        elif isinstance(seg, At):
            try:
                u = await api.get_group_member_info(GRP, seg.qq)
                nick = u.card or u.nickname
            except NapCatAPIError as e:
                logger.error(f"napcat api: {e}")
                nick = ""
            m += f"[:at {'ME ' if seg.qq == USR else ''}{seg.qq} ({nick})]"
        elif isinstance(seg, Face):
            m += f"[:face {format_face(seg.id)}]"
        elif isinstance(seg, Reply):
            m += f"[:refer {seg.id}]"
        elif isinstance(seg, Image):
            m += f"[:image {seg.file}]"
        # elif isinstance(seg, Video):
        #     m += f"[:video {seg.file_id}]"
        # elif isinstance(seg, Forward):
        #     m += f"[:forward {seg.id}]"
        else:
            m += "[:unsupported]"
    return m


async def msglfmt(events: List[GroupMessageEvent], with_id: bool | str, api: BotAPI):
    d = None
    t = None
    fro = None
    l = []
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
            if e.sender.user_id == USR:
                infoln += f"[from ME {e.sender.user_id} ({e.sender.card or e.sender.nickname})]"
            else:
                infoln += (
                    f"[from {e.sender.user_id} ({e.sender.card or e.sender.nickname})]"
                )
        fro = e.sender.user_id
        if len(infoln) > 0:
            l.append(infoln)
        msgln = ""
        if with_id == True:
            msgln += f"[id {e.message_id}]"
        elif with_id == e.message_id:
            msgln += "[this]"
        msgln += await format_msg(e.message, api)
        l.append(msgln)
    return "\n".join(l)


def main():
    msg = "[:refer 353734][:at 3266073720][:at ALL]haha[:face 11]"
    # msg2 = asyncio.run(format_msg(parse_msg(msg), None))
    print(msg)
    # print(msg2)


if __name__ == "__main__":
    main()
