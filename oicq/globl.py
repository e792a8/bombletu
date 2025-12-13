from typing import Literal
from mcp.server.fastmcp import FastMCP
from ncatbot.core.client import BotClient
from ncatbot.core.event.message import GroupMessageEvent, PrivateMessageEvent
from ncatbot.core.api import NapCatAPIError
from sys import argv
from config import CON, USR
from logging import getLogger

logger = getLogger(__name__)

qbot = BotClient()
qapi = qbot.api

port = int(argv[1]) if len(argv) > 1 and argv[1].isdigit() else None
if port:
    mcp = FastMCP("oicq", port=port)
else:
    mcp = FastMCP("oicq")

ChatTy = Literal["friend", "group"]


async def real_friend_id_list() -> list[str]:
    lst = await qapi.get_friend_list()
    for i, u in enumerate(lst):
        if str(u["user_id"]) == USR:
            del lst[i]
            break
    return [str(u["user_id"]) for u in lst]


async def real_group_id_list() -> list[str]:
    lst = await qapi.get_group_list()
    for i, g in enumerate(lst):
        if g == CON:
            del lst[i]
            break
    return lst  # type: ignore


async def get_messages_wrapped(
    cty: ChatTy, cid: str, msgid: str | int, count: int = 20, reverse: bool = False
) -> list[GroupMessageEvent] | list[PrivateMessageEvent]:
    try:
        if cty == "group":
            return await qapi.get_group_msg_history(cid, msgid, count, reverse)
        else:
            return await qapi.get_friend_msg_history(cid, msgid, count, reverse)
    except NapCatAPIError as e:
        logger.error(f"get messages error: {cty} {cid} {msgid}: {e}")
        return []
