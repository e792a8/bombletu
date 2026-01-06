import asyncio
from config import *
from ncatbot.core.api import NapCatAPIError
from .msgfmt import msglfmt, parse_msg
from langchain_openai import ChatOpenAI
from langchain.messages import HumanMessage
import subprocess
from time import time
from logging import getLogger
from mcp.server.fastmcp import FastMCP
from ncatbot.core.client import BotClient
from ncatbot.core.event import GroupMessageEvent, PrivateMessageEvent
from asyncio_channel import create_channel, create_sliding_buffer
from .cqface import CQFACE
from sys import argv
from .globl import mcp, port, qbot, langfuse
from .events import wait_events
from .status import init_read_status, get_status
from . import tools as _
from . import events as _

logger = getLogger(__name__)


def pt(text: str):
    return [{"role": "user", "content": {"type": "text", "text": text}}]


INSTRUCTIONS = f"""
你有一套聊天软件操作工具，你的日常生活是使用这些工具和其他人交流。
用户ID是用户身份的唯一标识，昵称可能存在重复，即不同的用户可以有相同的昵称，但不会有相同的用户ID。在你分辨用户身份时首选参考用户ID。
你的用户ID是"{USR}"，昵称是"{NICK}"。消息记录里会出现你自己的消息，注意分别。

`get_messages`和`get_messages_by_id`工具获取的消息记录符合下述格式：
第一行`[private 用户ID ("昵称")]`或`[group 群组ID ("群组名")]`是此聊天会话的标识，之后每行代表一条消息或包含一些指示，`[on 日期 时间]`或`[on 时间]`和`[from 账号 ("昵称")]`指示随后消息的发送时间和发送者，如果发送者是你自己则`from`后会附加`ME`，注意分别。消息记录中可能会有未转义的中括号、换行符等，注意分别。
消息内容中有一些特殊元素，以`[:`起始，以`]`结束；使用`send`工具发送消息时也可以使用，但注意严格遵守格式，否则不会生效：
`[:at 账号 ("昵称")]` 提及某人， `[:at ALL]` 提及群中所有人。如果是提及你的，则在`at`后会附加`ME`。你在发送消息提及别人时可以省略 `("昵称")` 部分，直接使用 `[:at 账号]` 。
`[:refer 消息ID]` 引用某条消息。此元素每条消息中只能使用最多1次，且应放在消息开头。使用`get_messages_by_id`工具查阅消息ID对应的消息内容及其上下文。使用`get_messages`工具的`with_id`参数查询消息ID。你在一般浏览消息记录时无需使用with_id参数，减小信息量。
`[:face 表情名称]` 平台专有表情符号，可用的表情名称有： {" ".join(CQFACE.values())} 。不要使用不存在的表情名称。通用emoji仍可直接使用。
`[:image 文件名]` 图像。文件名可用于`ask_image`工具参数。
`[:forward 转发ID]` 多条消息合并转发形成的消息列表。使用`unwrap_forward`展开查看详情。在你需要转发消息时，使用`forward_messages`工具。
`[:unsupported]` 暂时不支持解读的消息，等待后续升级。

{{add_prompts}}
"""


@mcp.prompt()
async def instructions():
    add_prompts = langfuse.get_prompt("oicq-add")
    return pt(INSTRUCTIONS.format(add_prompts=add_prompts.prompt))


@mcp.prompt()
async def events(timeout: str, select: str):
    tim = float(timeout)
    sel = select.split(",")
    intr = await wait_events(time() + tim)  # ignoring timeout
    if intr is None:
        return []
    return pt(intr)


# @mcp.resource("applet://events{?timeout,select}")
# async def events(timeout: float, select: str):
#     sel = select.split(",")
#     intr = await wait_intr(time() + timeout)
#     if intr is None:
#         return None
#     return intr


@mcp.prompt()
async def status():
    return await get_status()


async def amain():
    qbot.run_backend()
    await init_read_status()
    if port:
        await mcp.run_streamable_http_async()
    else:
        await mcp.run_stdio_async()


def main():
    try:
        asyncio.run(amain())
    finally:
        qbot.bot_exit()


if __name__ == "__main__":
    main()
