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
from ncatbot.core.event import GroupMessageEvent
from asyncio_channel import create_channel, create_sliding_buffer
from .cqface import CQFACE
from sys import argv

logger = getLogger(__name__)

qbot = BotClient()
qapi = qbot.api
newmsgchan = create_channel(create_sliding_buffer(100))  # type: ignore
intrchan = create_channel(create_sliding_buffer(1))  # type: ignore
unread_count = 0

port = int(argv[1]) if len(argv) > 1 and argv[1].isdigit() else None
if port:
    mcp = FastMCP("oicq", port=port)
else:
    mcp = FastMCP("oicq")


@qbot.on_group_message()  # type: ignore
async def group_message_handler(event: GroupMessageEvent):
    if event.group_id == GRP:
        await newmsgchan.put(event)
        if event.message.is_user_at(USR):
            logger.info(
                "group_message_handler: 提及我的消息 {event.raw_message} 送入intrchan"
            )
            await intrchan.put("提及我的消息")


async def wait_intr(until: float) -> str | None:  # FIXME
    while True:
        timeout = min(5, max(1, until - time()))
        intr = await intrchan.take(timeout=timeout)
        if intr is not None:
            return intr
        if time() > until:
            return None


async def _collect_unread() -> int:
    global unread_count
    collected = 0
    while ev := await newmsgchan.take(timeout=0):
        unread_count += 1
        collected += 1
    return collected


async def count_unread() -> int:
    global unread_count
    await _collect_unread()
    return unread_count


async def clear_unread():
    global unread_count
    unread_count = 0


@mcp.tool()
async def send(content: str) -> str:
    """在群里发送消息。"""
    logger.info(f"send: {content}")
    try:
        await qapi.send_group_msg(GRP, parse_msg(content).to_list())
        return "[success]"
    except NapCatAPIError as e:
        logger.warning(f"get_message error: {e}")
        return "[error 软件暂时故障]"


@mcp.tool()
async def get_messages(fro: int, to: int, with_id: bool = False) -> str:
    """查阅消息记录。
    参数fro,to表示消息序号区间的开始和结束，最新的消息序号为1，序号由新到旧递增，返回的列表按由旧到新的顺序排列。
    例：get_messages(fro=10,to=1)获取最新10条消息；get_messages(fro=30,to=21)获取最后第30到第21条消息。
    参数with_id控制是否附带消息ID，如为真则每条消息的行首将带有 [id 消息ID] 指示。
    调用该工具读取到最新消息（即fro=1或to=1）时，未读消息计数值将清零；如没有读取最新消息，则不影响未读消息计数值。"""
    if to > fro:
        fro, to = to, fro
    logger.info(f"get_messages: {fro}, {to}, {with_id}")
    if to == 1:
        await clear_unread()
    try:
        return await msglfmt(
            (await qapi.get_group_msg_history(GRP, 0, fro))[: fro - to + 1],
            with_id,
            qapi,
        )
    except NapCatAPIError as e:
        logger.warning(f"get_message error: {e}")
        return "[error 软件暂时故障]"


@mcp.tool()
async def get_messages_by_id(id: str, before: int = 0, after: int = 0) -> str:
    """按消息ID查阅消息记录。
    参数id为查阅的目标消息ID，before为在目标消息前附带的消息数量，after为在目标消息后附带的消息数量。
    返回的消息列表中，目标消息的行首带有指示 [this] ，该条消息的ID即为查询的ID。
    调用该工具不会影响未读消息计数值。
    """
    api = qapi
    try:
        bef = await api.get_group_msg_history(GRP, id, before + 1, True)
        aft = await api.get_group_msg_history(GRP, id, after + 1, False)
        lst = bef + aft[1:]
        return await msglfmt(lst, id, api)
    except NapCatAPIError as e:
        logger.error(f"{e}")
        return "[error 软件暂时故障]"


visual_model = ChatOpenAI(
    temperature=0.6,
    model=ENV["VIS_MODEL"],
    api_key=ENV["VIS_API_KEY"],  # type:ignore
    base_url=ENV["VIS_BASE_URL"],
)


@mcp.tool()
async def ask_image(file_name: str, prompt="详细描述图片内容") -> str:
    """向视觉模型询问关于某个图像的问题。
    参数file_name是图像的文件名。参数prompt是询问的问题，默认为“详细描述图片内容”。
    """
    try:
        img = await qapi.get_image(file=file_name)
    except BaseException as e:
        logger.error(f"ask_image 图像获取失败 {e}")
        return "[error 图像打开失败]"
    fpath = img.file
    fpath = fpath.replace(
        "/app/.config/QQ/", "./data/napcat/config_qq/"
    )  # NOTE napcat dependent
    b64img = subprocess.run(
        ["sudo", "base64", fpath], capture_output=True, text=True
    ).stdout.strip()
    msg = HumanMessage(
        content=[
            {"type": "text", "text": prompt},
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{b64img}"},
            },
        ]
    )
    try:
        ret = await visual_model.ainvoke([msg])
    except BaseException as e:
        logger.error(f"模型调用出错 {e}")
        return "[error 模型调用出错]"
    return str(ret.content)


def pt(text: str):
    return [{"role": "user", "content": {"type": "text", "text": text}}]


INSTRUCTIONS = f"""
你可以用get_messages工具获取群消息。
你的账号是"{USR}"，昵称是"{NICK}"。消息记录里会出现你自己的消息，注意分别。
账号是用户的唯一标识，昵称可能存在重复，即不同的用户可以有相同的昵称，但不会有相同的账号。注意分别。

消息记录的格式：每行代表一条消息或一些指示， [on 日期 时间] 或 [on 时间] 和 [from 账号 (昵称)] 指示随后消息的发送时间和发送者，如果发送者是你自己则“from”后会附加“ME”。消息记录中可能会有未转义的中括号、换行符等，注意分别。
消息内容中有一些特殊元素，以左方括号和冒号为开始，以右方括号为结束，你在发送消息时也可以使用：
[:at 账号 (昵称)] 提及某人， [:at ALL] 提及群中所有人。如果是提及你的，则在“at”后会附加“ME”。你在发送消息时可以省略 (昵称) 部分，直接使用 [:at 账号] 。
[:refer 消息ID] 引用某条消息。此元素每条消息中只能使用最多1次，且应放在消息开头。使用get_messages_by_id工具查阅消息ID对应的消息内容及其上下文。使用get_messages工具的with_id参数查询消息ID。你在一般浏览消息记录时无需使用with_id参数，减小信息量。
[:face 表情名称] 平台专有表情符号，可用的表情名称有： {" ".join(CQFACE.values())} ，不要使用不存在的表情名称。通用emoji仍可直接使用。
[:image 文件名] 图像。文件名可用于ask_image工具参数。
[:unsupported] 暂时不支持解读的消息，等待后续升级。

如果你想要向群里发送消息，就使用send工具。发送的消息不会进行markdown渲染，不要使用markdown标记设置内容格式。
"""


@mcp.prompt()
async def instructions():
    return pt(INSTRUCTIONS)


@mcp.prompt()
async def events(timeout: str, select: str):
    tim = float(timeout)
    sel = select.split(",")
    intr = await wait_intr(time() + tim)  # ignoring timeout
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
    unread = await count_unread()
    return pt(f"未读消息计数: {unread}")


async def amain():
    qbot.run_backend()
    if port:
        await mcp.run_streamable_http_async()
    else:
        await mcp.run_stdio_async()


def main():
    asyncio.run(amain())


if __name__ == "__main__":
    main()
