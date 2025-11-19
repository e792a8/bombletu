from langchain.tools import tool, ToolRuntime
from datetime import datetime
from langchain_core.tools import BaseTool, BaseToolkit
import pytz
from langchain_core.runnables import RunnableConfig
from config import *
from os import environ
from ncatbot.core.api import BotAPI, NapCatAPIError
from langchain_chroma import Chroma
from msgfmt import msglfmt, parse_msg
from langchain_openai import ChatOpenAI
from langchain.messages import HumanMessage
import subprocess
from app import App
from .types import BotContext, BotState

logger = get_log(__name__)

Rt = ToolRuntime[BotContext, BotState]
# NOTE langgraph 1.0.3, ToolRuntime inject only supports:
# argument name being exactly "runtime", or argument type being `ToolRuntime`
# without type arguments.


@tool
def date() -> str:
    """获取当前的本地日期和时间。"""
    return datetime.now(pytz.timezone(TZ)).isoformat(timespec="seconds")
    # return "2025-10-24T01:23:25+08:00"


@tool
async def send(runtime: Rt, content: str) -> str:
    """在群里发送消息。"""
    logger.info(f"send: {content}")
    try:
        await runtime.context.app.qapi.send_group_msg(GRP, parse_msg(content).to_list())
        return "[success]"
    except NapCatAPIError as e:
        logger.warning(f"get_message error: {e}")
        return "[error 软件暂时故障]"


@tool
async def get_unread(runtime: Rt, limit: int) -> str:
    """获取未读消息列表。
    参数limit表示限制返回的消息数量。
    返回的消息列表末尾带有指示 [unread 数量] 表示这些消息后剩余未读消息数量。
    """
    app = runtime.context.app  # type: ignore
    return await app.get_unread(limit)


@tool
async def get_messages(runtime: Rt, fro: int, to: int, with_id: bool = False) -> str:
    """查阅消息记录。
    参数fro,to表示消息序号区间的开始和结束，最新的消息序号为1，序号由新到旧递增，返回的列表按由旧到新的顺序排列。
    例：get_messages(fro=10,to=1)获取最新10条消息；get_messages(fro=30,to=21)获取最后第30到第21条消息。
    参数with_id控制是否附带消息ID，如为真则每条消息的行首将带有 [id 消息ID] 指示。"""
    logger.info(f"get_messages: {fro}, {to}, {with_id}")
    try:
        return await msglfmt(
            (await runtime.context.app.qapi.get_group_msg_history(GRP, 0, fro))[
                : fro - to + 1
            ],
            with_id,
            runtime.context.app.qapi,
        )
    except NapCatAPIError as e:
        logger.warning(f"get_message error: {e}")
        return "[error 软件暂时故障]"


@tool
async def get_messages_by_id(
    runtime: Rt, id: str, before: int = 0, after: int = 0
) -> str:
    """按消息ID查阅消息记录。
    参数id为查阅的目标消息ID，before为在目标消息前附带的消息数量，after为在目标消息后附带的消息数量。
    返回的消息列表中，目标消息的行首带有指示 [this] ，该条消息的ID即为查询的ID。
    """
    api = runtime.context.app.qapi
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
    model=environ["VIS_MODEL"],
    api_key=environ["VIS_API_KEY"],  # type:ignore
    base_url=environ["VIS_BASE_URL"],
)


@tool
async def ask_image(
    runtime: Rt,
    file_name: str,
    prompt="群友发了这个图，什么意思？",
) -> str:
    """向视觉模型询问关于某个图像的问题。
    参数file_name是图像的文件名。参数prompt是询问的问题，默认为“群友发了这个图，什么意思？”。
    """
    try:
        img = await runtime.context.app.qapi.get_image(file=file_name)
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


@tool
async def store_memory(runtime: Rt, contents: str) -> str:
    """存入记忆。
    参数contents是记忆内容，每行将作为单独一项条目存入记忆。
    返回存入条目对应的条目ID，每行一个。"""
    cr = runtime.context.app.chroma
    ids = await cr.aadd_texts(contents.split("\n"))
    logger.info(f"store memory: {ids}")
    return "\n".join(ids)


@tool
async def query_memory(
    runtime: Rt, query: str, k: int = 4, with_id: bool = False
) -> str:
    """查询记忆。
    参数query是查询目标。k为返回的条目个数，默认为4。with_id表示返回时是否附带记忆ID。
    返回格式：每行一个条目，如果with_id为真，则每个条目开头附带 [id 记忆ID] 。"""
    cr = runtime.context.app.chroma
    res = await cr.asimilarity_search(query, k=k)
    ret = "\n".join(
        map(lambda x: (f"[id {x.id}]" if with_id else "") + x.page_content, res)
    )
    return ret


@tool
async def delete_memory(runtime: Rt, ids: str) -> str:
    """删除记忆条目。
    参数ids为要删除的条目ID列表，每行一个。条目ID可使用query_memory工具的with_id参数获取。"""
    idl = ids.split("\n")
    cr = runtime.context.app.chroma
    await cr.adelete(idl)
    return "[success]"


@tool
async def expand_message(runtime: Rt, message_id: str) -> str:
    """展开一条复合消息的内容。"""
    # TODO
    return ""


ALL_TOOLS = [
    date,
    send,
    get_unread,
    get_messages,
    get_messages_by_id,
    ask_image,
    store_memory,
    query_memory,
    delete_memory,
]
