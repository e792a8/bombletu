from langchain.tools import tool
from datetime import datetime
import pytz
from langchain_core.runnables import RunnableConfig
from config import *
from os import environ
from ncatbot.core.api import BotAPI, NapCatAPIError
from msgfmt import msglfmt, parse_msg
from langchain_openai import ChatOpenAI
from langchain.messages import HumanMessage
import subprocess

logger = get_log(__name__)


def get_api(cfg: RunnableConfig) -> BotAPI:
    return cfg["configurable"]["qapi"]  # type: ignore


@tool
def date() -> str:
    """获取当前的本地日期和时间。"""
    return datetime.now(pytz.timezone(TZ)).isoformat(timespec="seconds")
    # return "2025-10-24T01:23:25+08:00"


@tool
async def send(cfg: RunnableConfig, content: str) -> str:
    """在群里发送消息。"""
    logger.info(f"send: {content}")
    try:
        await get_api(cfg).send_group_msg(GRP, parse_msg(content).to_list())
        return "[success]"
    except NapCatAPIError as e:
        logger.warning(f"get_message error: {e}")
        return "[error 软件暂时故障]"


@tool
async def get_unread(cfg: RunnableConfig, limit: int) -> str:
    """获取未读消息列表。
    参数limit表示限制返回的消息数量。
    返回的消息列表末尾带有指示 [unread 数量] 表示这些消息后剩余未读消息数量。
    """
    app = cfg["configurable"]["app"]  # type: ignore
    return await app.get_unread(limit)


@tool
async def get_messages(
    cfg: RunnableConfig, fro: int, to: int, with_id: bool = False
) -> str:
    """查阅消息记录。
    参数fro,to表示消息序号区间的开始和结束，最新的消息序号为1，序号由新到旧递增，返回的列表按由旧到新的顺序排列。
    例：get_messages(fro=10,to=1)获取最新10条消息；get_messages(fro=30,to=21)获取最后第30到第21条消息。
    参数with_id控制是否附带消息ID，如为真则每条消息的行首将带有 [id 消息ID] 指示。"""
    logger.info(f"get_messages: {fro}, {to}, {with_id}")
    try:
        return msglfmt(
            (await get_api(cfg).get_group_msg_history(GRP, 0, fro))[: fro - to + 1],
            with_id,
        )
    except NapCatAPIError as e:
        logger.warning(f"get_message error: {e}")
        return "[error 软件暂时故障]"


@tool
async def get_messages_by_id(
    cfg: RunnableConfig, id: str, before: int = 0, after: int = 0
) -> str:
    """按消息ID查阅消息记录。
    参数id为查阅的目标消息ID，before为在目标消息前附带的消息数量，after为在目标消息后附带的消息数量。
    返回的消息列表中，目标消息的行首带有指示 [this] ，该条消息的ID即为查询的ID。
    """
    api = get_api(cfg)
    try:
        bef = await api.get_group_msg_history(GRP, id, before + 1, True)
        aft = await api.get_group_msg_history(GRP, id, after + 1, False)
        lst = bef + aft[1:]
        return msglfmt(lst, id)
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
    cfg: RunnableConfig, file_name: str, prompt="群友发了这个图，什么意思？"
) -> str:
    """向视觉模型询问关于某个图像的问题。
    参数file_name是图像的文件名。参数prompt是询问的问题，默认为“群友发了这个图，什么意思？”。
    """
    try:
        img = await get_api(cfg).get_image(file=file_name)
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


ALL_TOOLS = [
    date,
    send,
    get_unread,
    get_messages,
    get_messages_by_id,
    ask_image,
]
