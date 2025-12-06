from langchain.tools import tool, ToolRuntime
from datetime import datetime
from langchain_core.tools import BaseTool, BaseToolkit
import pytz
from langchain_core.runnables import RunnableConfig
from config import *
from ncatbot.core.api import BotAPI, NapCatAPIError
from langchain_chroma import Chroma
from msgfmt import msglfmt, parse_msg
from langchain_openai import ChatOpenAI
from langchain.messages import HumanMessage, ToolMessage
from langgraph.types import Command
from langgraph.graph import END
import subprocess
from .types import BotContext, BotState, ToolRt
from utils import get_date
from time import time

logger = get_log(__name__)


@tool
def idle(runtime: ToolRt, minutes: int) -> Command:
    """暂停一段时间，参数为分钟数。
    暂停可以被一些特别事件中断，使你提前恢复运行。
    重要：必须单独调用，不可与其他工具并行调用。"""
    return Command(
        update={
            "messages": [
                ToolMessage(
                    f"Idle {minutes} minutes...",
                    tool_call_id=runtime.tool_call_id,
                )
            ],
            "idle_until": time() + minutes * 60,
        },
        # goto=END,
    )


@tool
def date() -> str:
    """获取当前的本地日期和时间。"""
    return get_date()
    # return "2025-10-24T01:23:25+08:00"


@tool
async def send(runtime: ToolRt, content: str) -> str:
    """在群里发送消息。"""
    logger.info(f"send: {content}")
    try:
        await runtime.context.app.qapi.send_group_msg(GRP, parse_msg(content).to_list())
        return "[success]"
    except NapCatAPIError as e:
        logger.warning(f"get_message error: {e}")
        return "[error 软件暂时故障]"


@tool
async def get_messages(
    runtime: ToolRt, fro: int, to: int, with_id: bool = False
) -> str:
    """查阅消息记录。
    参数fro,to表示消息序号区间的开始和结束，最新的消息序号为1，序号由新到旧递增，返回的列表按由旧到新的顺序排列。
    例：get_messages(fro=10,to=1)获取最新10条消息；get_messages(fro=30,to=21)获取最后第30到第21条消息。
    参数with_id控制是否附带消息ID，如为真则每条消息的行首将带有 [id 消息ID] 指示。
    调用该工具读取到最新消息（即fro=1或to=1）时，未读消息计数值将清零；如没有读取最新消息，则不影响未读消息计数值。"""
    if to > fro:
        fro, to = to, fro
    logger.info(f"get_messages: {fro}, {to}, {with_id}")
    if to == 1:
        await runtime.context.app.clear_unread()
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
    runtime: ToolRt, id: str, before: int = 0, after: int = 0
) -> str:
    """按消息ID查阅消息记录。
    参数id为查阅的目标消息ID，before为在目标消息前附带的消息数量，after为在目标消息后附带的消息数量。
    返回的消息列表中，目标消息的行首带有指示 [this] ，该条消息的ID即为查询的ID。
    调用该工具不会影响未读消息计数值。
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
    model=ENV["VIS_MODEL"],
    api_key=ENV["VIS_API_KEY"],  # type:ignore
    base_url=ENV["VIS_BASE_URL"],
)


@tool
async def ask_image(
    runtime: ToolRt,
    file_name: str,
    prompt="详细描述图片内容",
) -> str:
    """向视觉模型询问关于某个图像的问题。
    参数file_name是图像的文件名。参数prompt是询问的问题，默认为“详细描述图片内容”。
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
async def expand_message(runtime: ToolRt, message_id: str) -> str:
    """展开一条复合消息的内容。"""
    # TODO
    return ""


@tool
async def set_memory(runtime: ToolRt, content: str):
    """设置记忆内容，自动附加记录时间。
    会替换旧的记忆内容，因此需要注意将旧记忆中仍有需要的内容附加到新记忆中。
    """
    with open(DATADIR + "/memory.txt", "w") as f:
        f.write(f"记录时间：{get_date()}\n{content}")
    return "Success."


@tool
async def edit_note(
    runtime: ToolRt,
    adds: list[str] | None = None,
    deletes: list[int] | None = None,
):
    """
    编辑笔记。笔记将常驻在你的上下文记录中，你可以使用笔记记录需要长期保留的信息，避免在你的上下文长度不足时遗忘。例如记录需要在未来某个时间执行的行动。
    添加的笔记条目将按以下格式驻留在你的上下文中：

    编号 [添加时间] 条目内容

    其中"添加时间"由系统自动附注，你无需加在内容中。"编号"可用于"deletes"参数以删除笔记。

    参数：
        adds: 需要增加的笔记条目内容列表。
        deletes: 需要删除的笔记条目编号列表。
    deletes和adds参数可以只使用其中一个，也可以同时使用，方便批量操作。
    """
    notes = runtime.state.get("notes", [])
    if deletes:
        delset = sorted(set(deletes), reverse=True)
        for num in delset:
            if num > len(notes) or num < 1:
                return f"待删除的条目 {num} 不存在。笔记未修改。"
            notes.pop(num - 1)
    if adds:
        date = get_date()
        notes += [f"[{date}] {content}" for content in adds]
    return Command(
        update={
            "messages": [
                ToolMessage(
                    f"Success.",
                    tool_call_id=runtime.tool_call_id,
                )
            ],
            "notes": notes,
        },
    )


ALL_TOOLS = [
    idle,
    date,
    send,
    # get_unread,
    get_messages,
    get_messages_by_id,
    ask_image,
    edit_note,
    # store_memory,
    # query_memory,
    # delete_memory,
    # set_memory,
]
