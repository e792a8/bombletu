from time import time
from ncatbot.core.api import NapCatAPIError
from logging import getLogger
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
import subprocess
from config import *
from .msgfmt import format_msg, parse_msg, msglfmt
from .globl import ChatTy, get_messages_wrapped, mcp, qapi
from .status import clear_unread, get_chats_info, set_group_active

logger = getLogger(__name__)


@mcp.tool()
async def get_chats():
    """
    查看所有可用的会话列表，及各个会话最近活跃时间、未读消息数量等信息。
    返回的每行代表一个聊天会话，带有由`[]`包围的多个字段，第一个字段为`[friend 用户ID (用户昵称)]`表示一个好友私信会话，或`[group 群ID (群名称)]`表示一个群组会话。第二个字段表示该会话最新消息的时间，如没有消息则省略。第三个字段表示该会话未读消息数量，如无未读消息则省略。
    """
    return await get_chats_info()


@mcp.tool()
async def send(chat_type: ChatTy, chat_id: str, content: str) -> str:
    """
    发送消息。
    参数:
        chat_type: 发送到的聊天会话类型，为"friend"代表好友私聊或"group"代表群聊。
        chat_id: 发送到的聊天会话ID，如chat_type="friend"则为用户ID，chat_type="group"则为群组ID。
        content: 发送的内容。
    发送的消息不会进行markdown渲染，不要使用markdown标记设置内容格式。
    向群组发送消息后进入该群组的观望状态，2分钟内该群的任意消息会将你的暂停唤醒。
    调用该工具发送消息时，对应聊天会话的未读消息计数值将清零。
    """
    logger.info(f"send: {content}")
    try:
        if chat_type == "group":
            await qapi.send_group_msg(chat_id, parse_msg(content).to_list())
        else:
            await qapi.send_private_msg(chat_id, parse_msg(content).to_list())
        await clear_unread(chat_type, chat_id)
        if chat_type == "group":
            await set_group_active(chat_id, time() + 60 * 2)
        return "Success."
    except NapCatAPIError as e:
        logger.warning(f"get_message error: {e}")
        return f"Error: 软件故障"


@mcp.tool()
async def get_messages(
    chat_type: ChatTy, chat_id: str, fro: int, to: int, with_id: bool = False
) -> str:
    """
    查阅消息记录。
    参数:
        chat_type: 查阅的聊天会话类型，为"friend"代表好友私聊或"group"代表群聊。
        chat_id: 查阅的聊天会话ID，如chat_type="friend"则为用户ID，chat_type="group"则为群组ID。
        fro: 消息序号区间的开始。
        to: 消息序号区间的结束。
        with_id: 是否附带消息ID。
    消息序号规则：最新的消息序号为1，序号由新到旧递增，返回的列表按由旧到新的顺序排列。
    例：`get_messages(chat_type="friend",chat_id="111",fro=10,to=1)`获取与好友"111"私聊会话的最新10条消息；`get_messages(chat_type="group",chat_id="222",fro=30,to=21)`获取群聊"222"的最后第30到第21条消息。
    参数with_id控制是否附带消息ID，如为真则每条消息的行首将带有 `[id 消息ID]` 指示。
    调用该工具读取到最新消息（即fro=1或to=1）时，该会话的未读消息计数将清零；如没有读取最新消息，则不影响未读消息计数。
    """
    if to > fro:
        fro, to = to, fro
    logger.info(f"get_messages: {fro}, {to}, {with_id}")
    if to == 1:
        await clear_unread(chat_type, chat_id)
    try:
        msgs = (await get_messages_wrapped(chat_type, chat_id, 0, fro))[: fro - to + 1]
        return await msglfmt(
            chat_type,
            chat_id,
            msgs,
            with_id,
        )
    except NapCatAPIError as e:
        logger.warning(f"get_message error: {e}")
        return f"Error: 软件故障"


@mcp.tool()
async def get_messages_by_id(
    chat_type: ChatTy, chat_id: str, message_id: str, before: int = 0, after: int = 0
) -> str:
    """
    按消息ID查阅消息记录。
    参数:
        chat_type: 查阅的聊天会话类型，为"friend"代表好友私聊或"group"代表群聊。
        chat_id: 查阅的聊天会话ID，如chat_type="friend"则为用户ID，chat_type="group"则为群组ID。
        message_id: 查阅的目标消息ID
        before: 在目标消息前附带的消息数量
        after: 在目标消息后附带的消息数量
    返回的消息列表中，目标消息的行首带有指示`[this]`，该条消息的ID即为查询的ID。
    调用该工具不会影响未读消息计数值。
    """
    try:
        bef = await get_messages_wrapped(
            chat_type, chat_id, message_id, before + 1, True
        )
        aft = await get_messages_wrapped(
            chat_type, chat_id, message_id, after + 1, False
        )
        lst = bef + aft[1:]
        return await msglfmt(chat_type, chat_id, lst, message_id)  # type: ignore
    except NapCatAPIError as e:
        logger.error(f"{e}")
        return "Error: 软件故障"


@mcp.tool()
async def unwrap_forward(forward_id: str) -> str:
    """
    展开查看`[:forward]`类型消息的详情。
    参数:
        forward_id: 转发ID。
    """
    try:
        msg = await qapi.get_forward_msg(forward_id)
    except Exception as e:
        logger.error(f"unwrap_forward: {e}")
        return f"Error: {e}"
    lns = []
    lns.append(f"[forward {forward_id}]")
    last_from = None
    for node in msg.content:
        cur_from = f"[from {node.user_id} ({node.nickname})]"
        if cur_from != last_from:
            lns.append(cur_from)
            last_from = cur_from
        lns.append(await format_msg(node.content))
    return "\n".join(lns)


@mcp.tool()
async def forward_messages(chat_type: ChatTy, chat_id: str, message_ids: list[str]):
    """
    将多条消息合并为一个消息列表并转发到目标聊天会话。
    参数：
        chat_type: 转发到的目标聊天会话类型，为"friend"代表好友私聊或"group"代表群聊。
        chat_id: 转发到的目标聊天会话ID，如chat_type="friend"则为用户ID，chat_type="group"则为群组ID。
        message_ids: 需要转发的消息的消息ID列表，使用`get_messages`的`with_id`参数获取。
    向群组转发消息后进入该群组的观望状态，2分钟内该群的任意消息会将你的暂停唤醒。
    调用该工具时，发送目标聊天会话的未读消息计数值将清零。
    """
    try:
        if chat_type == "group":
            await qapi.send_group_forward_msg_by_id(chat_id, message_ids)  # type: ignore
        else:
            await qapi.send_private_forward_msg_by_id(chat_id, message_ids)  # type: ignore
        await clear_unread(chat_type, chat_id)
        if chat_type == "group":
            await set_group_active(chat_id, time() + 60 * 2)
        return "Success."
    except Exception as e:
        logger.error(f"forward_messages: {e}")
        return f"Error: {e}"


visual_model = ChatOpenAI(
    temperature=0.6,
    model=ENV["VIS_MODEL"],
    api_key=ENV["VIS_API_KEY"],  # type:ignore
    base_url=ENV["VIS_BASE_URL"],
)


@mcp.tool()
async def ask_image(file_name: str, prompt="详细描述图片内容") -> str:
    """
    向视觉模型询问关于某个图像的问题。
    参数:
        file_name: 图像的文件名。
        prompt: 询问的问题，如不指定则默认为"详细描述图片内容"。
    """
    try:
        img = await qapi.get_image(file=file_name)
    except BaseException as e:
        logger.error(f"ask_image 图像获取失败 {e}")
        return "Error: 图像打开失败"
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
        return "Error: 模型调用出错"
    return ret.text
