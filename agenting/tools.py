from langchain.tools import tool
from config import *
from langchain.messages import ToolMessage
from langgraph.types import Command
from .types import ToolRt
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
async def edit_note(
    runtime: ToolRt,
    adds: list[str] | None = None,
    deletes: list[int] | None = None,
):
    """
    编辑笔记。笔记将常驻在你的上下文记录中，你可以使用笔记记录需要长期保留的信息，避免在你的上下文长度不足时遗忘。例如记录需要长期执行或在未来某个时间执行的行动。当笔记条目不再有效，记得将其删除。
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
                    "Success.",
                    tool_call_id=runtime.tool_call_id,
                )
            ],
            "notes": notes,
        },
    )


LOCAL_TOOLS = [
    idle,
    date,
    edit_note,
]
