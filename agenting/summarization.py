from langchain_core.messages import (
    AnyMessage,
    AIMessage,
    HumanMessage,
    RemoveMessage,
    SystemMessage,
)
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from langchain_core.language_models import LanguageModelLike
from langgraph.func import task

# REF: langchain.agents.middleware.summarization.DEFAULT_SUMMARY_PROMPT
SUMMARY_PROMPT = """
你已接近可接受的token总数上限，现在需要从交互历史中提取最优质/最相关的信息片段。
随后，这段上下文将覆盖上方的交互历史记录。因此，请确保提取的内容仅包含对实现核心目标最关键的信息。

你接下来的唯一目标，是从上方交互历史中提取最优质/最相关的上下文形成摘要文本。

当前交互历史将被你在此步骤提取的上下文所取代。鉴于此，你必须竭尽全力提取并记录交互历史中所有最关键的内容。
为避免重复已完成的操作，所提取的上下文应聚焦于对整体目标最重要的信息。

除开头的系统消息外，上方完整的交互历史记录是你需要提取上下文的来源。请仔细通读所有内容，深入思考哪些信息对实现整体目标最为关键。

请基于以上要求，仔细审阅整个交互历史，提取最重要且相关的上下文用以替换原有记录，从而释放交互历史的空间。
注意：仅输出提取的上下文摘要文本，不要添加任何额外信息或前后缀说明。
"""

# KEEP_MSGS = 4
# MSGS_THRESH = 10
KEEP_MSGS = 20
MSGS_THRESH = 50


@task
async def summarize(model: LanguageModelLike, msgs: list[AnyMessage]):
    if len(msgs) <= MSGS_THRESH:
        return None
    cut = 0
    for i in range(len(msgs) - KEEP_MSGS, -1, -1):
        msg = msgs[i]
        if isinstance(msg, AIMessage) and msg.tool_calls:
            continue
        else:
            cut = i
            break
    if cut <= 0:
        return None
    msgs_to_sum = msgs[:cut]
    msgs_to_keep = msgs[cut:]
    sum = await model.ainvoke(
        msgs_to_sum
        + [
            SystemMessage(SUMMARY_PROMPT),
            HumanMessage("输出提取的上下文摘要文本："),
        ]
    )
    return [
        RemoveMessage(REMOVE_ALL_MESSAGES),
        HumanMessage(content="之前的交互历史摘要：\n" + sum.text),  # type: ignore
    ] + msgs_to_keep
