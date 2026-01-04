from langchain_core.messages import (
    AnyMessage,
    AIMessage,
    HumanMessage,
    RemoveMessage,
    SystemMessage,
    ToolMessage,
)
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from langchain_core.language_models import LanguageModelLike
from langgraph.func import task
from components import langfuse


KEEP_ROUNDS = 4


@task
async def summarize(model: LanguageModelLike, msgs: list[AnyMessage]):
    reduce_prompt = langfuse.get_prompt("context-reduce").prompt
    cut = 0
    tail_rounds = 0
    for i in range(len(msgs) - 2, -1, -1):
        if (
            isinstance(msgs[i], AIMessage) or isinstance(msgs[i], ToolMessage)
        ) and isinstance(msgs[i + 1], HumanMessage):
            tail_rounds += 1
            if tail_rounds >= KEEP_ROUNDS:
                cut = i + 1
                break
    if cut <= 2:
        return None, None
    msgs_to_sum = msgs[:cut]
    msgs_to_keep = msgs[cut:]
    sum = await model.ainvoke(
        msgs_to_sum
        + [
            SystemMessage(reduce_prompt),
            HumanMessage("输出提取的交互上下文摘要:"),
        ]
    )
    return (
        sum.text,  # type: ignore
        [
            RemoveMessage(REMOVE_ALL_MESSAGES),
        ]
        + msgs_to_keep,
    )
