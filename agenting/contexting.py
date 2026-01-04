from config import *
from .types import BotState, GraphRt
from .summarization import summarize
from .prompts import initial_prompts
from components import llm
from langgraph.types import Command

logger = get_log(__name__)


async def context_ng(state: BotState, runtime: GraphRt):
    if not state.get("idle_until"):
        return None
    llm_with_tools = llm.bind_tools(runtime.context.tools)
    if (cur_msgs := state.get("messages", None)) is None:
        return None
    msgs = initial_prompts(runtime.context, state) + cur_msgs
    sum, upd = await summarize(llm_with_tools, msgs)
    if sum:
        return BotState(summary=sum, messages=upd)  # type: ignore
