from config import *
from .types import BotState, GraphRt
from .summarization import summarize
from .prompts import initial_prompts
from components import llm

logger = get_log(__name__)


async def context_ng(state: BotState, runtime: GraphRt):
    llm_with_tools = llm.bind_tools(runtime.context.tools)
    if (cur_msgs := state.get("messages", None)) is None:
        return None
    msgs = initial_prompts(runtime.context) + cur_msgs
    sum = await summarize(llm_with_tools, msgs)
    if sum:
        return {"messages": sum}
