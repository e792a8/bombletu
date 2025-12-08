from langgraph.prebuilt.tool_node import ToolNode
from langchain_core.tools import BaseTool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.base import BaseCheckpointSaver
from langchain.messages import (
    HumanMessage,
)
from typing import TYPE_CHECKING
from langgraph.graph import StateGraph
from config import *
from .types import BotContext, BotState, GraphRt
from .prompts import initial_prompts
from components import llm
from .contexting import context_ng

logger = get_log(__name__)


async def state_guard(state: BotState, runtime: GraphRt) -> BotState:
    return {
        "messages": [HumanMessage(state.get("info_inject"))],
        "info_inject": None,
        "idle_until": None,
    }


async def llm_call(state: BotState, runtime: GraphRt):
    """LLM decides whether to call a tool or not"""

    notes = [f"{i + 1} {n}" for i, n in enumerate(state.get("notes", []))]
    if len(notes) == 0:
        notes = "当前无笔记"
    else:
        notes = "\n".join(notes)
    notes_msg = HumanMessage(f'当前笔记（使用"edit_note"编辑笔记）：\n\n{notes}')

    llm_with_tools = llm.bind_tools(runtime.context.tools)
    msgs = state.get("messages", [])
    prompts = initial_prompts(runtime.context) + msgs[:-1] + [notes_msg] + msgs[-1:]
    return {
        "messages": [llm_with_tools.invoke(prompts)],
    }


def make_graph(
    tools: list[BaseTool],
    ckptr: BaseCheckpointSaver = InMemorySaver(),
    store_dir: str | None = None,
):
    # Build workflow
    builder = StateGraph(BotState, BotContext)

    # Add edges to connect nodes
    builder.add_sequence(
        [
            state_guard,
            llm_call,
            ("tool_node", ToolNode(tools)),
            context_ng,
        ]
    )
    builder.set_entry_point("state_guard")
    builder.set_finish_point("context_ng")

    # Compile the agent
    graph = builder.compile(checkpointer=ckptr)
    return graph


"""
def make_agent_deep(
    app: "App",
    ckptr: BaseCheckpointSaver = InMemorySaver(),
    store_dir: str | None = None,
):
    from deepagents import create_deep_agent
    from deepagents.backends import FilesystemBackend
    from langchain.agents.structured_output import ToolStrategy

    agent = create_deep_agent(
        llm,
        app.model_tools,
        backend=FilesystemBackend(DATADIR + "/agentfs"),
        system_prompt=SYSTEM_PROMPT,
        context_schema=BotContext,
        response_format=ToolStrategy(Idle),
    )
    return agent
"""

# agent = create_agent(
#     model=llm, tools=tools, system_prompt=SYSTEM_PROMPT, middleware=[after_model_do]
# )


def main():
    from ncatbot.utils.logger import setup_logging

    setup_logging()
    # cr = make_chroma("test")
    # cr.add_texts("我家在长春。")


if __name__ == "__main__":
    main()
