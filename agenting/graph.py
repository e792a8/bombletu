from langgraph.prebuilt.tool_node import ToolNode
from langchain_core.tools import BaseTool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.base import BaseCheckpointSaver
from langchain.messages import (
    HumanMessage,
)
from langgraph.graph import StateGraph
from config import *
from .types import BotContext, BotState, GraphRt
from .prompts import initial_prompts
from components import llm
from .contexting import context_ng

logger = get_log(__name__)


async def state_preguard(state: BotState, runtime: GraphRt) -> BotState:
    return BotState(idle_minutes=None, idle_until=None)


async def llm_call(state: BotState, runtime: GraphRt):
    info_inject = state.get("info_inject") or ""
    tire_level = state.get("tire_level", 0)
    if tire_level > 10:
        info_inject += "\nHint: 你短时间内活动较密集，建议适时使用`idle`暂停"
    info_msg = HumanMessage(info_inject)

    notes = [f"{i + 1} {n}" for i, n in enumerate(state.get("notes", []))]
    if len(notes) == 0:
        notes = "当前无笔记"
    else:
        notes = "\n".join(notes)
    notes_msg = HumanMessage(
        f"当前你的笔记内容(按需使用`edit_note`编辑笔记):\n\n{notes}"
    )

    last_msg = HumanMessage("继续你接下来的行动:")

    llm_with_tools = llm.bind_tools(runtime.context.tools)
    msgs = state.get("messages", [])
    prompts_send = (
        initial_prompts(runtime.context) + msgs + [notes_msg, info_msg, last_msg]
    )

    llm_return = llm_with_tools.invoke(prompts_send)
    messages_update = [info_msg, llm_return]
    return BotState(messages=messages_update)


async def state_postguard(state: BotState, runtime: GraphRt):
    idle_minutes = state.get("idle_minutes")
    tire_level = state.get("tire_level", 0)
    if not idle_minutes:
        tire_level += 1
    else:
        tire_level /= max(1.2, idle_minutes / 2)
    update = BotState(info_inject=None, tire_level=tire_level)
    return update


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
            state_preguard,
            llm_call,
            ("tool_node", ToolNode(tools)),
            context_ng,
            state_postguard,
        ]
    )
    builder.set_entry_point("state_preguard")
    builder.set_finish_point("state_postguard")

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
