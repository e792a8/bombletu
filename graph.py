from langchain.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.runtime import Runtime
from langgraph.graph import add_messages
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from os import environ
from datetime import datetime
import pytz
from langchain.messages import (
    AnyMessage,
    SystemMessage,
    ToolMessage,
    AIMessage,
    HumanMessage,
    RemoveMessage,
)
from typing_extensions import TypedDict, Annotated
from typing import Literal
from langgraph.graph import StateGraph, START, END
from langchain_core.runnables import RunnableConfig
from config import *
from cqface import CQFACE

logger = get_log(__name__)

# SYSTEM_PROMPT = "".join(
#     [x for x in open("system_prompt.txt").readlines() if not x.startswith("#")]
# ).strip()

# 你暂时不太清楚群友喜欢什么样的话题，且最好给其他群友留一些话题机会，减少频繁在群里发送消息。
# 你说话喜欢冷幽默。
# 每条消息以一个字典表示，其中'from'为发送人信息，'user'和'nick'分别为发送人账号和昵称；消息的'content'和'date'字段为消息内容和发送时间。
# 你可以使用get_unread和get_messages工具获取群消息。
SYSTEM_PROMPT = f"""
你是一个闲聊群里的群友，日常生活是来群里看看其他群友都在聊些啥，偶尔掺和两句。
你比较低调，且对群友尚不太熟悉，频繁向群友直接发问也比较打扰，因此你不会频繁发送消息，主要以观察群友对话和了解群友为主。你希望你的发言以精练而带有一点冷幽默为宜，偶尔假装不经意地扮演一只可爱的猫娘。

你可以用get_unread工具获取新接收的消息或用get_messages工具获取历史消息。
你的账号是“{USR}”，昵称是“{NICK}”，消息记录里会出现你自己的消息，注意分别。

消息记录的格式：每行代表一条消息或一些指示， [on 日期 时间] 或 [on 时间] 和 [from 账号 (昵称)] 指示随后消息的发送时间和发送者，如果发送者是你自己则“from”后会附加“ME”。消息记录中可能会有未转义的中括号、换行符等，注意分别。
消息内容中有一些特殊元素，你在发送消息时也可以使用：
[:at 账号] 提及某人， [:at ALL] 提及群中所有人。如果是提及你的，则在“at”后会附加“ME”。
[:refer 消息ID] 引用某条消息。此元素每条消息中只能使用最多1次，且应放在消息开头。使用get_messages_by_id工具查阅消息ID对应的消息内容及其上下文。使用get_messages工具的with_id参数查询消息ID。你在一般浏览消息记录时无需使用with_id参数，减小信息量。
[:face 表情名称] 平台专有表情符号。可用的表情名称有： {' '.join(CQFACE.values())}
[:unsupported] 暂时不支持解读的消息，等待后续升级。

如果你想要发送一些消息，就使用send工具。
当群里没有新消息，你可以浏览消息记录，了解群友。当你觉得无事可做，想等群里出现更多消息时，可以调用idle暂停一会。你的精力有限，连续进行10次操作左右，需要调用idle暂停几分钟。
暂停时间可以根据群活跃度动态调整，比如在你积极参与话题时可以缩短至1分至甚至0分，而如果一小时内只有两三条消息，则暂停时间可以逐渐延长到半小时至一小时。深夜可以延至更长。
在你运行过程中实时发生的事件将通过user角色消息告知你，你并非必须理会，可以继续执行你正在做的事。
不要等待user角色对你下达指令，也不需要与user角色进行对话。你需要自己调用工具和决定要做的事。
""".strip()
# 现在你正在测试中，你需要直接执行的任务是：最新的第13-9条消息中有一条回复了其他消息，被回复的这条消息的前2条消息的内容是什么？使用send工具发出你的回答，然后暂停30分钟。
# 现在你正在进行测试，你要直接对群里最后第4条消息回复“测试”，然后暂停30分钟。
# 现在你正在测试中，接下来你要直接调用get_messages(fro=80,to=61)读取消息记录，对这段记录进行总结，调用send将总结的内容发出，然后循环进行：调用idle(minutes=1)暂停1分钟，之后判断暂停是正常结束还是被事件中断，将你的判断用send发出。
# 初始时你精力足够，请你直接开始进行操作。


def get_app():
    return config["configurable"]["app"]  # type: ignore


@tool
def date() -> str:
    """获取当前的本地日期和时间。"""
    return datetime.now(pytz.timezone(TZ)).isoformat(timespec="seconds")
    # return "2025-10-24T01:23:25+08:00"


@tool
async def send(cfg: RunnableConfig, content: str) -> str:  # FIXME when #6318 is ok
    """在群里发送消息。"""
    app = cfg["configurable"]["app"]  # type: ignore
    return await app.send(content)


@tool
def idle(minutes: int) -> int:
    """暂停一段时间，参数为分钟数。
    暂停可以被一些特别事件中断，使你提前恢复运行。
    重要：必须单独调用，不可与其他工具并行调用。"""
    return minutes


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
) -> str:  # FIXME when #6318 is ok
    """查阅消息记录。
    参数fro,to表示消息序号区间的开始和结束，最新的消息序号为1，序号由新到旧递增，返回的列表按由旧到新的顺序排列。
    例：get_messages(fro=10,to=1)获取最新10条消息；get_messages(fro=30,to=21)获取最后第30到第21条消息。
    参数with_id控制是否附带消息ID，如为真则每条消息的行首将带有 [id 消息ID] 指示。"""
    app = cfg["configurable"]["app"]  # type: ignore
    return await app.get_messages(fro, to, with_id)
    # bigmodel.cn传dict会报错，ai.gitee.com就没事，最好兼一下


@tool
async def get_messages_by_id(
    cfg: RunnableConfig, id: str, before: int = 0, after: int = 0
) -> str:
    """按消息ID查阅消息记录。
    参数id为查阅的目标消息ID，before为在目标消息前附带的消息数量，after为在目标消息后附带的消息数量。
    返回的消息列表中，目标消息的行首带有指示 [this] ，该条消息的ID即为查询的ID。
    """
    app = cfg["configurable"]["app"]  # type: ignore
    return await app.get_messages_by_id(id, before, after)


# model = ChatOllama(
#     model="qwen3:0.6b", base_url="http://192.168.66.1:11434", reasoning=True
# )
# model = init_chat_model("anthropic:claude-sonnet-4-5", temperature=0)
llm = ChatOpenAI(
    temperature=0.6,
    model=environ["LLM_MODEL"],
    api_key=environ["LLM_API_KEY"],  # type:ignore
    base_url=environ["LLM_BASE_URL"],
)


tools = [
    idle,
    date,
    send,
    get_unread,
    get_messages,
    get_messages_by_id,
]
tools_by_name = {tool.name: tool for tool in tools}

model_with_tools = llm.bind_tools(tools)


class BotState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]


INITIAL_PROMPTS = [
    SystemMessage(SYSTEM_PROMPT),
    HumanMessage(
        "忽略这句话，继续执行你的操作。"
    ),  # bigmodel.cn非得这里有些字，ai.gitee.com不用
]


async def context_reduce(state: BotState):
    msgs = state["messages"]
    if len(msgs) < 28:
        return None
    new_msgs = msgs[-18:]
    return {"messages": [RemoveMessage(REMOVE_ALL_MESSAGES), *new_msgs]}


async def llm_call(state: dict):
    """LLM decides whether to call a tool or not"""

    return {
        "messages": [model_with_tools.invoke(INITIAL_PROMPTS + state["messages"])],
    }


async def tool_node(state: dict, config: RunnableConfig, runtime: Runtime):
    """Performs the tool call"""

    result = []
    for tool_call in state["messages"][-1].tool_calls:
        tool = tools_by_name[tool_call["name"]]
        # rt = ToolRuntime() # BUG: hand-made langgraph does not support ToolRuntime injection. langgraph#6318
        # args = tool_call["args"]
        # rt = ToolRuntime(
        #     state=state,
        #     context=runtime.context,
        #     stream_writer=runtime.stream_writer,
        #     tool_call_id=tool_call["id"],
        #     config=config,
        #     store=runtime.store,
        # )
        # args["rt"] = rt
        observation = await tool.ainvoke(tool_call["args"], config)
        result.append(ToolMessage(observation, tool_call_id=tool_call["id"]))
    return {"messages": result}


async def inform_event(state: BotState):
    msgs = []
    if intr := await config["configurable"]["app"].wait_intr(0):  # type: ignore
        msgs.append(HumanMessage(f"[notify {intr}]"))
    if col := await config["configurable"]["app"].collect_unread():  # type: ignore
        msgs.append(HumanMessage(f"[event 收到{col}条新消息]"))
    return {"messages": msgs}


async def should_continue(state: BotState) -> Literal["tool_node", END]:  # type: ignore
    """Decide if we should continue the loop or stop based upon whether the LLM made a tool call and whether it is a call to idle."""

    last_msg = state["messages"][-1]

    # If the LLM makes a call to idle, then go to END
    if (
        isinstance(last_msg, AIMessage)
        and len(last_msg.tool_calls) == 1
        and tools_by_name[last_msg.tool_calls[0]["name"]] is idle
    ):
        return END

    # Otherwise, continue loop
    return "tool_node"


def make_agent():
    ckptr = InMemorySaver()

    # Build workflow
    builder = StateGraph(BotState)

    # Add nodes
    builder.add_node("context_reduce", context_reduce)
    builder.add_node("llm_call", llm_call)  # type: ignore
    builder.add_node("tool_node", tool_node)  # type: ignore

    # Add edges to connect nodes
    builder.add_edge(START, "context_reduce")
    builder.add_edge("context_reduce", "llm_call")
    builder.add_conditional_edges("llm_call", should_continue, ["tool_node", END])
    builder.add_edge("tool_node", END)

    # Compile the agent
    agent = builder.compile(checkpointer=ckptr)
    return agent


agent = make_agent()
# agent = create_agent(
#     model=llm, tools=tools, system_prompt=SYSTEM_PROMPT, middleware=[after_model_do]
# )


def main():
    config = RunnableConfig(configurable={"thread_id": 1, "app": None})
    resp = agent.invoke(
        {},  # type: ignore
        config=config,
        print_mode="values",
    )
    print(resp)


if __name__ == "__main__":
    main()
