from dataclasses import dataclass
from langchain.tools import tool
from langgraph.prebuilt.tool_node import ToolNode
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import add_messages, MessagesState
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from langgraph.types import Command
from langchain_core.embeddings import Embeddings
from langchain_core.rate_limiters import InMemoryRateLimiter
from os import environ
from langchain.messages import (
    AnyMessage,
    SystemMessage,
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
from adapt import GiteeAIEmbeddings
from app import App
from mem0 import Memory
from .tools import ALL_TOOLS
from .types import BotContext, BotState

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
你比较低调，不喜欢高强度发送消息，主要以观察群友对话和了解群友为主。你的发言习惯和风格向群友学习，但保持一点简要、精练、冷幽默的个性。

你需要积极使用长期记忆工具 store_memory, query_memory, delete_memory 来记录、回想和维护你认为重要的信息，例如群友的相关情况、你与群友互动过程中令你印象深刻的事情等。

你可以用get_unread工具获取新接收的消息或用get_messages工具获取历史消息。
你的账号是“{USR}”，昵称是“{NICK}”，消息记录里会出现你自己的消息，注意分别。

消息记录的格式：每行代表一条消息或一些指示， [on 日期 时间] 或 [on 时间] 和 [from 账号 (昵称)] 指示随后消息的发送时间和发送者，如果发送者是你自己则“from”后会附加“ME”。消息记录中可能会有未转义的中括号、换行符等，注意分别。
消息内容中有一些特殊元素，以左方括号和冒号为开始，以右方括号为结束，你在发送消息时也可以使用：
[:at 账号 (昵称)] 提及某人， [:at ALL] 提及群中所有人。如果是提及你的，则在“at”后会附加“ME”。你在发送消息时可以省略 (昵称) 部分，直接使用 [:at 账号] 。
[:refer 消息ID] 引用某条消息。此元素每条消息中只能使用最多1次，且应放在消息开头。使用get_messages_by_id工具查阅消息ID对应的消息内容及其上下文。使用get_messages工具的with_id参数查询消息ID。你在一般浏览消息记录时无需使用with_id参数，减小信息量。
[:face 表情名称] 平台专有表情符号，可用的表情名称有： {' '.join(CQFACE.values())} 。通用emoji仍可直接使用。
[:image 文件名] 图像。文件名可用于ask_image工具参数。
[:unsupported] 暂时不支持解读的消息，等待后续升级。

如果你想要发送一些消息，就使用send工具。
当群里没有新消息，你可以浏览消息记录，了解群友。当你觉得无事可做，想等群里出现更多消息时，可以调用idle暂停一会。你的精力有限，连续进行10次操作左右，需要调用idle暂停几分钟。
暂停时间可以根据群活跃度动态调整，比如在你积极参与话题时可以缩短至1分至甚至0分，而如果一小时内只有两三条消息，则暂停时间可以逐渐延长到半小时至一小时。深夜可以延至更长。
参与讨论发言时注意话题时效，多注意最新的消息记录，不要对着久远之前的消息记录发不合时宜的回复。
你使用send工具发出的消息不会进行markdown渲染，不要试图使用markdown标记设置内容格式。
在你运行过程中实时发生的事件将通过user角色消息告知你，你并非必须理会，可以继续执行你正在做的事。
不要等待user角色对你下达指令，也不需要与user角色进行对话。你需要自主决定要做的事和调用工具。
""".strip()
# 与当前上下文相关的长期记忆也会通过user角色消息输入，供你参考。
# 现在你正在测试中，你需要直接执行：获取群里最新20条消息，然后暂停30分钟。
# 现在你正在进行测试，你要直接对群里最后第4条消息回复“测试”，然后暂停30分钟。
# 现在你正在测试中，接下来你要直接调用get_messages(fro=80,to=61)读取消息记录，对这段记录进行总结，调用send将总结的内容发出，然后循环进行：调用idle(minutes=1)暂停1分钟，之后判断暂停是正常结束还是被事件中断，将你的判断用send发出。
# 初始时你精力足够，请你直接开始进行操作。


# model = ChatOllama(
#     model="qwen3:0.6b", base_url="http://192.168.66.1:11434", reasoning=True
# )
# model = init_chat_model("anthropic:claude-sonnet-4-5", temperature=0)
llm = ChatOpenAI(
    rate_limiter=InMemoryRateLimiter(requests_per_second=1),
    max_retries=30,
    temperature=0.6,
    model=environ["LLM_MODEL"],
    api_key=environ["LLM_API_KEY"],  # type:ignore
    base_url=environ["LLM_BASE_URL"],
)


tools_by_name = {tool.name: tool for tool in ALL_TOOLS}

model_with_tools = llm.bind_tools(ALL_TOOLS)


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


async def llm_call(state: BotState):
    """LLM decides whether to call a tool or not"""

    return {
        "messages": [model_with_tools.invoke(INITIAL_PROMPTS + state["messages"])],
    }


embed = GiteeAIEmbeddings(
    dimensions=1024,
    model=environ["EMBED_MODEL"],
    base_url=environ["EMBED_BASE_URL"],
    api_key=environ["EMBED_API_KEY"],  # type: ignore
)


def make_chroma(col: str, persist_dir: str | None = None):
    # embed = OpenAIEmbeddings(
    return Chroma(col, embedding_function=embed, persist_directory=persist_dir)


def make_mem0() -> Memory:
    mem0_config = {
        "vector_store": {
            "provider": "langchain",
            "config": {"client": make_chroma("mem0", DATADIR + "/chroma")},
        },
        "llm": {"provider": "langchain", "config": {"model": llm}},
        "embedder": {"provider": "langchain", "config": {"model": embed}},
        "reranker": {
            "provider": "llm_reranker",
            "config": {
                "llm": {"provider": "langchain", "config": {"model": llm}},
            },
        },
    }
    mem = Memory.from_config(mem0_config)
    return mem


def make_agent(
    ckptr: BaseCheckpointSaver = InMemorySaver(), store_dir: str | None = None
):
    # Build workflow
    builder = StateGraph(BotState, BotContext)

    # Add nodes
    builder.add_node("context_reduce", context_reduce)
    builder.add_node("llm_call", llm_call)  # type: ignore
    builder.add_node("tool_node", ToolNode(ALL_TOOLS))  # type: ignore

    # Add edges to connect nodes
    builder.add_edge(START, "context_reduce")
    builder.add_edge("context_reduce", "llm_call")
    builder.add_edge("llm_call", "tool_node")
    builder.add_edge("tool_node", END)

    # Compile the agent
    agent = builder.compile(checkpointer=ckptr)
    return agent


agent = make_agent()
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
