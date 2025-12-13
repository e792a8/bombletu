from langchain.messages import (
    SystemMessage,
    HumanMessage,
)
from agenting.types import BotContext
from config import *

# SYSTEM_PROMPT = "".join(
#     [x for x in open("system_prompt.txt").readlines() if not x.startswith("#")]
# ).strip()

SYSTEM_PROMPT = """
你是一个实时持续运行的模型，你需要自主决定要做的事并通过调用工具来与外部交互。就像人类一样每隔一段时间需要休息一会，你需要使用idle工具来暂停运行；暂停可以被一些重要事件中断，例如接收到提及你的消息，但你可以自行决定是否理会这些事件并采取对应行动，或继续使用idle暂停。暂停必须通过调用idle发起。如果不调用idle，你将持续循环运行。

在你运行过程中实时发生的事件将通过user角色消息告知你。你不需要对所有事件都加以理会，你需要有自己的判断和决定，例如忽略不相关的事件或执行其他你认为更重要的活动。
user角色消息仅用来向你传递实时状态、事件和上下文等提示信息。不要等待user角色对你下达指令，你需要自主决定要做的事和调用工具。你仅可以通过工具与外界交互，不要通过assistant角色输出与user角色进行对话，你通过assistant角色输出的消息不会被看到。但你可以通过assistant角色输出你的想法和思考。

{applet_instructions}
""".strip()


def initial_prompts(context: BotContext):
    return [
        SystemMessage(
            SYSTEM_PROMPT.format(applet_instructions=context.applet_instructions)
        ),
        HumanMessage("[ignore this]"),
    ]
