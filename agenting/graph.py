from dataclasses import dataclass
from oicq.mcp import get_instructions
from time import time
from pydantic import BaseModel
from agenting.summarization import summarize
from config import *
from oicq.events import wait_events
from oicq.status import get_status
from utils import get_date
from .types import BotDeps, BotState
from .prompts import initial_prompts
from components import chat_model
from pydantic_graph import BaseNode, Graph
from pydantic_ai.direct import model_request
from pydantic_ai.models import ModelRequestParameters
from pydantic_ai import (
    ModelRequest,
    RunContext,
    RunUsage,
    ToolReturnPart,
)
from pydantic_ai.tools import ToolDefinition

logger = get_log(__name__)

BotNode = BaseNode[BotState, BotDeps, None]


@dataclass
class Idle(BotNode):
    until: float | None = None

    async def run(self, ctx) -> "Action":
        if self.until is not None:
            logger.info(f"agent idle until {get_date(self.until)}")
        else:
            logger.info("agent do not idle")
        intr = await wait_events(self.until or 0)
        status = await get_status() or ""
        info_inject = [f"Now: {get_date()}"]
        if self.until is not None:
            if intr:
                info_inject.append("Idle: interrupted")
            else:
                info_inject.append("Idle: finished")
        if intr:
            info_inject.append(f"{intr}")
        if status:
            info_inject.append(f"{status}")
        if ctx.state.tire_level > 10:
            info_inject.append("Hint: 你短时间内活动较密集，建议适时使用`idle`暂停")
        logger.info("idle end")
        ctx.deps.applet_instructions = await get_instructions()
        return Action(info_inject="\n".join(info_inject))


class Route(BaseModel):
    _name: str

    @classmethod
    def to_tool_def(cls) -> ToolDefinition:
        return ToolDefinition(
            name=cls._name,
            description=cls.__doc__,
            parameters_json_schema=cls.model_json_schema(),
        )


class IdleCall(BaseModel):
    """
    暂停一段时间，参数为分钟数。
    暂停可以被一些特别事件中断，使你提前恢复运行。
    """

    minutes: int


@dataclass
class Action(BotNode):
    info_inject: str | None = None

    async def run(self, ctx) -> "Idle | ContextNg":
        msgs = ctx.state.messages
        tool_ctx = RunContext(deps=ctx.deps, model=chat_model, usage=RunUsage())
        toolset = ctx.deps.toolset
        tool_defs = [t.tool_def for t in (await toolset.get_tools(tool_ctx)).values()]
        tool_defs.append(
            ToolDefinition(
                name="idle",
                description=IdleCall.__doc__,
                parameters_json_schema=IdleCall.model_json_schema(),
            )
        )
        if self.info_inject:
            msgs.append(ModelRequest.user_text_prompt(self.info_inject))
        msgs_send = initial_prompts(ctx.deps, ctx.state) + msgs
        resp = await model_request(
            chat_model,
            msgs_send,
            model_request_parameters=ModelRequestParameters(
                function_tools=tool_defs, output_mode="tool"
            ),
        )
        msgs.append(resp)
        ctx.state.messages = msgs
        tool_ctx.messages = msgs
        now = time()
        for call in resp.tool_calls:
            if call.tool_name == "idle":
                idle_call = IdleCall.model_validate(call.args_as_dict())
                idle_minutes = idle_call.minutes

                ctx.state.messages.append(
                    ModelRequest(
                        [
                            ToolReturnPart(
                                tool_name=call.tool_name,
                                tool_call_id=call.tool_call_id,
                                content=f"Idle {idle_call.minutes} minutes...",
                            )
                        ]
                    )
                )

                tire_level = ctx.state.tire_level
                if not idle_minutes:
                    tire_level += 1
                else:
                    tire_level /= max(1.2, idle_minutes / 2)
                ctx.state.tire_level = tire_level

                return ContextNg(now + 60 * idle_minutes, tool_defs)
            else:
                tools = await toolset.get_tools(tool_ctx)
                tool_res = await toolset.call_tool(
                    call.tool_name, call.args_as_dict(), tool_ctx, tools[call.tool_name]
                )
                ctx.state.messages.append(
                    ModelRequest(
                        [
                            ToolReturnPart(
                                tool_name=call.tool_name,
                                tool_call_id=call.tool_call_id,
                                content=tool_res,
                            )
                        ]
                    )
                )
        return Idle(None)


@dataclass
class ContextNg(BotNode):
    idle_until: float
    tool_defs: list[ToolDefinition]

    async def run(self, ctx) -> Idle:
        msgs_send = initial_prompts(ctx.deps, ctx.state) + ctx.state.messages
        sum, upd = await summarize(chat_model, self.tool_defs, msgs_send)
        if sum and upd:
            ctx.state.messages = upd
            ctx.state.summary = sum
        return Idle(self.idle_until)


graph = Graph(nodes=[Idle, Action, ContextNg])
