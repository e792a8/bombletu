from agenting.tools import LOCAL_TOOLS
from applet.base import BaseApplet
from agenting import (
    BotContext,
    BotState,
    make_graph,
)
from utils import get_date
import asyncio
from langchain_core.runnables import RunnableConfig
from langgraph.graph.state import CompiledStateGraph
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from time import time
import traceback
from config import *
from langfuse import get_client
from langfuse.langchain import CallbackHandler
from langchain_mcp_adapters.client import MultiServerMCPClient
from logging import getLogger

logger = getLogger(__name__)


class Agent:
    def __init__(self, applets: list[BaseApplet] = [], mcp_config: dict = {}) -> None:
        assert len(applets) <= 1, "TODO more applets"
        self.applets = applets
        self.mcp_client = MultiServerMCPClient(mcp_config)

    async def wait_events(self, until: float):
        # TODO real polling
        while True:
            timeout = max(0, until - time())
            ev = []
            for ap in self.applets:
                if e := await ap.poll_events(timeout):
                    ev.append(e)
            if len(ev) > 0:
                return "\n".join(ev)
            if time() > until:
                return None

    async def collect_status(self):
        col = []
        for a in self.applets:
            if st := await a.get_status():
                col.append(st)
        return "\n".join(col)

    async def collect_applet_instructions(self):
        col = []
        for a in self.applets:
            if ins := await a.get_instructions():
                col.append(ins)
        return "\n".join(col)

    async def collect_tools(self):
        tools = []
        for apt in self.applets:
            for t in await apt.get_tools():
                tools.append(t)
        return LOCAL_TOOLS + tools + await self.mcp_client.get_tools()

    async def agent_loop(
        self,
        graph: CompiledStateGraph[BotState, BotContext],
        agentconfig: RunnableConfig,
    ):
        langfuse = get_client()
        langfuse_handler = CallbackHandler()
        agentconfig = RunnableConfig(
            callbacks=[langfuse_handler],
            configurable={"thread_id": "1"},
        )
        resumed_state = await graph.aget_state(agentconfig)
        idle_until = resumed_state.values.get("idle_until")
        while True:
            if idle_until is not None:
                logger.info(f"agent idle until {get_date(idle_until)}")
            else:
                logger.info("agent continuing")
            intr = await self.wait_events(idle_until or 0)
            status = await self.collect_status()
            info_inject = [f"Now: {get_date()}"]
            if idle_until is not None:
                if intr:
                    info_inject.append("Idle: interrupted")
                else:
                    info_inject.append("Idle: finished")
            if intr:
                info_inject.append(f"{intr}")
            if status:
                info_inject.append(f"{status}")
            logger.info("agent invoking")
            applet_instructions = await self.collect_applet_instructions()
            with langfuse.start_as_current_observation(
                as_type="span",
                name="langchain-request",
                trace_context={"trace_id": langfuse.create_trace_id()},
            ) as span:
                info = "\n".join(info_inject)
                logger.info(f"info inject: {info}")
                span.update_trace(input=info)
                ret = await graph.ainvoke(
                    {"info_inject": info},
                    config=agentconfig,
                    context=BotContext(
                        applet_instructions=applet_instructions,
                        tools=self.tools,
                    ),  # type: ignore
                    print_mode="updates",
                )
                span.update_trace(output=ret)
            idle_until = ret.get("idle_until")
            # idle_mins = ret["structured_response"]["idle_minutes"]
            logger.debug(f"agent return: {ret}")

    async def run(self, alarm=None):
        get_client()
        langfuse_handler = CallbackHandler()
        retry_delay = 10
        graphconfig = RunnableConfig(
            callbacks=[langfuse_handler],
            configurable={"thread_id": "1"},
        )
        self.tools = await self.collect_tools()
        while True:
            await asyncio.sleep(10)
            logger.info("agent loop starting")
            try:
                async with AsyncSqliteSaver.from_conn_string(
                    DATADIR + "/ckpt/ckpt.sqlite"
                ) as ckptr:
                    (await ckptr.aget(graphconfig))
                    # ckptr = InMemorySaver()
                    graph = make_graph(self.tools, ckptr)
                    # agent = make_agent_deep(ckptr)
                    await self.agent_loop(graph, graphconfig)
                    retry_delay = max(10, retry_delay * 0.8)
            except BaseException as e:
                logger.error(f"agent loop exception: {traceback.format_exc()}")
                if alarm:
                    await alarm(e, retry_delay)
                await asyncio.sleep(retry_delay)
                retry_delay = min(1800, retry_delay * 2)
