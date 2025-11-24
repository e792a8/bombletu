from typing import List
from ncatbot.core import BotClient, MessageArray
from ncatbot.core.event import GroupMessageEvent
from asyncio_channel import create_channel, create_sliding_buffer
from agenting import BotContext, BotState, make_agent, make_chroma, make_agent_deep
from utils import get_date
import asyncio
from datetime import datetime
from pytz import timezone
from ncatbot.core.api import NapCatAPIError
import os, signal
from langchain.messages import AIMessage, ToolMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langchain_chroma import Chroma
from langgraph.graph.state import CompiledStateGraph
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.checkpoint.memory import InMemorySaver
from time import time
from msgfmt import msglfmt, parse_msg
import traceback
from config import *
from app import App
from langfuse import get_client
from langfuse.langchain import CallbackHandler

logger = get_log(__name__)


def check_idle_call(ivk):
    if (
        "messages" in ivk
        and len(ivk["messages"]) > 0
        and isinstance(last_msg := ivk["messages"][-1], AIMessage)
        and len(last_msg.tool_calls) == 1
        and (call := last_msg.tool_calls[0])["name"] == "idle"
    ):
        call_id = call["id"]
        mins = int(call["args"]["minutes"])
        return call_id, mins
    return None, 0


async def agent_loop(
    app: App,
    agent: CompiledStateGraph[BotState, BotContext],
    agentconfig: RunnableConfig,
):
    langfuse = get_client()
    idle_mins = None
    while True:
        if idle_mins is not None:
            logger.info(f"agent sleeping {idle_mins}min")
        else:
            logger.info(f"agent continuing")
        intr = await app.wait_intr(idle_mins or 0)
        unread = await app.count_unread()
        info_inject = [f"[now {get_date()}]"]
        msg_inject = []
        if idle_mins is not None:
            if intr:
                info_inject.append("[idle interrupted]")
            else:
                info_inject.append("[idle finished]")
        if intr:
            info_inject.append(f"[notify {intr}]")
        info_inject.append(f"[status 未读消息计数: {unread}]")
        if len(info_inject) > 0:
            msg_inject.append(HumanMessage("\n".join(info_inject)))
        logger.info("agent invoking")
        with langfuse.start_as_current_observation(
            as_type="span",
            name="langchain-request",
            trace_context={"trace_id": langfuse.create_trace_id()},
        ) as span:
            span.update_trace(input=msg_inject)
            ret = await agent.ainvoke(
                {"messages": msg_inject, "idle_minutes": None},
                # {"messages": msg_inject},
                config=agentconfig,
                context=BotContext(app),  # type: ignore
                print_mode="updates",
            )
            span.update_trace(output=ret)
        idle_mins = ret["idle_minutes"]
        # idle_mins = ret["structured_response"]["idle_minutes"]
        logger.debug(f"agent return: {ret}")


def make_agent_loop(app: App):
    async def agent_loop_wrapper(_):
        langfuse = get_client()
        langfuse_handler = CallbackHandler()
        retry_delay = 10
        while True:
            await asyncio.sleep(10)
            logger.info("agent loop starting")
            try:
                # async with AsyncSqliteSaver.from_conn_string(
                #     DATADIR + "/ckpt.sqlite"
                # ) as ckptr:
                ckptr = InMemorySaver()
                chroma = make_chroma(GRP, DATADIR + "/chroma")
                agent = make_agent(ckptr)
                # agent = make_agent_deep(ckptr)
                agentconfig = RunnableConfig(
                    callbacks=[langfuse_handler],
                    configurable={
                        "thread_id": "1",
                        "app": app,
                        "qapi": app.qbot.api,
                        "chroma": chroma,
                    },
                )
                await agent_loop(app, agent, agentconfig)
                retry_delay = max(10, retry_delay * 0.8)
            except BaseException as e:
                logger.error(f"agent loop exception: {traceback.format_exc()}")
                await asyncio.sleep(1)
                await app.qbot.api.send_group_text(
                    GRP,
                    "Someone tell [CQ:at,qq=1571224208] there is a problem with my AI.",
                )
                await asyncio.sleep(1)
                await app.qbot.api.send_group_text(
                    CON, f"{GRP} {traceback.format_exc()} delay: {retry_delay}"
                )
                await asyncio.sleep(retry_delay)
                retry_delay = min(1800, retry_delay * 2)

    return agent_loop_wrapper


def main():
    App(make_agent_loop, make_chroma(GRP, DATADIR + "/chroma")).run()


if __name__ == "__main__":
    main()
